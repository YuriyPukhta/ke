"""
Fine-tune ASRCNN (TMA) so soft attn matches Kokoro duration_proj hard path.

  .venv\\Scripts\\python.exe -m gt_cache.tma_finetune.train --epochs 10 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gt_cache.tma_finetune import DEFAULT_CKPT_DIR, DEFAULT_DATA_DIR  # noqa: E402
from gt_cache.tma_finetune.common import (  # noqa: E402
    duration_corr,
    forward_soft_attn,
    load_asr_trainable,
    save_asr_checkpoint,
    soft_attn_nll,
    soft_to_durs,
)
from gt_cache.tma_finetune.dataset import TmaKokoroDurDataset  # noqa: E402


def collate_one(batch: list[dict]) -> dict:
    """batch_size must be 1 (ASR S2S is sequential / variable length)."""
    assert len(batch) == 1
    return batch[0]


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    losses = []
    corrs = []
    for item in loader:
        mel = item["mel"].to(device)
        text_ids = item["text_ids"].to(device)
        hard = item["hard"].to(device)
        soft = forward_soft_attn(model, mel, text_ids)
        n = min(soft.shape[1], hard.shape[0])
        t = min(soft.shape[2], hard.shape[1])
        soft_c = soft[:, :n, :t]
        hard_c = hard[:n, :t].unsqueeze(0)
        losses.append(float(soft_attn_nll(soft_c, hard_c).item()))
        d_pred = soft_to_durs(soft_c)
        d_tgt = [max(1, int(round(float(x)))) for x in item["d_tgt"][:n].tolist()]
        corrs.append(duration_corr(d_pred, d_tgt))
    model.train()
    valid = [c for c in corrs if c == c]
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "corr_dur_mean": sum(valid) / max(1, len(valid)) if valid else float("nan"),
        "n": len(corrs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune TMA on Kokoro durations")
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out", type=str, default=str(DEFAULT_CKPT_DIR))
    ap.add_argument("--init-weights", type=str, default="")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--max-mel", type=int, default=600)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1)
    args = ap.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = TmaKokoroDurDataset(data_root, "train", max_mel=args.max_mel)
    val_ds = TmaKokoroDurDataset(data_root, "val", max_mel=args.max_mel)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_one)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_one)

    init = Path(args.init_weights) if args.init_weights else None
    model = load_asr_trainable(device, weights=init)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_corr = -1.0
    best_path = out_dir / "tma_kokoro_best.pth"
    last_path = out_dir / "tma_kokoro_last.pth"

    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)} "
        f"epochs={args.epochs} lr={args.lr} max_mel={args.max_mel}",
        flush=True,
    )

    base = evaluate(model, val_loader, device)
    print(
        f"baseline val loss={base['loss']:.4f} corr_dur={base['corr_dur_mean']:.4f}",
        flush=True,
    )
    history.append({"epoch": 0, **base, "split": "val_baseline"})

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for item in train_loader:
            mel = item["mel"].to(device)
            text_ids = item["text_ids"].to(device)
            hard = item["hard"].to(device)
            soft = forward_soft_attn(model, mel, text_ids)
            n = min(soft.shape[1], hard.shape[0])
            t = min(soft.shape[2], hard.shape[1])
            loss = soft_attn_nll(soft[:, :n, :t], hard[:n, :t].unsqueeze(0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(1, n_batches)
        row = {"epoch": epoch, "train_loss": train_loss, "elapsed_s": round(time.time() - t0, 1)}
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            val = evaluate(model, val_loader, device)
            row["val_loss"] = val["loss"]
            row["val_corr_dur"] = val["corr_dur_mean"]
            print(
                f"epoch {epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={val['loss']:.4f} corr_dur={val['corr_dur_mean']:.4f}",
                flush=True,
            )
            meta = {"epoch": epoch, "val": val, "args": vars(args)}
            save_asr_checkpoint(model, last_path, meta=meta)
            if val["corr_dur_mean"] == val["corr_dur_mean"] and val["corr_dur_mean"] > best_corr:
                best_corr = val["corr_dur_mean"]
                save_asr_checkpoint(model, best_path, meta=meta)
                print(f"  new best corr_dur={best_corr:.4f}", flush=True)
            elif epoch % args.save_every == 0:
                save_asr_checkpoint(model, out_dir / f"tma_kokoro_ep{epoch:03d}.pth", meta=meta)
        else:
            print(f"epoch {epoch:03d} train_loss={train_loss:.4f}", flush=True)
        history.append(row)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done best_corr_dur={best_corr:.4f} -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
