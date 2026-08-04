"""
Fine-tune JDCNet so mel(Kokoro wav) → F0Ntrain (Kokoro units).

Usage:
  .venv\\Scripts\\python.exe -m gt_cache.jdc_finetune.train --epochs 25 --batch-size 12 --accum-steps 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gt_cache.jdc_finetune import DEFAULT_CKPT_DIR, DEFAULT_DATA_DIR  # noqa: E402
from gt_cache.jdc_finetune.common import (  # noqa: E402
    corr_voiced,
    load_jdc_trainable,
    save_jdc_checkpoint,
)
from gt_cache.jdc_finetune.dataset import (  # noqa: E402
    JdcKokoroF0Dataset,
    LengthBucketBatchSampler,
    collate_pad,
)


def f0_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    voiced_thresh: float = 50.0,
    voiced_weight: float = 2.0,
) -> torch.Tensor:
    """Smooth L1 on valid frames; upweight Kokoro-voiced regions."""
    w = torch.ones_like(target)
    w = torch.where(target > voiced_thresh, w * voiced_weight, w)
    w = w * mask.float()
    per = F.smooth_l1_loss(pred, target, reduction="none")
    denom = w.sum().clamp_min(1.0)
    return (per * w).sum() / denom


@torch.no_grad()
def evaluate(model, loader, device: torch.device, voiced_thresh: float) -> dict:
    model.eval()
    losses = []
    corrs = []
    for batch in loader:
        mel = batch["mel"].to(device)
        f0 = batch["f0"].to(device)
        mask = batch["mask"].to(device)
        pred, _, _ = model(mel.unsqueeze(1))
        pred = pred.float()
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        t = min(pred.shape[-1], f0.shape[-1], mask.shape[-1])
        pred = pred[..., :t]
        f0_t = f0[..., :t]
        mask_t = mask[..., :t]
        losses.append(float(f0_loss(pred, f0_t, mask_t, voiced_thresh=voiced_thresh).item()))
        for i in range(pred.shape[0]):
            m = mask_t[i]
            corrs.append(corr_voiced(pred[i][m], f0_t[i][m], voiced_thresh=voiced_thresh))
    model.train()
    valid = [c for c in corrs if c == c]
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "corr_voiced_mean": sum(valid) / max(1, len(valid)) if valid else float("nan"),
        "n": len(corrs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune JDCNet on Kokoro F0Ntrain")
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out", type=str, default=str(DEFAULT_CKPT_DIR))
    ap.add_argument("--init-weights", type=str, default="", help="Optional .t7/.pt; default ST2 bst.t7")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument(
        "--accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation; effective batch ~= batch-size * accum-steps",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--voiced-thresh", type=float, default=50.0)
    ap.add_argument("--voiced-weight", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=400, help="Center-crop long clips (0=off)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=1)
    args = ap.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = JdcKokoroF0Dataset(data_root, "train", max_frames=args.max_frames)
    val_ds = JdcKokoroF0Dataset(data_root, "val", max_frames=args.max_frames)
    train_sampler = LengthBucketBatchSampler(train_ds.lengths(), args.batch_size, seed=0)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_pad,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pad,
    )

    init = Path(args.init_weights) if args.init_weights else None
    model = load_jdc_trainable(device, weights=init)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_corr = -1.0
    best_path = out_dir / "jdc_kokoro_best.t7"
    last_path = out_dir / "jdc_kokoro_last.t7"
    eff_bs = args.batch_size * max(1, args.accum_steps)

    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)} "
        f"epochs={args.epochs} bs={args.batch_size} accum={args.accum_steps} "
        f"eff_bs={eff_bs} max_frames={args.max_frames} lr={args.lr}",
        flush=True,
    )

    base = evaluate(model, val_loader, device, args.voiced_thresh)
    print(f"baseline val loss={base['loss']:.4f} corr_v={base['corr_voiced_mean']:.4f}", flush=True)
    history.append({"epoch": 0, **base, "split": "val_baseline"})

    t0 = time.time()
    accum = max(1, args.accum_steps)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            mel = batch["mel"].to(device)
            f0 = batch["f0"].to(device)
            mask = batch["mask"].to(device)
            pred, _, _ = model(mel.unsqueeze(1))
            pred = pred.float()
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            t = min(pred.shape[-1], f0.shape[-1], mask.shape[-1])
            loss = f0_loss(
                pred[..., :t],
                f0[..., :t],
                mask[..., :t],
                voiced_thresh=args.voiced_thresh,
                voiced_weight=args.voiced_weight,
            )
            (loss / accum).backward()
            running += float(loss.item())
            n_batches += 1
            if step % accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

        train_loss = running / max(1, n_batches)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "elapsed_s": round(time.time() - t0, 1),
        }
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            val = evaluate(model, val_loader, device, args.voiced_thresh)
            row["val_loss"] = val["loss"]
            row["val_corr_voiced"] = val["corr_voiced_mean"]
            print(
                f"epoch {epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={val['loss']:.4f} corr_v={val['corr_voiced_mean']:.4f}",
                flush=True,
            )
            meta = {"epoch": epoch, "val": val, "args": vars(args)}
            save_jdc_checkpoint(model, last_path, meta=meta)
            if val["corr_voiced_mean"] == val["corr_voiced_mean"] and val["corr_voiced_mean"] > best_corr:
                best_corr = val["corr_voiced_mean"]
                save_jdc_checkpoint(model, best_path, meta=meta)
                print(f"  new best corr_v={best_corr:.4f}", flush=True)
            elif epoch % args.save_every == 0:
                save_jdc_checkpoint(model, out_dir / f"jdc_kokoro_ep{epoch:03d}.t7", meta=meta)
        else:
            print(f"epoch {epoch:03d} train_loss={train_loss:.4f}", flush=True)
            save_jdc_checkpoint(model, last_path, meta={"epoch": epoch, "args": vars(args)})
        history.append(row)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done best_corr_v={best_corr:.4f} -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
