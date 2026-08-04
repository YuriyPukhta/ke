"""
PCA + PSO voice similarity (not emotion).

Modes:
  --target prosody  freeze timbre; PCA/PSO on [128:] only (default)
  --target full     PCA/PSO on entire ref_s[256], expand to pack (Fabul8-like)

Unlike Fabul8: PSO in PCA space (no top-dim enum, no finite-diff GD).
Unlike kokoro_hack: target = match speaker WAV (mel / WavLM).

  .venv\\Scripts\\python.exe prosody_steer/optimize_pca_pso.py --target full --speaker 6209
  .venv\\Scripts\\python.exe prosody_steer/optimize_pca_pso.py --target prosody --speaker 6209 \\
      --timbre-ckpt timbre_encoder/runs/speaker_ref_s_6209_bankmean_norm/timbre.pt
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import (  # noqa: E402
    assemble_pack_both,
    build_alignment,
    kokoro_ids,
    load_speaker_rows,
    load_timbre_ckpt,
    load_wav,
    make_ref_s,
    pack_slot,
    write_listen,
)
from timbre_encoder.voice_bank import CORE_VOICES, US_VOICES  # noqa: E402
from timbre_encoder.voice_pack import assemble_voice_pack  # noqa: E402


def collect_style_matrix(
    pipeline: KPipeline,
    names: list[str],
    frame_idx: int,
    target: str,
) -> tuple[np.ndarray, list[str]]:
    """[n_voices, D] — D=128 prosody or D=256 full frame."""
    rows: list[np.ndarray] = []
    used: list[str] = []
    for name in names:
        try:
            pack = pipeline.load_voice(name)
        except Exception as e:
            print(f"skip {name}: {e}", flush=True)
            continue
        if pack.dim() == 2:
            pack = pack.unsqueeze(1)
        idx = min(frame_idx, pack.shape[0] - 1)
        frame = pack[idx, 0].detach().float().cpu().numpy()  # [256]
        rows.append(frame[128:] if target == "prosody" else frame)
        used.append(name)
    if len(rows) < 3:
        raise SystemExit(f"Need >=3 voices for PCA, got {len(rows)}")
    return np.stack(rows, axis=0), used


class StylePCA:
    """PCA on centered official-voice style vectors (128 or 256)."""

    def __init__(self, X: np.ndarray, n_components: int):
        self.dim = X.shape[1]
        self.mean = X.mean(axis=0)
        centered = X - self.mean
        n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(centered)
        coords = self.pca.transform(centered)
        self.data_min = coords.min(axis=0)
        self.data_max = coords.max(axis=0)
        self.n_components = n_comp
        self.explained = float(self.pca.explained_variance_ratio_.sum())

    def decode(self, z: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(z, torch.Tensor):
            z_np = z.detach().float().cpu().numpy()
            device = z.device
        else:
            z_np = np.asarray(z, dtype=np.float64)
            device = torch.device("cpu")
        vec = self.pca.inverse_transform(z_np.reshape(1, -1))[0] + self.mean
        return torch.from_numpy(vec.astype(np.float32)).to(device)

    def bounds(self, exaggerate: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        lo = self.data_min * exaggerate
        hi = self.data_max * exaggerate
        swap = lo > hi
        lo[swap], hi[swap] = hi[swap], lo[swap]
        pad = 0.05 * (np.abs(hi - lo) + 1e-3)
        return lo - pad, hi + pad


def assemble_expand(frame256: torch.Tensor) -> torch.Tensor:
    f = frame256.detach().float().cpu().view(-1)
    if f.numel() != 256:
        raise ValueError(f"Expected 256, got {f.numel()}")
    return f.view(1, 1, 256).expand(510, 1, 256).contiguous().clone()


@torch.no_grad()
def clip_loss_full(
    model: KModel,
    row: dict,
    ref_s: torch.Tensor,
    device: torch.device,
    max_wav_sec: float,
    lambda_mel: float,
    lambda_slm: float,
    slm,
) -> float | None:
    phonemes = (row.get("phonemes_st2") or "").strip()
    d_gt = parse_d_gt(row.get("d_gt"))
    if not phonemes or not d_gt:
        return None
    input_ids, ids, dropped = kokoro_ids(model, phonemes, device)
    if dropped or len(ids) != len(phonemes):
        return None
    aln = build_alignment(d_gt, device)
    if aln.shape[1] != input_ids.shape[1]:
        return None
    try:
        wave = load_wav(row["audio_path"], max_wav_sec).to(device)
    except Exception:
        return None
    if wave.numel() < SAMPLE_RATE // 4:
        return None

    rs = ref_s.view(1, -1) if ref_s.dim() == 1 else ref_s
    audio_pred, _, _, _ = model.forward_with_alignment(input_ids, rs, aln)
    if audio_pred.dim() == 0 or audio_pred.numel() < 100:
        return None

    loss = 0.0
    if lambda_mel > 0:
        pred_mel = waveform_to_mel(
            audio_pred.unsqueeze(0) if audio_pred.dim() == 1 else audio_pred,
            device=device,
        )
        tgt_mel = waveform_to_mel(wave.unsqueeze(0), device=device)
        t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
        if t < 4:
            return None
        loss += lambda_mel * float(F.l1_loss(pred_mel[..., :t], tgt_mel[..., :t]).item())

    if slm is not None and lambda_slm > 0:
        pred_w = audio_pred.view(1, -1) if audio_pred.dim() == 1 else audio_pred
        real_w = wave.view(1, -1)
        tw = min(pred_w.shape[-1], real_w.shape[-1])
        sc = slm.similarity(pred_w[..., :tw], real_w[..., :tw])
        loss += lambda_slm * (float(sc["wavlm_l1"]) + (1.0 - float(sc["wavlm_cos"])))
    return loss


@torch.no_grad()
def clip_loss_prosody(
    model: KModel,
    row: dict,
    timbre: torch.Tensor,
    prosody: torch.Tensor,
    device: torch.device,
    max_wav_sec: float,
    lambda_mel: float,
    lambda_slm: float,
    slm,
    mode: str,
    baseline_pack: torch.Tensor | None,
) -> float | None:
    phonemes = (row.get("phonemes_st2") or "").strip()
    d_gt = parse_d_gt(row.get("d_gt"))
    if not phonemes or not d_gt:
        return None
    input_ids, ids, dropped = kokoro_ids(model, phonemes, device)
    if dropped or len(ids) != len(phonemes):
        return None
    aln = build_alignment(d_gt, device)
    if aln.shape[1] != input_ids.shape[1]:
        return None
    try:
        wave = load_wav(row["audio_path"], max_wav_sec).to(device)
    except Exception:
        return None
    if wave.numel() < SAMPLE_RATE // 4:
        return None

    if mode == "delta":
        assert baseline_pack is not None
        slot = pack_slot(baseline_pack, len(phonemes))
        ref_s = make_ref_s(timbre, slot[0, 128:].detach() + prosody)
    else:
        ref_s = make_ref_s(timbre, prosody)

    audio_pred, _, _, _ = model.forward_with_alignment(input_ids, ref_s, aln)
    if audio_pred.dim() == 0 or audio_pred.numel() < 100:
        return None

    loss = 0.0
    if lambda_mel > 0:
        pred_mel = waveform_to_mel(
            audio_pred.unsqueeze(0) if audio_pred.dim() == 1 else audio_pred,
            device=device,
        )
        tgt_mel = waveform_to_mel(wave.unsqueeze(0), device=device)
        t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
        if t < 4:
            return None
        loss += lambda_mel * float(F.l1_loss(pred_mel[..., :t], tgt_mel[..., :t]).item())

    if slm is not None and lambda_slm > 0:
        pred_w = audio_pred.view(1, -1) if audio_pred.dim() == 1 else audio_pred
        real_w = wave.view(1, -1)
        tw = min(pred_w.shape[-1], real_w.shape[-1])
        sc = slm.similarity(pred_w[..., :tw], real_w[..., :tw])
        loss += lambda_slm * (float(sc["wavlm_l1"]) + (1.0 - float(sc["wavlm_cos"])))
    return loss


def batch_loss(
    *,
    target: str,
    model: KModel,
    rows: list[dict],
    device: torch.device,
    max_wav_sec: float,
    lambda_mel: float,
    lambda_slm: float,
    slm,
    eval_clips: int,
    rng: random.Random,
    # full
    ref_s: torch.Tensor | None = None,
    # prosody
    timbre: torch.Tensor | None = None,
    prosody: torch.Tensor | None = None,
    prosody_mode: str = "delta",
    baseline_pack: torch.Tensor | None = None,
) -> float:
    sample = rows if eval_clips <= 0 or eval_clips >= len(rows) else rng.sample(rows, eval_clips)
    vals: list[float] = []
    for row in sample:
        if target == "full":
            assert ref_s is not None
            v = clip_loss_full(
                model, row, ref_s, device, max_wav_sec, lambda_mel, lambda_slm, slm
            )
        else:
            assert timbre is not None and prosody is not None
            v = clip_loss_prosody(
                model,
                row,
                timbre,
                prosody,
                device,
                max_wav_sec,
                lambda_mel,
                lambda_slm,
                slm,
                prosody_mode,
                baseline_pack,
            )
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser(description="PCA+PSO voice similarity")
    ap.add_argument("--target", choices=("prosody", "full"), default="full")
    ap.add_argument("--speaker", type=str, default="6209")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument(
        "--timbre-ckpt",
        type=str,
        default=str(ROOT / "timbre_encoder/runs/speaker_ref_s_6209_bankmean_norm/timbre.pt"),
        help="Only used with --target prosody",
    )
    ap.add_argument("--voice-set", choices=("us", "core"), default="us")
    ap.add_argument("--pca-frame", type=int, default=50)
    ap.add_argument("--pca-components", type=int, default=19)
    ap.add_argument(
        "--prosody-mode",
        choices=("replace", "delta"),
        default="delta",
        help="Only for --target prosody",
    )
    ap.add_argument("--baseline", type=str, default="af_heart")
    ap.add_argument("--exaggerate", type=float, default=2.0)
    ap.add_argument("--max-clips", type=int, default=8)
    ap.add_argument("--max-phonemes", type=int, default=80)
    ap.add_argument("--max-wav-sec", type=float, default=6.0)
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-particles", type=int, default=20)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--eval-clips", type=int, default=0)
    ap.add_argument("--w", type=float, default=0.72)
    ap.add_argument("--c1", type=float, default=1.49)
    ap.add_argument("--c2", type=float, default=1.49)
    ap.add_argument("--vel-clip-frac", type=float, default=0.25)
    ap.add_argument("--early-stopping", type=int, default=12)
    ap.add_argument("--lambda-mel", type=float, default=1.0)
    ap.add_argument("--lambda-slm", type=float, default=0.0)
    ap.add_argument("--slm-device", type=str, default="cpu")
    ap.add_argument("--slm-layers", type=str, default="4,8,12")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    tag = "full" if args.target == "full" else "prosody"
    out_dir = Path(args.out_dir) if args.out_dir else (
        HERE / "runs" / f"speaker_pca_pso_{tag}_{args.speaker}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} target={args.target} speaker={args.speaker} "
        f"pca_k={args.pca_components} particles={args.n_particles} iters={args.iters}",
        flush=True,
    )

    slm = None
    if args.lambda_slm > 0:
        from timbre_encoder.slm_loss import WavLMFeatureLoss

        layers = [int(x) for x in args.slm_layers.split(",") if x.strip()]
        print(f"Loading WavLM on {args.slm_device} layers={layers}...", flush=True)
        slm = WavLMFeatureLoss(layers=layers, device=args.slm_device)

    pipeline = KPipeline(lang_code="a", model=False)
    names = US_VOICES if args.voice_set == "us" else CORE_VOICES
    X, used = collect_style_matrix(pipeline, names, args.pca_frame, args.target)
    pca_sys = StylePCA(X, args.pca_components)
    lo, hi = pca_sys.bounds(args.exaggerate)
    span = np.maximum(hi - lo, 1e-3)
    print(
        f"PCA voices={len(used)} dim={pca_sys.dim} k={pca_sys.n_components} "
        f"explained={pca_sys.explained:.4f}",
        flush=True,
    )

    baseline_pack = pipeline.load_voice(args.baseline).to(device)
    if baseline_pack.dim() == 2:
        baseline_pack = baseline_pack.unsqueeze(1)

    timbre: torch.Tensor | None = None
    if args.target == "prosody":
        timbre = load_timbre_ckpt(Path(args.timbre_ckpt), device).detach()
        print(f"timbre frozen |t|={timbre.norm().item():.4f}", flush=True)

    rows = load_speaker_rows(
        Path(args.manifest),
        args.speaker,
        max_phonemes=args.max_phonemes,
        max_clips=args.max_clips + args.holdout,
        seed=args.seed,
    )
    if len(rows) < args.holdout + 1:
        raise SystemExit(f"Need >= {args.holdout + 1} clips, got {len(rows)}")
    holdout = rows[: args.holdout]
    train_rows = rows[args.holdout :][: args.max_clips]
    print(f"train_clips={len(train_rows)} holdout={len(holdout)}", flush=True)

    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dim = pca_sys.n_components
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    rng = random.Random(args.seed)

    lo_t = torch.tensor(lo, device=device, dtype=torch.float32)
    hi_t = torch.tensor(hi, device=device, dtype=torch.float32)
    pos = lo_t + (hi_t - lo_t) * torch.rand(args.n_particles, dim, device=device, generator=g)
    pos[0].zero_()
    vel = torch.zeros_like(pos)
    vel_clip = torch.tensor(args.vel_clip_frac * span, device=device, dtype=torch.float32)

    mean_t = torch.from_numpy(pca_sys.mean.astype(np.float32)).to(device)

    def z_to_style(z: torch.Tensor) -> torch.Tensor:
        """Absolute decoded vector (128 or 256)."""
        return pca_sys.decode(z)

    def z_to_prosody_part(z: torch.Tensor) -> torch.Tensor:
        abs_p = z_to_style(z)
        if args.prosody_mode == "replace":
            return abs_p
        return abs_p - mean_t

    def eval_z(z: torch.Tensor) -> float:
        if args.target == "full":
            return batch_loss(
                target="full",
                model=model,
                rows=train_rows,
                device=device,
                max_wav_sec=args.max_wav_sec,
                lambda_mel=args.lambda_mel,
                lambda_slm=args.lambda_slm,
                slm=slm,
                eval_clips=args.eval_clips,
                rng=rng,
                ref_s=z_to_style(z),
            )
        return batch_loss(
            target="prosody",
            model=model,
            rows=train_rows,
            device=device,
            max_wav_sec=args.max_wav_sec,
            lambda_mel=args.lambda_mel,
            lambda_slm=args.lambda_slm,
            slm=slm,
            eval_clips=args.eval_clips,
            rng=rng,
            timbre=timbre,
            prosody=z_to_prosody_part(z),
            prosody_mode=args.prosody_mode,
            baseline_pack=baseline_pack if args.prosody_mode == "delta" else None,
        )

    zero_loss = eval_z(torch.zeros(dim, device=device))
    print(f"[PCA-PSO] zero-z loss={zero_loss:.6f}", flush=True)

    pbest_pos = pos.clone()
    pbest_loss = torch.full((args.n_particles,), float("inf"), device=device)
    gbest_pos = torch.zeros(dim, device=device)
    gbest_loss = float("inf")
    no_improve = 0
    history: list[dict] = []

    try:
        for it in range(args.iters):
            losses = [eval_z(pos[p]) for p in range(args.n_particles)]
            losses_t = torch.tensor(losses, device=device)
            improved = losses_t < pbest_loss
            pbest_loss = torch.where(improved, losses_t, pbest_loss)
            pbest_pos[improved] = pos[improved]

            min_idx = int(torch.argmin(pbest_loss).item())
            current = float(pbest_loss[min_idx].item())
            if current < gbest_loss:
                gbest_loss = current
                gbest_pos = pbest_pos[min_idx].clone()
                no_improve = 0
            else:
                no_improve += 1

            history.append(
                {"iter": it, "gbest_loss": gbest_loss, "mean_pbest": float(pbest_loss.mean())}
            )
            print(
                f"[PCA-PSO] iter={it:03d} gbest={gbest_loss:.6f} "
                f"mean_pbest={history[-1]['mean_pbest']:.6f} no_improve={no_improve}",
                flush=True,
            )
            if no_improve >= args.early_stopping:
                print(f"[PCA-PSO] early stop at iter={it}", flush=True)
                break

            r1 = torch.rand(args.n_particles, dim, device=device, generator=g)
            r2 = torch.rand(args.n_particles, dim, device=device, generator=g)
            vel = (
                args.w * vel
                + args.c1 * r1 * (pbest_pos - pos)
                + args.c2 * r2 * (gbest_pos.unsqueeze(0) - pos)
            )
            vel = torch.max(torch.min(vel, vel_clip), -vel_clip)
            pos = torch.clamp(pos + vel, lo_t, hi_t)
    except KeyboardInterrupt:
        print("\n[PCA-PSO] interrupted; saving best", flush=True)

    style_best = z_to_style(gbest_pos)
    if args.target == "full":
        final_loss = batch_loss(
            target="full",
            model=model,
            rows=train_rows,
            device=device,
            max_wav_sec=args.max_wav_sec,
            lambda_mel=args.lambda_mel,
            lambda_slm=args.lambda_slm,
            slm=slm,
            eval_clips=0,
            rng=rng,
            ref_s=style_best,
        )
        frame = style_best.detach()
        timbre_out = frame[:128]
        prosody_out = frame[128:]
        pack = assemble_expand(frame)
        print(
            f"[PCA-PSO] done full-vector zero={zero_loss:.6f} gbest={gbest_loss:.6f} "
            f"full={final_loss:.6f} |z|={gbest_pos.norm().item():.4f} "
            f"|frame|={frame.norm().item():.4f}",
            flush=True,
        )
        torch.save(frame.cpu(), out_dir / "frame256.pt")
        torch.save(timbre_out.cpu(), out_dir / "timbre.pt")
        torch.save(prosody_out.cpu(), out_dir / "prosody.pt")
        listen_timbre = timbre_out
        mid = pack_slot(baseline_pack, args.pca_frame)[0, 128:].detach()
        listen_delta = prosody_out - mid
    else:
        prosody_vec = z_to_prosody_part(gbest_pos)
        final_loss = batch_loss(
            target="prosody",
            model=model,
            rows=train_rows,
            device=device,
            max_wav_sec=args.max_wav_sec,
            lambda_mel=args.lambda_mel,
            lambda_slm=args.lambda_slm,
            slm=slm,
            eval_clips=0,
            rng=rng,
            timbre=timbre,
            prosody=prosody_vec,
            prosody_mode=args.prosody_mode,
            baseline_pack=baseline_pack if args.prosody_mode == "delta" else None,
        )
        assert timbre is not None
        timbre_out = timbre
        print(
            f"[PCA-PSO] done prosody zero={zero_loss:.6f} gbest={gbest_loss:.6f} "
            f"full={final_loss:.6f} |z|={gbest_pos.norm().item():.4f}",
            flush=True,
        )
        torch.save(timbre_out.detach().cpu(), out_dir / "timbre.pt")
        prosody_cpu = prosody_vec.detach().cpu()
        torch.save(
            prosody_cpu,
            out_dir / ("prosody_delta.pt" if args.prosody_mode == "delta" else "prosody.pt"),
        )
        if args.prosody_mode == "delta":
            pack = assemble_pack_both(
                timbre_out.cpu(), prosody_cpu, baseline_pack.cpu(), mode="delta"
            )
        else:
            pack = assemble_pack_both(
                timbre_out.cpu(), prosody_cpu, baseline_pack.cpu(), mode="replace"
            )
        listen_timbre = timbre_out
        if args.prosody_mode == "delta":
            listen_delta = prosody_vec
        else:
            mid = pack_slot(baseline_pack, args.pca_frame)[0, 128:].detach()
            listen_delta = prosody_vec - mid

    torch.save(gbest_pos.detach().cpu(), out_dir / "pca_coords.pt")
    torch.save(pack, out_dir / "voice_pack.pt")
    with open(out_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(
            {
                "mean": pca_sys.mean,
                "pca": pca_sys.pca,
                "data_min": pca_sys.data_min,
                "data_max": pca_sys.data_max,
                "voices": used,
                "frame_idx": args.pca_frame,
                "target": args.target,
                "dim": pca_sys.dim,
            },
            f,
        )
    summary = {
        "speaker": args.speaker,
        "method": "pca_pso",
        "target": args.target,
        "prosody_mode": args.prosody_mode if args.target == "prosody" else None,
        "timbre_ckpt": args.timbre_ckpt if args.target == "prosody" else None,
        "pca_dim": pca_sys.dim,
        "pca_components": pca_sys.n_components,
        "pca_explained": pca_sys.explained,
        "zero_loss": zero_loss,
        "gbest_loss": gbest_loss,
        "gbest_full_loss": final_loss,
        "train_clips": len(train_rows),
        "iters_ran": len(history),
        "voices_for_pca": used,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({**summary, "history": history, "args": vars(args)}, out_dir / "style.pt")

    write_listen(
        out_dir,
        model,
        baseline_pack,
        listen_timbre,
        listen_delta,
        holdout,
        device,
        args.max_wav_sec,
        title=f"Speaker {args.speaker} PCA+PSO target={args.target}",
        prosody_mode="shared",
        slm_scorer=slm,
    )
    # Also dump expand-pack free synth note in summary if full
    if args.target == "full":
        # Control pack: optimized timbre + baseline length prosody
        ctrl = assemble_voice_pack(listen_timbre.cpu(), baseline_pack.cpu())
        torch.save(ctrl, out_dir / "voice_pack_timbre_plus_baseline_p.pt")

    print(f"Saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
