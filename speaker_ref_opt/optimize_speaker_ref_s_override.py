"""
Optimize Kokoro timbre style[:128] with forced d_gt + cached F0/N overrides.

Duration, pitch and energy come from cache (not from style predictors).
Decoder sees: alignment(d_gt), f0_override, n_override, trainable timbre.
Prosody half is frozen baseline (unused by decoder when F0/N are overridden).

  .venv\\Scripts\\python.exe speaker_ref_opt/optimize_speaker_ref_s_override.py \\
      --speaker 6209 --baseline mean --voice-bank us --lr 3e-3 --steps 200 --batch-size 8
"""
from __future__ import annotations

import argparse
import csv
import html as html_lib
import random
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(HERE))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from gt_cache import DEFAULT_FEATURES_ROOT, feature_path, load_curve  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import (  # noqa: E402
    calibrate_st2_to_kokoro,
    kokoro_f0n_for_ref,
)
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.voice_bank import (  # noqa: E402
    CORE_VOICES,
    US_VOICES,
    build_voice_bank,
    load_bank,
    normalize_to_bank,
    save_bank,
)
from timbre_encoder.voice_pack import assemble_voice_pack  # noqa: E402

from optimize_speaker_ref_s import (  # noqa: E402
    build_alignment,
    kokoro_ids,
    load_wav,
    make_ref_s,
    pack_slot,
)


def load_speaker_rows_f0n(
    csv_path: Path,
    speaker: str,
    max_phonemes: int,
    max_clips: int,
    seed: int,
    features_root: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("speaker_id") != speaker:
                continue
            if row.get("align_status") != "ok":
                continue
            if row.get("f0_status") != "ok" or row.get("n_status") != "ok":
                continue
            ph_st2 = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            if not ph_st2 or not d_gt or len(d_gt) != len(ph_st2):
                continue
            if not (2 <= len(ph_st2) <= max_phonemes):
                continue
            if not feature_path(row["audio_path"], "f0", features_root).is_file():
                continue
            if not feature_path(row["audio_path"], "n", features_root).is_file():
                continue
            rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_clips > 0 and len(rows) > max_clips:
        rows = rows[:max_clips]
    return rows


def load_cached_f0n(
    audio_path: str,
    t_mel: int,
    features_root: Path,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    f0 = load_curve(feature_path(audio_path, "f0", features_root))
    n = load_curve(feature_path(audio_path, "n", features_root))
    if f0 is None or n is None:
        return None
    if f0.numel() < t_mel or n.numel() < t_mel:
        return None
    return f0[:t_mel], n[:t_mel]


@torch.no_grad()
def cached_f0n_override(
    model: KModel,
    input_ids: torch.Tensor,
    slot: torch.Tensor,
    aln: torch.Tensor,
    f0_st2: torch.Tensor,
    n_st2: torch.Tensor,
    device: torch.device,
    calibrate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ST2 cache → Kokoro-length override curves (detached constants)."""
    f0_ref, n_ref = kokoro_f0n_for_ref(model, input_ids, slot.detach(), aln)
    if calibrate:
        f0_ov, n_ov = calibrate_st2_to_kokoro(
            f0_st2.to(device), n_st2.to(device), f0_ref, n_ref
        )
    else:
        # naive resize into Kokoro frame grid
        f0_ov = F.interpolate(
            f0_st2.view(1, 1, -1).to(device),
            size=f0_ref.numel(),
            mode="linear",
            align_corners=False,
        ).view(-1)
        n_ov = F.interpolate(
            n_st2.view(1, 1, -1).to(device),
            size=n_ref.numel(),
            mode="linear",
            align_corners=False,
        ).view(-1)
    return f0_ov.detach(), n_ov.detach()


@torch.no_grad()
def write_listen_override(
    out_dir: Path,
    model: KModel,
    baseline_pack: torch.Tensor,
    timbre: torch.Tensor,
    holdout: list[dict[str, str]],
    device: torch.device,
    max_wav_sec: float,
    features_root: Path,
    calibrate: bool,
    title: str,
) -> None:
    listen = out_dir / "listen_test"
    listen.mkdir(parents=True, exist_ok=True)
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html_lib.escape(title)}</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:1.5rem auto}"
        "audio{width:100%;margin:.2rem 0}.card{margin:1.25rem 0;padding:1rem;"
        "border:1px solid #ccc}</style></head><body>",
        f"<h1>{html_lib.escape(title)}</h1>",
        "<p><b>ref</b> · <b>opt+override</b> (trained) · <b>opt native F0N</b> · "
        "<b>baseline+override</b></p>",
        "<p>d_gt forced; F0/N from ST2 cache (calibrated) unless native</p>",
    ]
    for i, row in enumerate(holdout):
        phonemes = (row.get("phonemes_st2") or "").strip()
        d_gt = parse_d_gt(row.get("d_gt"))
        if not phonemes or not d_gt:
            continue
        wav = load_wav(row["audio_path"], max_wav_sec)
        ref_path = listen / f"{i:02d}_ref.wav"
        sf.write(str(ref_path), wav.numpy(), SAMPLE_RATE)

        input_ids, ids, dropped = kokoro_ids(model, phonemes, device)
        if dropped or len(ids) != len(phonemes):
            continue
        aln = build_alignment(d_gt, device)
        if aln.shape[1] != input_ids.shape[1]:
            continue

        t_mel = int(waveform_to_mel(wav.unsqueeze(0), device=device).shape[-1])
        cached = load_cached_f0n(row["audio_path"], t_mel, features_root)
        if cached is None:
            continue
        f0_st2, n_st2 = cached
        slot = pack_slot(baseline_pack, len(phonemes))
        base_p = slot[0, 128:]
        ref_opt = make_ref_s(timbre.detach(), base_p).to(device)
        f0_ov, n_ov = cached_f0n_override(
            model, input_ids, slot, aln, f0_st2, n_st2, device, calibrate
        )

        def synth(ref_s: torch.Tensor, use_ov: bool) -> torch.Tensor | None:
            kw = {}
            if use_ov:
                kw = {"f0_override": f0_ov, "n_override": n_ov}
            audio, _, _, _ = model.forward_with_alignment(input_ids, ref_s, aln, **kw)
            if audio.numel() < 100:
                return None
            return audio.detach().cpu()

        text = html_lib.escape((row.get("text") or "")[:160])
        block = [
            f"<div class='card'><h3>#{i} spk={row['speaker_id']} len={len(phonemes)}</h3>",
            f"<p>{text}</p>",
            f"<p>ref</p><audio controls src='{ref_path.name}'></audio>",
        ]
        for label, audio, name in (
            ("opt+F0N override", synth(ref_opt, True), f"{i:02d}_opt_override.wav"),
            ("opt native F0N", synth(ref_opt, False), f"{i:02d}_opt_native.wav"),
            ("baseline+override", synth(slot.to(device), True), f"{i:02d}_base_override.wav"),
        ):
            if audio is None:
                block.append(f"<p>{label}: SKIP</p>")
                continue
            sf.write(str(listen / name), audio.numpy(), SAMPLE_RATE)
            block.append(f"<p>{label}</p><audio controls src='{name}'></audio>")
        block.append("</div>")
        lines.extend(block)

    lines.append("</body></html>")
    (listen / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optimize timbre with d_gt + cached F0/N overrides"
    )
    ap.add_argument("--speaker", type=str, default="6209")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument("--features-root", type=str, default=str(DEFAULT_FEATURES_ROOT))
    ap.add_argument("--baseline", type=str, default="mean")
    ap.add_argument("--voice-bank", choices=("none", "us", "core"), default="us")
    ap.add_argument("--norm-to-bank", action="store_true")
    ap.add_argument("--z-clip", type=float, default=2.5)
    ap.add_argument("--bank-cache", type=str, default="")
    ap.add_argument("--timbre-ckpt", type=str, default="", help="Optional warm-start")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lambda-mel", type=float, default=1.0)
    ap.add_argument("--lambda-res", type=float, default=0.1)
    ap.add_argument("--lambda-slm", type=float, default=0.0)
    ap.add_argument("--slm-device", type=str, default="cpu")
    ap.add_argument("--slm-layers", type=str, default="4,8,12")
    ap.add_argument("--max-clips", type=int, default=64)
    ap.add_argument("--max-phonemes", type=int, default=100)
    ap.add_argument("--max-wav-sec", type=float, default=8.0)
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip ST2→Kokoro calibration (naive resize only)",
    )
    args = ap.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    out_dir = Path(args.out_dir) if args.out_dir else (
        HERE / "runs" / f"speaker_ref_s_override_{args.speaker}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    features_root = Path(args.features_root)
    calibrate = not args.no_calibrate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} speaker={args.speaker} mode=timbre+F0N_override "
        f"baseline={args.baseline} batch={args.batch_size} lr={args.lr} "
        f"mel={args.lambda_mel} res={args.lambda_res} slm={args.lambda_slm} "
        f"calibrate={calibrate} out={out_dir}",
        flush=True,
    )

    slm = None
    if args.lambda_slm > 0:
        from timbre_encoder.slm_loss import WavLMFeatureLoss

        layers = [int(x) for x in args.slm_layers.split(",") if x.strip()]
        print(f"Loading WavLM on {args.slm_device} layers={layers}...", flush=True)
        slm = WavLMFeatureLoss(layers=layers, device=args.slm_device)

    rows = load_speaker_rows_f0n(
        Path(args.manifest),
        args.speaker,
        max_phonemes=args.max_phonemes,
        max_clips=args.max_clips + args.holdout,
        seed=args.seed,
        features_root=features_root,
    )
    if len(rows) < args.holdout + 1:
        raise SystemExit(
            f"Need >= {args.holdout + 1} clips with align+f0+n for {args.speaker}, got {len(rows)}"
        )
    holdout = rows[: args.holdout]
    train_rows = rows[args.holdout :]
    if args.max_clips > 0:
        train_rows = train_rows[: args.max_clips]
    print(f"train_clips={len(train_rows)} holdout={len(holdout)}", flush=True)

    pipeline = KPipeline(lang_code="a", model=False)
    bank = None
    if args.voice_bank != "none":
        cache = Path(args.bank_cache) if args.bank_cache else (out_dir / "voice_bank.pt")
        if cache.is_file():
            print(f"Loading voice bank cache {cache}", flush=True)
            bank = load_bank(cache, device=device)
        else:
            names = US_VOICES if args.voice_bank == "us" else CORE_VOICES
            bank = build_voice_bank(pipeline, names).to(device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            save_bank(bank, cache)
            print(f"Saved voice bank -> {cache}", flush=True)

    if args.baseline == "mean":
        if bank is None:
            raise SystemExit("--baseline mean requires --voice-bank us|core")
        baseline_pack = bank.mean_pack.to(device)
        print("baseline_pack = voice-bank MEAN pack", flush=True)
    else:
        baseline_pack = pipeline.load_voice(args.baseline).to(device)
        if baseline_pack.dim() == 2:
            baseline_pack = baseline_pack.unsqueeze(1)

    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    if args.timbre_ckpt:
        from optimize_speaker_ref_s import load_timbre_ckpt

        timbre0 = load_timbre_ckpt(Path(args.timbre_ckpt), device).detach()
        print(f"init timbre from {args.timbre_ckpt}", flush=True)
    elif bank is not None:
        timbre0 = bank.timbre_mean.detach().clone()
        print("init timbre = bank.timbre_mean", flush=True)
    else:
        timbre0 = baseline_pack[0, 0, :128].detach().clone()
        print("init timbre = baseline slot0", flush=True)
    timbre = timbre0.clone().requires_grad_(True)

    opt = torch.optim.Adam([timbre], lr=args.lr)
    rng = random.Random(args.seed)
    history: list[dict[str, float]] = []
    skipped = 0
    bs = args.batch_size

    def microbatch(row: dict[str, str]) -> dict[str, torch.Tensor] | None:
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
            wave = load_wav(row["audio_path"], args.max_wav_sec).to(device)
        except Exception:
            return None
        if wave.numel() < SAMPLE_RATE // 4:
            return None

        tgt_mel = waveform_to_mel(wave.unsqueeze(0), device=device)
        cached = load_cached_f0n(row["audio_path"], int(tgt_mel.shape[-1]), features_root)
        if cached is None:
            return None
        f0_st2, n_st2 = cached

        slot = pack_slot(baseline_pack, len(phonemes))
        base_p = slot[0, 128:].detach()
        # Decoder AdaIN uses only [:128]; prosody frozen baseline
        ref_s = make_ref_s(timbre, base_p)

        with torch.no_grad():
            f0_ov, n_ov = cached_f0n_override(
                model, input_ids, slot, aln, f0_st2, n_st2, device, calibrate
            )

        audio_pred, _, _, _ = model.forward_with_alignment(
            input_ids, ref_s, aln, f0_override=f0_ov, n_override=n_ov
        )
        if audio_pred.dim() == 0 or audio_pred.numel() < 100:
            return None

        pred_mel = waveform_to_mel(
            audio_pred.unsqueeze(0) if audio_pred.dim() == 1 else audio_pred,
            device=device,
        )
        t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
        if t < 4:
            return None
        loss_mel = F.l1_loss(pred_mel[..., :t], tgt_mel[..., :t].detach())
        loss_res = (timbre - timbre0).pow(2).mean()

        loss_slm = ref_s.new_zeros(())
        if slm is not None and args.lambda_slm > 0:
            pred_w = audio_pred.view(1, -1) if audio_pred.dim() == 1 else audio_pred
            real_w = wave.view(1, -1)
            tw = min(pred_w.shape[-1], real_w.shape[-1])
            loss_slm = slm(pred_w[..., :tw], real_w[..., :tw].detach())

        loss = (
            args.lambda_mel * loss_mel
            + args.lambda_res * loss_res
            + args.lambda_slm * loss_slm
        )
        return {"loss": loss, "mel": loss_mel, "res": loss_res, "slm": loss_slm}

    for step in range(1, args.steps + 1):
        opt.zero_grad()
        acc = {"loss": 0.0, "mel": 0.0, "res": 0.0, "slm": 0.0}
        n_ok = 0
        attempts = 0
        while n_ok < bs and attempts < bs * 20:
            attempts += 1
            try:
                parts = microbatch(train_rows[rng.randrange(len(train_rows))])
            except RuntimeError as e:
                skipped += 1
                print(f"step={step} microbatch RuntimeError: {e}", flush=True)
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
            if parts is None:
                skipped += 1
                continue
            (parts["loss"] / bs).backward()
            for k in acc:
                acc[k] += float(parts[k].detach().item())
            n_ok += 1
            # free graph refs early
            del parts

        if n_ok == 0:
            print(f"step={step} WARN: empty batch", flush=True)
            continue
        if n_ok < bs:
            scale = bs / n_ok
            if timbre.grad is not None:
                timbre.grad.mul_(scale)

        opt.step()
        if args.norm_to_bank and bank is not None:
            with torch.no_grad():
                timbre.copy_(
                    normalize_to_bank(timbre, bank.timbre_mean, bank.timbre_std, args.z_clip)
                )

        hist = {
            "step": float(step),
            "loss": acc["loss"] / n_ok,
            "mel": acc["mel"] / n_ok,
            "res": acc["res"] / n_ok,
            "slm": acc["slm"] / n_ok,
        }
        history.append(hist)
        log_every = 1 if args.lambda_slm > 0 else 10
        if step == 1 or step % log_every == 0 or step == args.steps:
            print(
                f"step={step}/{args.steps} batch={n_ok}/{bs} "
                f"loss={hist['loss']:.4f} mel={hist['mel']:.4f} "
                f"slm={hist['slm']:.4f} res={hist['res']:.4f} skipped={skipped}",
                flush=True,
            )

    timbre_cpu = timbre.detach().cpu()
    torch.save(
        {
            "timbre": timbre_cpu,
            "speaker": args.speaker,
            "baseline": args.baseline,
            "train_target": "timbre",
            "f0n_mode": "override_cache",
            "calibrate": calibrate,
        },
        out_dir / "style.pt",
    )
    torch.save(
        {"timbre": timbre_cpu, "speaker": args.speaker, "baseline": args.baseline},
        out_dir / "timbre.pt",
    )
    pack = assemble_voice_pack(timbre_cpu, baseline_pack.cpu())
    torch.save(pack, out_dir / "voice_pack.pt")

    first_n = history[:50]
    last_n = history[-50:] if len(history) >= 50 else history

    def avg(key: str, xs: list[dict[str, float]]) -> float:
        if not xs:
            return float("nan")
        return sum(h[key] for h in xs) / len(xs)

    summary = {
        "speaker": args.speaker,
        "train_target": "timbre",
        "f0n_mode": "override_cache",
        "calibrate": calibrate,
        "baseline": args.baseline,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_mel": args.lambda_mel,
        "lambda_res": args.lambda_res,
        "lambda_slm": args.lambda_slm,
        "train_clips": len(train_rows),
        "skipped": skipped,
        "mel_first50": avg("mel", first_n),
        "mel_last50": avg("mel", last_n),
        "slm_last50": avg("slm", last_n),
        "res_last50": avg("res", last_n),
        "vec_l2_from_start": float((timbre_cpu - timbre0.cpu()).pow(2).sum().sqrt()),
    }
    print("summary", summary, flush=True)
    (out_dir / "summary.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in summary.items()) + "\n",
        encoding="utf-8",
    )

    model.eval()
    write_listen_override(
        out_dir,
        model,
        baseline_pack,
        timbre,
        holdout,
        device,
        args.max_wav_sec,
        features_root,
        calibrate,
        title=f"Speaker {args.speaker} timbre+F0N override base={args.baseline} lr={args.lr}",
    )
    print(f"saved {out_dir} listen={out_dir / 'listen_test' / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
