"""
Train TimbreEncoder (mel → [:128]) with frozen Kokoro + WavLM.

Hard device split for 4GB GPUs (GTX 1650 Ti):

  GPU  — TimbreEncoder (+ optional Kokoro if --kokoro-device cuda)
  CPU  — WavLM always; MPD/MSD GAN disc (default); Kokoro if --kokoro-device cpu

Forced d_gt alignment; baseline prosody [128:].
F0/N: --f0n-source cache (JDC), rmvpe, jdc_ft, or baseline (F0Ntrain).

Optional StyleTTS2 waveform GAN (--lambda-gen): MPD+MSD d_loss / loss_gen.

  .venv\\Scripts\\python.exe speaker_ref_opt/train_encoder.py \\
      --speaker 6209 --steps 200 --batch-size 4 --lambda-teacher 0 \\
      --f0n-source jdc_ft --d-gt-source tma_ft --lambda-gen 1.0 \\
      --enc-device cuda --kokoro-device cuda --slm-device cpu --disc-device cpu
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

from timbre_encoder.dual_encoder import MIN_STYLE_FRAMES, crop_mel_for_style  # noqa: E402
from timbre_encoder.encoder import TimbreEncoder  # noqa: E402
from gt_cache import DEFAULT_FEATURES_ROOT, feature_path, load_curve  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import (  # noqa: E402
    calibrate_st2_to_kokoro,
    freeze_module,
    kokoro_f0n_for_ref,
    match_mean_std,
    resize_1d,
)
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.slm_loss import WavLMFeatureLoss  # noqa: E402
from timbre_encoder.voice_bank import load_bank  # noqa: E402

from optimize_speaker_ref_s import (  # noqa: E402
    build_alignment,
    kokoro_ids,
    load_wav,
    make_ref_s,
    pack_slot,
)
from st2_gan import StyleTTS2GAN  # noqa: E402


def f0_kind_for_source(f0n_source: str) -> str:
    if f0n_source == "rmvpe":
        return "f0_rmvpe"
    if f0n_source == "jdc_ft":
        return "f0_jdc_ft"
    return "f0"


def f0_status_col(f0n_source: str) -> str:
    if f0n_source == "rmvpe":
        return "f0_rmvpe_status"
    if f0n_source == "jdc_ft":
        return "f0_jdc_ft_status"
    return "f0_status"


def d_gt_fields(d_gt_source: str) -> tuple[str, str]:
    if d_gt_source == "tma_ft":
        return "d_gt_tma_ft", "align_status_tma_ft"
    return "d_gt", "align_status"


def load_rows(
    manifest: Path,
    speaker: str | None,
    max_clips: int,
    max_phonemes: int,
    require_f0n: bool = False,
    features_root: Path | None = None,
    f0n_source: str = "cache",
    d_gt_source: str = "baseline",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    f0_kind = f0_kind_for_source(f0n_source)
    f0_col = f0_status_col(f0n_source)
    d_col, align_col = d_gt_fields(d_gt_source)
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if speaker and row.get("speaker_id") != speaker:
                continue
            if row.get(align_col) != "ok":
                continue
            if require_f0n:
                if row.get(f0_col) != "ok" or row.get("n_status") != "ok":
                    continue
                if features_root is not None:
                    if not feature_path(row["audio_path"], f0_kind, features_root).is_file():
                        continue
                    if not feature_path(row["audio_path"], "n", features_root).is_file():
                        continue
            ph = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get(d_col))
            if not ph or not d_gt or len(d_gt) != len(ph):
                continue
            if not (2 <= len(ph) <= max_phonemes):
                continue
            rows.append(row)
    rows.sort(key=lambda r: (r["speaker_id"], r.get("audio_path") or ""))
    if max_clips > 0:
        rows = rows[:max_clips]
    return rows


def load_cached_f0n(
    audio_path: str,
    t_mel: int,
    features_root: Path,
    f0_kind: str = "f0",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    f0 = load_curve(feature_path(audio_path, f0_kind, features_root))
    n = load_curve(feature_path(audio_path, "n", features_root))
    if f0 is None or n is None:
        return None
    if f0.numel() < t_mel or n.numel() < t_mel:
        return None
    return f0[:t_mel], n[:t_mel]


def resolve_f0n_override(
    kmodel: KModel,
    input_ids: torch.Tensor,
    slot: torch.Tensor,
    aln: torch.Tensor,
    row: dict[str, str],
    wav: torch.Tensor,
    features_root: Path,
    f0n_source: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Returns (f0, n) in Kokoro units for decoder override.
    baseline: F0Ntrain(slot)
    cache / rmvpe: calibrate → Kokoro via baseline F0Ntrain
    jdc_ft: already Kokoro units — resize F0; N mean/std-matched
    """
    f0_ref, n_ref = kokoro_f0n_for_ref(kmodel, input_ids, slot, aln)
    if f0n_source == "baseline":
        return f0_ref, n_ref
    mel = waveform_to_mel(wav.to(device) if wav.device != device else wav, device=device)
    t_mel = int(mel.squeeze(0).shape[-1]) if mel.dim() == 3 else int(mel.shape[-1])
    cached = load_cached_f0n(
        row["audio_path"],
        t_mel,
        features_root,
        f0_kind=f0_kind_for_source(f0n_source),
    )
    if cached is None:
        return None
    f0_src, n_src = cached
    if f0n_source == "jdc_ft":
        f0 = resize_1d(f0_src.to(device), f0_ref.numel())
        n = match_mean_std(n_src.to(device), n_ref)
        return f0, n
    return calibrate_st2_to_kokoro(
        f0_src.to(device), n_src.to(device), f0_ref, n_ref
    )


def encode_multicrop(
    enc: TimbreEncoder,
    mel: torch.Tensor,
    n_crops: int,
    style_frames: int,
    min_frames: int,
) -> torch.Tensor:
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    vecs = [
        enc.encode_mel(
            crop_mel_for_style(mel, max_frames=style_frames, min_frames=min_frames)
        )
        for _ in range(max(1, n_crops))
    ]
    return torch.stack(vecs, dim=0).mean(dim=0)


def utt_loss(
    enc: TimbreEncoder,
    kmodel: KModel,
    baseline_pack: torch.Tensor,
    timbre0: torch.Tensor,
    teacher: torch.Tensor | None,
    slm: WavLMFeatureLoss | None,
    row: dict[str, str],
    enc_device: torch.device,
    kokoro_device: torch.device,
    features_root: Path,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor] | None:
    ph = (row.get("phonemes_st2") or "").strip()
    d_col, _ = d_gt_fields(args.d_gt_source)
    d_gt = parse_d_gt(row.get(d_col))
    if not ph or not d_gt:
        return None
    input_ids, ids, dropped = kokoro_ids(kmodel, ph, kokoro_device)
    if dropped or len(ids) != len(ph):
        return None
    aln = build_alignment(d_gt, kokoro_device)
    if aln.shape[1] != input_ids.shape[1]:
        return None
    try:
        wav = load_wav(row["audio_path"], args.max_wav_sec)
    except Exception:
        return None
    if wav.numel() < SAMPLE_RATE // 4:
        return None

    mel = waveform_to_mel(wav.to(enc_device)).squeeze(0)
    if mel.shape[-1] < 16:
        return None

    timbre = encode_multicrop(
        enc, mel, args.n_crops, args.style_frames, args.min_style_frames
    ).squeeze(0)

    slot = pack_slot(baseline_pack, len(ph))
    base_p = slot[0, 128:].detach()
    ref_s = make_ref_s(timbre.to(kokoro_device), base_p)

    zero = timbre.new_zeros(())
    loss_mel = zero
    loss_slm = zero
    loss_tea = zero
    loss_res = zero
    pred_wav: torch.Tensor | None = None
    real_wav: torch.Tensor | None = None

    need_audio = (
        args.lambda_mel > 0
        or (slm is not None and args.lambda_slm > 0)
        or args.lambda_gen > 0
    )
    if need_audio:
        with torch.no_grad():
            ov = resolve_f0n_override(
                kmodel,
                input_ids,
                slot,
                aln,
                row,
                wav,
                features_root,
                args.f0n_source,
                kokoro_device,
            )
        if ov is None:
            return None
        f0_ov, n_ov = ov
        audio, _, _, _ = kmodel.forward_with_alignment(
            input_ids, ref_s, aln, f0_override=f0_ov, n_override=n_ov
        )
        if audio.numel() < 100:
            return None
        pred_wav = audio.view(1, -1) if audio.dim() == 1 else audio
        real_wav = wav.view(1, -1).to(pred_wav.device)
        if args.lambda_mel > 0:
            pred_mel = waveform_to_mel(pred_wav, device=kokoro_device)
            tgt_mel = waveform_to_mel(real_wav, device=kokoro_device)
            t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
            if t < 4:
                return None
            loss_mel = F.l1_loss(pred_mel[..., :t], tgt_mel[..., :t].detach()).to(enc_device)
        if slm is not None and args.lambda_slm > 0:
            tw = min(pred_wav.shape[-1], real_wav.shape[-1])
            loss_slm = slm(pred_wav[..., :tw], real_wav[..., :tw].detach()).to(enc_device)

    if teacher is not None and args.lambda_teacher > 0:
        loss_tea = F.l1_loss(timbre, teacher) + (
            1.0 - F.cosine_similarity(timbre.unsqueeze(0), teacher.unsqueeze(0)).mean()
        )

    if args.lambda_res > 0:
        loss_res = (timbre - timbre0).pow(2).mean()

    loss = (
        args.lambda_mel * loss_mel
        + args.lambda_slm * loss_slm
        + args.lambda_teacher * loss_tea
        + args.lambda_res * loss_res
    )
    return {
        "loss": loss,
        "mel": loss_mel.detach(),
        "slm": loss_slm.detach() if torch.is_tensor(loss_slm) else zero,
        "tea": loss_tea.detach() if torch.is_tensor(loss_tea) else zero,
        "res": loss_res.detach() if torch.is_tensor(loss_res) else zero,
        "pred_wav": pred_wav,
        "real_wav": real_wav,
    }


@torch.no_grad()
def write_listen(
    out_dir: Path,
    enc: TimbreEncoder,
    kmodel: KModel,
    baseline_pack: torch.Tensor,
    teacher: torch.Tensor | None,
    holdout: list[dict[str, str]],
    enc_device: torch.device,
    kokoro_device: torch.device,
    features_root: Path,
    args: argparse.Namespace,
) -> None:
    listen = out_dir / "listen_test"
    listen.mkdir(parents=True, exist_ok=True)
    enc.eval()
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Encoder train</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:1.5rem auto}"
        "audio{width:100%}.card{margin:1rem 0;padding:1rem;border:1px solid #ccc}</style>"
        "</head><body>",
        f"<h1>{html_lib.escape(out_dir.name)}</h1>",
        f"<p>forced <b>d_gt={html_lib.escape(args.d_gt_source)}</b> · baseline prosody [128:] · "
        f"F0/N: <b>base_f0n</b>=baseline F0Ntrain · "
        f"<b>ov_f0n</b>={html_lib.escape(args.f0n_source)}</p>",
        "<p>ref | enc+base_f0n | enc+ov_f0n | baseline+base_f0n | baseline+ov_f0n</p>",
    ]
    d_col, _ = d_gt_fields(args.d_gt_source)
    for i, row in enumerate(holdout[: args.listen_n]):
        ph = (row.get("phonemes_st2") or "").strip()
        d_gt = parse_d_gt(row.get(d_col))
        if not ph or not d_gt:
            continue
        input_ids, ids, dropped = kokoro_ids(kmodel, ph, kokoro_device)
        if dropped or len(ids) != len(ph):
            continue
        aln = build_alignment(d_gt, kokoro_device)
        if aln.shape[1] != input_ids.shape[1]:
            continue
        wav = load_wav(row["audio_path"], args.max_wav_sec)
        mel = waveform_to_mel(wav.to(enc_device)).squeeze(0)
        timbre = encode_multicrop(
            enc, mel, args.n_crops, args.style_frames, args.min_style_frames
        ).squeeze(0)
        slot = pack_slot(baseline_pack, len(ph))
        base_p = slot[0, 128:]
        base_a = slot[0, :128]

        f0_base, n_base = kokoro_f0n_for_ref(kmodel, input_ids, slot, aln)
        ov_src = args.f0n_source if args.f0n_source != "baseline" else "jdc_ft"
        ov = resolve_f0n_override(
            kmodel,
            input_ids,
            slot,
            aln,
            row,
            wav,
            features_root,
            ov_src,
            kokoro_device,
        )
        if ov is None:
            ov = resolve_f0n_override(
                kmodel,
                input_ids,
                slot,
                aln,
                row,
                wav,
                features_root,
                "rmvpe",
                kokoro_device,
            )

        def synth(
            timbre_vec: torch.Tensor,
            f0: torch.Tensor | None,
            n: torch.Tensor | None,
        ):
            ref_s = make_ref_s(timbre_vec.to(kokoro_device), base_p)
            audio, _, _, _ = kmodel.forward_with_alignment(
                input_ids, ref_s, aln, f0_override=f0, n_override=n
            )
            if audio.numel() < 100:
                return None
            return audio.detach().cpu().numpy().reshape(-1)

        stem = f"{i:02d}_{row['speaker_id']}"
        sf.write(str(listen / f"{stem}_0_ref.wav"), wav.numpy(), SAMPLE_RATE)
        block = [
            f"<div class='card'><h3>{html_lib.escape(stem)}</h3>",
            f"<p>{html_lib.escape((row.get('text') or '')[:160])}</p>",
            f"<p>ref</p><audio controls src='{stem}_0_ref.wav'></audio>",
        ]

        def add(label: str, audio, name: str) -> None:
            if audio is None:
                block.append(f"<p>{label}: SKIP</p>")
                return
            sf.write(str(listen / name), audio, SAMPLE_RATE)
            block.append(f"<p>{label}</p><audio controls src='{name}'></audio>")

        add("enc + base_f0n", synth(timbre, f0_base, n_base), f"{stem}_1_enc_base_f0n.wav")
        if ov is not None:
            add("enc + ov_f0n", synth(timbre, ov[0], ov[1]), f"{stem}_2_enc_ov_f0n.wav")
        else:
            block.append("<p>enc + ov_f0n: SKIP (no cache)</p>")
        add(
            "baseline + base_f0n",
            synth(base_a, f0_base, n_base),
            f"{stem}_3_base_base_f0n.wav",
        )
        if ov is not None:
            add(
                "baseline + ov_f0n",
                synth(base_a, ov[0], ov[1]),
                f"{stem}_4_base_ov_f0n.wav",
            )
        if teacher is not None:
            add(
                "teacher + ov_f0n" if ov is not None else "teacher + base_f0n",
                synth(teacher, *(ov if ov is not None else (f0_base, n_base))),
                f"{stem}_5_teacher.wav",
            )
        block.append("</div>")
        lines.extend(block)
    lines.append("</body></html>")
    (listen / "index.html").write_text("\n".join(lines), encoding="utf-8")
    print(f"listen -> {listen / 'index.html'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TimbreEncoder with device split")
    ap.add_argument("--speaker", type=str, default="6209")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument("--baseline", type=str, default="mean")
    ap.add_argument(
        "--bank-cache",
        type=str,
        default=str(HERE / "runs/speaker_ref_s_6209_bankmean_lr1e5/voice_bank.pt"),
    )
    ap.add_argument("--enc-device", type=str, default="cuda")
    ap.add_argument(
        "--kokoro-device",
        type=str,
        default="cuda",
        help="Put frozen Kokoro on cuda|cpu (cpu saves VRAM, much slower)",
    )
    ap.add_argument("--slm-device", type=str, default="cpu")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-mel", type=float, default=0.25)
    ap.add_argument("--lambda-slm", type=float, default=1.0)
    ap.add_argument("--slm-cos", action="store_true")
    ap.add_argument("--lambda-teacher", type=float, default=1.0)
    ap.add_argument("--lambda-res", type=float, default=0.0)
    ap.add_argument(
        "--lambda-gen",
        type=float,
        default=0.0,
        help="StyleTTS2 MPD+MSD generator loss weight (also enables d_loss updates)",
    )
    ap.add_argument(
        "--disc-device",
        type=str,
        default="cpu",
        help="MPD/MSD device (cpu recommended on 4GB)",
    )
    ap.add_argument("--disc-lr", type=float, default=1e-4)
    ap.add_argument(
        "--slim-disc",
        action="store_true",
        help="Smaller MPD channels/periods (safer if disc on cuda)",
    )
    ap.add_argument("--slm-layers", type=str, default="4,8,12")
    ap.add_argument(
        "--f0n-source",
        choices=("cache", "rmvpe", "jdc_ft", "baseline"),
        default="jdc_ft",
        help="F0: cache=JDC, rmvpe, jdc_ft=fine-tuned Kokoro units, baseline=F0Ntrain; N=log_norm",
    )
    ap.add_argument(
        "--d-gt-source",
        choices=("baseline", "tma_ft"),
        default="tma_ft",
        help="Alignment: baseline d_gt or fine-tuned TMA d_gt_tma_ft",
    )
    ap.add_argument(
        "--features-root",
        type=str,
        default=str(DEFAULT_FEATURES_ROOT),
    )
    ap.add_argument(
        "--teacher-ckpt",
        type=str,
        default=str(
            HERE
            / "runs/speaker_ref_s_6209_f0n_accum_wavlm_l1cos_mel025_bs8_200/timbre.pt"
        ),
    )
    ap.add_argument("--max-clips", type=int, default=64)
    ap.add_argument("--max-phonemes", type=int, default=50)
    ap.add_argument("--max-wav-sec", type=float, default=8.0)
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--listen-n", type=int, default=3)
    ap.add_argument("--n-crops", type=int, default=2)
    ap.add_argument("--style-frames", type=int, default=192)
    ap.add_argument("--min-style-frames", type=int, default=MIN_STYLE_FRAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if not args.slm_cos and args.lambda_slm > 0:
        args.slm_cos = True

    enc_device = torch.device(
        args.enc_device if args.enc_device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    kokoro_device = torch.device(
        args.kokoro_device
        if args.kokoro_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    slm_device = torch.device(args.slm_device)
    disc_device = torch.device(
        args.disc_device
        if args.disc_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    features_root = Path(args.features_root)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        name = (
            f"encoder_{args.speaker}_dgt{args.d_gt_source}_f0{args.f0n_source}"
            f"_mel{args.lambda_mel}"
        )
        if args.lambda_gen > 0:
            name += f"_gen{args.lambda_gen}"
        out_dir = HERE / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("DEVICE MAP (hard split)", flush=True)
    print(f"  TimbreEncoder  -> {enc_device}", flush=True)
    print(f"  Kokoro (frozen)-> {kokoro_device}", flush=True)
    print(f"  WavLM (frozen) -> {slm_device}", flush=True)
    print(f"  GAN disc MPD/MSD -> {disc_device} (lambda_gen={args.lambda_gen})", flush=True)
    print(f"  d_gt source    -> {args.d_gt_source}", flush=True)
    print(f"  F0/N override  -> {args.f0n_source}", flush=True)
    print("=" * 60, flush=True)
    print(
        f"speaker={args.speaker} steps={args.steps} batch={args.batch_size} lr={args.lr} "
        f"mel={args.lambda_mel} slm={args.lambda_slm} cos={args.slm_cos} "
        f"teacher={args.lambda_teacher} res={args.lambda_res} gen={args.lambda_gen} "
        f"d_gt={args.d_gt_source} f0n={args.f0n_source} "
        f"max_ph={args.max_phonemes} out={out_dir}",
        flush=True,
    )

    need_cache = args.f0n_source in ("cache", "rmvpe", "jdc_ft")
    rows = load_rows(
        Path(args.manifest),
        args.speaker or None,
        max_clips=args.max_clips + args.holdout,
        max_phonemes=args.max_phonemes,
        require_f0n=need_cache,
        features_root=features_root if need_cache else None,
        f0n_source=args.f0n_source,
        d_gt_source=args.d_gt_source,
    )
    if len(rows) < args.holdout + args.batch_size:
        raise SystemExit(f"not enough clips: {len(rows)}")
    # shuffle then holdout first (same as opt scripts)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    holdout = rows[: args.holdout]
    train_rows = rows[args.holdout :]
    if args.max_clips > 0:
        train_rows = train_rows[: args.max_clips]
    print(f"train={len(train_rows)} holdout={len(holdout)}", flush=True)

    pipeline = KPipeline(lang_code="a", model=False)
    bank = None
    if args.baseline == "mean":
        cache = Path(args.bank_cache)
        if not cache.is_file():
            raise SystemExit(f"bank cache missing: {cache}")
        bank = load_bank(cache, device="cpu")
        baseline_pack = bank.mean_pack.to(kokoro_device)
        timbre0 = bank.timbre_mean.detach().to(enc_device)
        print("baseline=mean bank", flush=True)
    else:
        baseline_pack = pipeline.load_voice(args.baseline).to(kokoro_device)
        if baseline_pack.dim() == 2:
            baseline_pack = baseline_pack.unsqueeze(1)
        timbre0 = baseline_pack[0, 0, :128].detach().to(enc_device)

    print(f"Loading Kokoro on {kokoro_device}...", flush=True)
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M").to(kokoro_device)
    freeze_module(kmodel)

    enc = TimbreEncoder().to(enc_device)
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)

    teacher = None
    if args.teacher_ckpt and Path(args.teacher_ckpt).is_file() and args.lambda_teacher > 0:
        ck = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        t = ck["timbre"] if isinstance(ck, dict) and "timbre" in ck else ck
        teacher = t.detach().float().view(-1).to(enc_device)
        if teacher.numel() != 128:
            raise SystemExit(f"teacher timbre must be [128], got {tuple(teacher.shape)}")
        print(f"teacher <- {args.teacher_ckpt}", flush=True)
    else:
        args.lambda_teacher = 0.0
        print("no teacher (lambda_teacher=0)", flush=True)

    slm = None
    if args.lambda_slm > 0:
        layers = [int(x) for x in args.slm_layers.split(",") if x.strip()]
        print(
            f"Loading WavLM on {slm_device} layers={layers} include_cos={args.slm_cos}...",
            flush=True,
        )
        slm = WavLMFeatureLoss(
            layers=layers, device=slm_device, include_cos=args.slm_cos
        )

    gan: StyleTTS2GAN | None = None
    opt_d = None
    if args.lambda_gen > 0:
        print(
            f"Loading StyleTTS2 MPD+MSD on {disc_device} slim={args.slim_disc}...",
            flush=True,
        )
        gan = StyleTTS2GAN(slim=args.slim_disc).to(disc_device)
        gan.train()
        opt_d = torch.optim.Adam(gan.parameters(), lr=args.disc_lr)

    if enc_device.type == "cuda":
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1024**2
        print(f"cuda allocated after load: {alloc:.0f} MiB", flush=True)

    history: list[dict[str, float]] = []
    skipped = 0
    bs = max(1, args.batch_size)

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        acc = {
            "loss": 0.0,
            "mel": 0.0,
            "slm": 0.0,
            "tea": 0.0,
            "res": 0.0,
            "gen": 0.0,
            "d": 0.0,
        }
        n_ok = 0
        attempts = 0
        while n_ok < bs and attempts < bs * 20:
            attempts += 1
            row = train_rows[rng.randrange(len(train_rows))]
            try:
                parts = utt_loss(
                    enc,
                    kmodel,
                    baseline_pack,
                    timbre0,
                    teacher,
                    slm,
                    row,
                    enc_device,
                    kokoro_device,
                    features_root,
                    args,
                )
            except torch.cuda.OutOfMemoryError:
                if enc_device.type == "cuda":
                    torch.cuda.empty_cache()
                skipped += 1
                print(f"step={step} OOM microbatch, skip", flush=True)
                continue
            if parts is None:
                skipped += 1
                continue

            loss_g = parts["loss"]
            loss_gen_v = 0.0
            loss_d_v = 0.0
            if (
                gan is not None
                and opt_d is not None
                and parts.get("pred_wav") is not None
                and parts.get("real_wav") is not None
            ):
                pred_w = parts["pred_wav"]
                real_w = parts["real_wav"]
                try:
                    # StyleTTS2 order: update D on detach, then gen through D
                    opt_d.zero_grad(set_to_none=True)
                    d_loss = gan.d_loss(
                        real_w.detach().to(disc_device),
                        pred_w.detach().to(disc_device),
                    )
                    d_loss.backward()
                    opt_d.step()
                    loss_d_v = float(d_loss.detach().item())
                    opt_d.zero_grad(set_to_none=True)

                    g_loss = gan.g_loss(
                        real_w.detach().to(disc_device),
                        pred_w.to(disc_device),
                    )
                    loss_g = loss_g + args.lambda_gen * g_loss.to(enc_device)
                    loss_gen_v = float(g_loss.detach().item())
                except Exception as e:
                    skipped += 1
                    print(f"step={step} GAN skip: {e}", flush=True)
                    del parts
                    continue

            (loss_g / bs).backward()
            for k in ("mel", "slm", "tea", "res"):
                acc[k] += float(parts[k].detach().item())
            acc["gen"] += loss_gen_v
            acc["d"] += loss_d_v
            acc["loss"] += float(loss_g.detach().item())
            n_ok += 1
            del parts

        if n_ok == 0:
            print(f"step={step} empty batch", flush=True)
            continue
        if n_ok < bs:
            scale = bs / n_ok
            for p in enc.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(enc.parameters(), args.grad_clip)
        opt.step()

        hist = {k: acc[k] / n_ok for k in acc}
        hist["step"] = float(step)
        hist["batch_ok"] = float(n_ok)
        history.append(hist)
        if step == 1 or step % 10 == 0 or step == args.steps:
            mem = ""
            if enc_device.type == "cuda":
                mem = f" alloc={torch.cuda.memory_allocated()/1024**2:.0f}MiB"
            print(
                f"step={step}/{args.steps} batch={n_ok}/{bs} "
                f"loss={hist['loss']:.4f} mel={hist['mel']:.4f} "
                f"slm={hist['slm']:.4f} gen={hist['gen']:.4f} d={hist['d']:.4f} "
                f"tea={hist['tea']:.4f} skipped={skipped}{mem}",
                flush=True,
            )
        if enc_device.type == "cuda" and step % 20 == 0:
            torch.cuda.empty_cache()

    ckpt = {
        "encoder": enc.state_dict(),
        "speaker": args.speaker,
        "baseline": args.baseline,
        "steps": args.steps,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device_map": {
            "encoder": str(enc_device),
            "kokoro": str(kokoro_device),
            "wavlm": str(slm_device),
            "disc": str(disc_device),
        },
    }
    if gan is not None:
        ckpt["gan"] = gan.state_dict()
    torch.save(ckpt, out_dir / "encoder.pt")

    def avg(key: str, xs: list[dict[str, float]]) -> float:
        if not xs:
            return float("nan")
        return sum(h[key] for h in xs) / len(xs)

    first = history[:50]
    last = history[-50:] if len(history) >= 50 else history
    summary = {
        "speaker": args.speaker,
        "steps": args.steps,
        "batch_size": bs,
        "lr": args.lr,
        "lambda_mel": args.lambda_mel,
        "lambda_slm": args.lambda_slm,
        "slm_cos": args.slm_cos,
        "lambda_teacher": args.lambda_teacher,
        "lambda_gen": args.lambda_gen,
        "f0n_source": args.f0n_source,
        "d_gt_source": args.d_gt_source,
        "enc_device": str(enc_device),
        "kokoro_device": str(kokoro_device),
        "slm_device": str(slm_device),
        "disc_device": str(disc_device),
        "train_clips": len(train_rows),
        "skipped": skipped,
        "loss_first50": avg("loss", first),
        "loss_last50": avg("loss", last),
        "mel_last50": avg("mel", last),
        "slm_last50": avg("slm", last),
        "gen_last50": avg("gen", last),
        "d_last50": avg("d", last),
        "tea_last50": avg("tea", last),
    }
    print("summary", summary, flush=True)
    (out_dir / "summary.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in summary.items()) + "\n", encoding="utf-8"
    )

    write_listen(
        out_dir,
        enc,
        kmodel,
        baseline_pack,
        teacher,
        holdout,
        enc_device,
        kokoro_device,
        features_root,
        args,
    )
    print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
