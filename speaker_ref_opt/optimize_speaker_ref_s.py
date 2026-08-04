"""
Optimize a single Kokoro style vector for one LibriTTS speaker
using cached GT alignment (forced d_gt), no neural encoder.

Copy of timbre_encoder/optimize_speaker_ref_s.py — runs live under speaker_ref_opt/.

Modes:
  timbre  — train [:128], freeze baseline prosody (default)
  prosody — freeze timbre (from ckpt), lightly train [128:] (drives F0/N)

  .venv\\Scripts\\python.exe speaker_ref_opt/optimize_speaker_ref_s.py --speaker 4195 --steps 300
  .venv\\Scripts\\python.exe speaker_ref_opt/optimize_speaker_ref_s.py --train-target prosody \\
      --timbre-ckpt speaker_ref_opt/runs/speaker_ref_s_4195/timbre.pt --steps 150 --lr 3e-4
"""
from __future__ import annotations

import argparse
import csv
import html as html_lib
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.aligner import FrozenTextAligner  # noqa: E402
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

HOP = 300  # StyleTTS2 / Kokoro-ish frame hop for F0/N targets


def load_speaker_rows(
    csv_path: Path,
    speaker: str,
    max_phonemes: int,
    max_clips: int,
    seed: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("speaker_id") != speaker:
                continue
            if row.get("align_status") != "ok":
                continue
            ph_st2 = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            if not ph_st2 or not d_gt:
                continue
            if len(d_gt) != len(ph_st2):
                continue
            if not (2 <= len(ph_st2) <= max_phonemes):
                continue
            rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_clips > 0 and len(rows) > max_clips:
        rows = rows[:max_clips]
    return rows


def kokoro_ids(model: KModel, phonemes: str, device: torch.device):
    dropped: list[str] = []
    ids: list[int] = []
    for p in phonemes:
        i = model.vocab.get(p)
        if i is None:
            dropped.append(p)
        else:
            ids.append(i)
    input_ids = torch.LongTensor([[0, *ids, 0]]).to(device)
    return input_ids, ids, dropped


def build_alignment(d_gt: list[int], device: torch.device) -> torch.Tensor:
    d_list = [1] + [max(1, int(x)) for x in d_gt] + [1]
    return FrozenTextAligner.d_gt_to_a_gt(d_list, device=device)


def load_wav(path: str, max_sec: float) -> torch.Tensor:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    max_samples = int(max_sec * SAMPLE_RATE)
    if wav.shape[0] > max_samples:
        wav = wav[:max_samples]
    return torch.from_numpy(wav.astype("float32"))


def extract_f0_n(wave: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[torch.Tensor, torch.Tensor]:
    f0 = librosa.yin(wave, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=HOP)
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    S = np.abs(librosa.stft(wave, n_fft=2048, hop_length=HOP, win_length=1200))
    energy = np.log(np.clip(S.mean(axis=0), 1e-5, None)).astype(np.float32)
    n = min(len(f0), len(energy))
    return torch.from_numpy(f0[:n]), torch.from_numpy(energy[:n])


def pack_slot(baseline_pack: torch.Tensor, phoneme_len: int) -> torch.Tensor:
    idx = max(0, min(phoneme_len - 1, baseline_pack.shape[0] - 1))
    slot = baseline_pack[idx]
    if slot.dim() == 1:
        slot = slot.unsqueeze(0)
    return slot


def make_ref_s(
    timbre: torch.Tensor,
    prosody: torch.Tensor,
) -> torch.Tensor:
    """Assemble [1, 256] from halves (either may be trainable)."""
    return torch.cat([timbre.view(-1), prosody.view(-1)], dim=0).unsqueeze(0)


def assemble_pack_both(
    timbre: torch.Tensor,
    prosody_or_delta: torch.Tensor,
    baseline_pack: torch.Tensor,
    mode: str = "delta",
) -> torch.Tensor:
    """
    Voice pack [510,1,256]: always replace timbre.
    mode=replace: same shared prosody on every slot.
    mode=delta: baseline + shared delta [128].
    mode=slot_delta: baseline + per-slot delta [510,128].
    """
    pack = assemble_voice_pack(timbre, baseline_pack)
    pack = pack.clone()
    if mode == "replace":
        p = prosody_or_delta.view(1, 1, 128)
        pack[:, :, 128:] = p
    elif mode == "slot_delta":
        d = prosody_or_delta
        if d.dim() == 2:
            d = d.unsqueeze(1)  # [510,1,128]
        pack[:, :, 128:] = pack[:, :, 128:] + d.to(pack.device)
    else:
        p = prosody_or_delta.view(1, 1, 128)
        pack[:, :, 128:] = pack[:, :, 128:] + p
    return pack


@torch.no_grad()
def synth_forced(
    model: KModel,
    phonemes: str,
    d_gt: list[int],
    ref_s: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    input_ids, ids, dropped = kokoro_ids(model, phonemes, device)
    if dropped or len(ids) != len(phonemes):
        return None
    aln = build_alignment(d_gt, device)
    if aln.shape[1] != input_ids.shape[1]:
        return None
    audio, _, _, _ = model.forward_with_alignment(input_ids, ref_s, aln)
    return audio.detach().cpu()


def write_listen(
    out_dir: Path,
    model: KModel,
    baseline_pack: torch.Tensor,
    timbre: torch.Tensor,
    prosody_delta: torch.Tensor | None,
    holdout: list[dict[str, str]],
    device: torch.device,
    max_wav_sec: float,
    title: str,
    prosody_mode: str = "shared",
    slm_scorer=None,
) -> None:
    listen = out_dir / "listen_test"
    listen.mkdir(parents=True, exist_ok=True)
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html_lib.escape(title)}</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:1.5rem auto}"
        "audio{width:100%;margin:.2rem 0}.card{margin:1.25rem 0;padding:1rem;"
        "border:1px solid #ccc}.score{color:#333;font-size:.9rem}"
        "table{border-collapse:collapse;margin:.5rem 0}td,th{padding:.2rem .6rem;"
        "border:1px solid #ddd;text-align:left}</style></head><body>",
        f"<h1>{html_lib.escape(title)}</h1>",
        f"<p>holdout={len(holdout)} clips · prosody_mode={html_lib.escape(prosody_mode)}</p>",
        "<p><b>ref</b>=LibriTTS · "
        "<b>timbre+base_p</b>=trained[:128]+baseline[128:] · "
        "<b>base_a+prosody</b>=baseline[:128]+trained[128:] · "
        "<b>opt</b>=trained both · <b>baseline</b>=pack only</p>",
        "<p class='score'><b>WavLM cos</b> ↑ better similarity · <b>WavLM L1</b> ↓ closer features</p>",
    ]
    score_rows: list[dict] = []

    def resolve_prosody(
        base_p: torch.Tensor,
        idx: int,
    ) -> torch.Tensor | None:
        if prosody_delta is None:
            return None
        if prosody_mode == "absolute":
            return prosody_delta.view(-1)[:128].detach()
        if prosody_mode == "slot" and prosody_delta.dim() == 2:
            return (base_p + prosody_delta[idx].detach()).view(-1)[:128]
        return (base_p + prosody_delta.view(-1)[:128].detach()).view(-1)[:128]

    for i, row in enumerate(holdout):
        phonemes = (row.get("phonemes_st2") or "").strip()
        d_gt = parse_d_gt(row.get("d_gt"))
        if not phonemes or not d_gt:
            continue
        wav = load_wav(row["audio_path"], max_wav_sec)
        ref_path = listen / f"{i:02d}_ref.wav"
        sf.write(str(ref_path), wav.numpy(), SAMPLE_RATE)

        slot = pack_slot(baseline_pack, len(phonemes))
        idx = max(0, min(len(phonemes) - 1, baseline_pack.shape[0] - 1))
        base_a = slot[0, :128].detach()
        base_p = slot[0, 128:].detach()
        opt_p = resolve_prosody(base_p, idx)

        ref_timbre_base_p = make_ref_s(timbre.detach(), base_p).to(device)
        if opt_p is not None:
            ref_opt = make_ref_s(timbre.detach(), opt_p).to(device)
            ref_base_a_prosody = make_ref_s(base_a, opt_p).to(device)
        else:
            ref_opt = ref_timbre_base_p
            ref_base_a_prosody = None
        base_ref = slot.to(device)

        audio_tb = synth_forced(model, phonemes, d_gt, ref_timbre_base_p, device)
        audio_bp = (
            synth_forced(model, phonemes, d_gt, ref_base_a_prosody, device)
            if ref_base_a_prosody is not None
            else None
        )
        audio_opt = synth_forced(model, phonemes, d_gt, ref_opt, device)
        audio_base = synth_forced(model, phonemes, d_gt, base_ref, device)

        text = html_lib.escape((row.get("text") or "")[:160])
        block = [
            f"<div class='card'><h3>#{i} spk={row['speaker_id']} len={len(phonemes)} slot={idx}</h3>",
            f"<p>{text}</p>",
        ]
        block.append(f"<p>ref</p><audio controls src='{ref_path.name}'></audio>")

        def score_vs_ref(audio: torch.Tensor | None) -> dict[str, float] | None:
            if audio is None or slm_scorer is None:
                return None
            return slm_scorer.similarity(audio.view(1, -1), wav.view(1, -1))

        def add(label: str, audio: torch.Tensor | None, name: str, key: str) -> None:
            path = listen / name
            if audio is None:
                block.append(f"<p>{label}: SKIP</p>")
                return
            sf.write(str(path), audio.numpy(), SAMPLE_RATE)
            sc = score_vs_ref(audio)
            if sc:
                block.append(
                    f"<p>{label} <span class='score'>"
                    f"(WavLM cos={sc['wavlm_cos']:.4f} · L1={sc['wavlm_l1']:.4f})"
                    f"</span></p>"
                )
                score_rows.append(
                    {"clip": i, "variant": key, "cos": sc["wavlm_cos"], "l1": sc["wavlm_l1"]}
                )
            else:
                block.append(f"<p>{label}</p>")
            block.append(f"<audio controls src='{path.name}'></audio>")

        add("timbre+base_p", audio_tb, f"{i:02d}_timbre_base_p.wav", "timbre_base_p")
        add("base_a+prosody", audio_bp, f"{i:02d}_base_a_prosody.wav", "base_a_prosody")
        add("opt", audio_opt, f"{i:02d}_opt_forced.wav", "opt")
        add("baseline", audio_base, f"{i:02d}_baseline_forced.wav", "baseline")
        block.append("</div>")
        lines.extend(block)

    if score_rows:
        lines.append("<h2>WavLM summary (vs ref)</h2><table><tr><th>variant</th><th>mean cos ↑</th><th>mean L1 ↓</th></tr>")
        from collections import defaultdict

        acc: dict[str, list] = defaultdict(list)
        for r in score_rows:
            acc[r["variant"]].append(r)
        for key in ("timbre_base_p", "base_a_prosody", "opt", "baseline"):
            xs = acc.get(key) or []
            if not xs:
                continue
            mc = sum(x["cos"] for x in xs) / len(xs)
            ml = sum(x["l1"] for x in xs) / len(xs)
            lines.append(f"<tr><td>{key}</td><td>{mc:.4f}</td><td>{ml:.4f}</td></tr>")
        lines.append("</table>")
        (listen / "wavlm_scores.json").write_text(
            __import__("json").dumps(score_rows, indent=2), encoding="utf-8"
        )

    lines.append("</body></html>")
    (listen / "index.html").write_text("\n".join(lines), encoding="utf-8")


def load_timbre_ckpt(path: Path, device: torch.device) -> torch.Tensor:
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "timbre" in obj:
        t = obj["timbre"]
    else:
        t = obj
    t = t.detach().float().view(-1)
    if t.numel() != 128:
        raise SystemExit(f"Expected timbre [128], got {tuple(t.shape)}")
    return t.to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize Kokoro style half for one speaker")
    ap.add_argument("--speaker", type=str, default="4195")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument("--baseline", type=str, default="af_heart", help="Voice name or 'mean' (voice-bank mean pack)")
    ap.add_argument("--train-target", choices=("timbre", "prosody"), default="timbre")
    ap.add_argument(
        "--prosody-mode",
        choices=("shared", "slot", "absolute"),
        default="absolute",
        help="absolute=free prosody[128] (classic); shared/slot=delta on baseline (legacy)",
    )
    ap.add_argument(
        "--voice-bank",
        choices=("none", "us", "core"),
        default="none",
        help="Build mean/std bank from official voices; use as init + optional norm",
    )
    ap.add_argument(
        "--norm-to-bank",
        action="store_true",
        help="After each step, project trained vector(s) onto bank mean/std (z-clip)",
    )
    ap.add_argument("--z-clip", type=float, default=2.5, help="Z-score clip for --norm-to-bank")
    ap.add_argument(
        "--bank-cache",
        type=str,
        default="",
        help="Optional path to save/load voice_bank.pt",
    )
    ap.add_argument(
        "--timbre-ckpt",
        type=str,
        default="",
        help="Frozen timbre .pt when --train-target prosody",
    )
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lambda-mel", type=float, default=None)
    ap.add_argument("--lambda-res", type=float, default=None)
    ap.add_argument("--lambda-smooth", type=float, default=0.0, help="Adjacent slot delta smoothness")
    ap.add_argument("--lambda-f0", type=float, default=None)
    ap.add_argument("--lambda-n", type=float, default=None)
    ap.add_argument("--lambda-slm", type=float, default=0.0, help="WavLM feature L1 weight")
    ap.add_argument(
        "--slm-device",
        type=str,
        default="cpu",
        help="Device for WavLM (use cpu to avoid GPU OOM)",
    )
    ap.add_argument("--slm-layers", type=str, default="4,8,12")
    ap.add_argument("--max-clips", type=int, default=64)
    ap.add_argument("--max-phonemes", type=int, default=100)
    ap.add_argument("--max-wav-sec", type=float, default=6.0)
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    # Fill unset knobs; explicit CLI values are never overwritten.
    if args.train_target == "prosody":
        # Prosody defaults: absolute free vector + F0/N (classic Kokoro heads)
        if args.lr is None:
            args.lr = 3e-4
        if args.lambda_mel is None:
            args.lambda_mel = 0.5
        if args.lambda_res is None:
            args.lambda_res = 0.05
        if args.lambda_f0 is None:
            args.lambda_f0 = 1.0
        if args.lambda_n is None:
            args.lambda_n = 1.0
        if args.steps is None:
            args.steps = 150
        if not args.out_dir:
            args.out_dir = str(
                HERE / "runs" / f"speaker_prosody_abs_{args.speaker}"
            )
        if not args.timbre_ckpt:
            mel_t = HERE / "runs" / f"speaker_ref_s_{args.speaker}" / "timbre.pt"
            args.timbre_ckpt = str(mel_t)
    else:
        if args.lr is None:
            args.lr = 1e-2
        if args.lambda_mel is None:
            args.lambda_mel = 1.0
        if args.lambda_res is None:
            args.lambda_res = 0.1
        if args.lambda_f0 is None:
            args.lambda_f0 = 0.0
        if args.lambda_n is None:
            args.lambda_n = 0.0
        if args.steps is None:
            args.steps = 300

    out_dir = Path(args.out_dir) if args.out_dir else (
        HERE / "runs" / f"speaker_ref_s_{args.speaker}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} speaker={args.speaker} target={args.train_target} "
        f"prosody_mode={args.prosody_mode} baseline={args.baseline} "
        f"lr={args.lr} mel={args.lambda_mel} res={args.lambda_res} "
        f"smooth={args.lambda_smooth} f0={args.lambda_f0} n={args.lambda_n} "
        f"slm={args.lambda_slm} out={out_dir}",
        flush=True,
    )

    slm = None
    if args.lambda_slm > 0:
        from timbre_encoder.slm_loss import WavLMFeatureLoss

        layers = [int(x) for x in args.slm_layers.split(",") if x.strip()]
        print(f"Loading WavLM on {args.slm_device} layers={layers}...", flush=True)
        slm = WavLMFeatureLoss(layers=layers, device=args.slm_device)

    rows = load_speaker_rows(
        Path(args.manifest),
        args.speaker,
        max_phonemes=args.max_phonemes,
        max_clips=args.max_clips + args.holdout,
        seed=args.seed,
    )
    if len(rows) < args.holdout + 1:
        raise SystemExit(
            f"Need at least {args.holdout + 1} full-cache clips for speaker {args.speaker}, got {len(rows)}"
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

    # train() for cuDNN LSTM backward into style vector; weights frozen
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    prosody0: torch.Tensor | None = None
    if args.train_target == "timbre":
        if bank is not None:
            timbre0 = bank.timbre_mean.detach().clone()
            print("init timbre = bank.timbre_mean", flush=True)
        else:
            timbre0 = baseline_pack[0, 0, :128].detach().clone()
        timbre = timbre0.clone().requires_grad_(True)
        prosody_delta = None
        trainable = [timbre]
    else:
        timbre = load_timbre_ckpt(Path(args.timbre_ckpt), device).detach()
        if args.norm_to_bank and bank is not None:
            timbre = normalize_to_bank(timbre, bank.timbre_mean, bank.timbre_std, args.z_clip)
            print("frozen timbre projected onto bank", flush=True)
        n_slots = baseline_pack.shape[0]
        if args.prosody_mode == "absolute":
            if bank is not None:
                prosody0 = bank.prosody_mean.detach().clone()
                print("init absolute prosody = bank.prosody_mean", flush=True)
            else:
                prosody0 = baseline_pack[0, 0, 128:].detach().clone()
                print("init absolute prosody = baseline slot0", flush=True)
            prosody_delta = prosody0.clone().requires_grad_(True)
            print(
                f"frozen timbre from {args.timbre_ckpt}; "
                f"train absolute prosody[128] (no residual)",
                flush=True,
            )
        elif args.prosody_mode == "slot":
            prosody_delta = torch.zeros(n_slots, 128, device=device, requires_grad=True)
            print(
                f"frozen timbre from {args.timbre_ckpt}; "
                f"train per-slot prosody delta [{n_slots},128]",
                flush=True,
            )
        else:
            prosody_delta = torch.zeros(128, device=device, requires_grad=True)
            print(f"frozen timbre from {args.timbre_ckpt}; train shared prosody delta", flush=True)
        trainable = [prosody_delta]

    opt = torch.optim.Adam(trainable, lr=args.lr)
    rng = random.Random(args.seed)
    history: list[dict[str, float]] = []
    skipped = 0

    for step in range(1, args.steps + 1):
        row = train_rows[rng.randrange(len(train_rows))]
        phonemes = (row.get("phonemes_st2") or "").strip()
        d_gt = parse_d_gt(row.get("d_gt"))
        if not phonemes or not d_gt:
            skipped += 1
            continue

        input_ids, ids, dropped = kokoro_ids(model, phonemes, device)
        if dropped or len(ids) != len(phonemes):
            skipped += 1
            continue
        aln = build_alignment(d_gt, device)
        if aln.shape[1] != input_ids.shape[1]:
            skipped += 1
            continue

        try:
            wave = load_wav(row["audio_path"], args.max_wav_sec).to(device)
        except Exception:
            skipped += 1
            continue
        if wave.numel() < SAMPLE_RATE // 4:
            skipped += 1
            continue

        slot = pack_slot(baseline_pack, len(phonemes))
        idx = max(0, min(len(phonemes) - 1, baseline_pack.shape[0] - 1))
        base_p = slot[0, 128:].detach()
        if args.train_target == "timbre":
            ref_s = make_ref_s(timbre, base_p)
            loss_res = (timbre - timbre0).pow(2).mean()
            loss_smooth = ref_s.new_zeros(())
        else:
            assert prosody_delta is not None
            if args.prosody_mode == "absolute":
                ref_s = make_ref_s(timbre, prosody_delta)
                loss_res = (prosody_delta - prosody0).pow(2).mean()
                loss_smooth = ref_s.new_zeros(())
            elif args.prosody_mode == "slot":
                d_slot = prosody_delta[idx]
                ref_s = make_ref_s(timbre, base_p + d_slot)
                loss_res = d_slot.pow(2).mean()
                # Smooth only around touched slot (cheap local)
                loss_smooth = ref_s.new_zeros(())
                if args.lambda_smooth > 0 and idx + 1 < prosody_delta.shape[0]:
                    loss_smooth = loss_smooth + (prosody_delta[idx] - prosody_delta[idx + 1]).pow(2).mean()
                if args.lambda_smooth > 0 and idx > 0:
                    loss_smooth = loss_smooth + (prosody_delta[idx] - prosody_delta[idx - 1]).pow(2).mean()
            else:
                ref_s = make_ref_s(timbre, base_p + prosody_delta)
                loss_res = prosody_delta.pow(2).mean()
                loss_smooth = ref_s.new_zeros(())

        audio_pred, f0_pred, n_pred, _ = model.forward_with_alignment(input_ids, ref_s, aln)
        if audio_pred.dim() == 0 or audio_pred.numel() < 100:
            skipped += 1
            continue

        pred_mel = waveform_to_mel(
            audio_pred.unsqueeze(0) if audio_pred.dim() == 1 else audio_pred,
            device=device,
        )
        tgt_mel = waveform_to_mel(wave.unsqueeze(0), device=device)
        t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
        if t < 4:
            skipped += 1
            continue
        if args.lambda_mel > 0:
            loss_mel = F.l1_loss(pred_mel[..., :t], tgt_mel[..., :t].detach())
        else:
            loss_mel = ref_s.new_zeros(())

        loss_slm = ref_s.new_zeros(())
        if slm is not None and args.lambda_slm > 0:
            pred_w = audio_pred.view(1, -1) if audio_pred.dim() == 1 else audio_pred
            real_w = wave.view(1, -1)
            # Match lengths before SLM
            tw = min(pred_w.shape[-1], real_w.shape[-1])
            loss_slm = slm(pred_w[..., :tw], real_w[..., :tw].detach())

        loss_f0 = ref_s.new_zeros(())
        loss_n = ref_s.new_zeros(())
        if args.lambda_f0 > 0 or args.lambda_n > 0:
            f0_tgt, e_tgt = extract_f0_n(wave.detach().cpu().numpy())
            f0_tgt = f0_tgt.to(device)
            e_tgt = e_tgt.to(device)
            f0_p = f0_pred.squeeze()
            n_p = n_pred.squeeze()
            if f0_p.dim() > 1:
                f0_p = f0_p[0]
            if n_p.dim() > 1:
                n_p = n_p[0]
            if args.lambda_f0 > 0 and f0_p.numel() > 4 and f0_tgt.numel() > 4:
                f0_pi = F.interpolate(
                    f0_p.view(1, 1, -1), size=f0_tgt.numel(), mode="linear", align_corners=False
                ).view(-1)
                # scale-match (Kokoro F0 units ≠ Hz)
                f0_tgt_n = f0_tgt / 500.0
                scale = f0_tgt_n.mean() / (f0_pi.detach().abs().mean() + 1e-3)
                loss_f0 = F.l1_loss(f0_pi * scale, f0_tgt_n)
            if args.lambda_n > 0 and n_p.numel() > 4 and e_tgt.numel() > 4:
                n_pi = F.interpolate(
                    n_p.view(1, 1, -1), size=e_tgt.numel(), mode="linear", align_corners=False
                ).view(-1)
                e_n = (e_tgt - e_tgt.mean()) / (e_tgt.std() + 1e-3)
                n_n = (n_pi - n_pi.mean()) / (n_pi.std() + 1e-3)
                loss_n = F.l1_loss(n_n, e_n)

        loss = (
            args.lambda_mel * loss_mel
            + args.lambda_res * loss_res
            + args.lambda_smooth * loss_smooth
            + args.lambda_f0 * loss_f0
            + args.lambda_n * loss_n
            + args.lambda_slm * loss_slm
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Keep trained style inside official voice distribution
        if args.norm_to_bank and bank is not None:
            with torch.no_grad():
                if args.train_target == "timbre":
                    timbre.copy_(
                        normalize_to_bank(timbre, bank.timbre_mean, bank.timbre_std, args.z_clip)
                    )
                elif prosody_delta is not None:
                    if args.prosody_mode == "absolute":
                        prosody_delta.copy_(
                            normalize_to_bank(
                                prosody_delta, bank.prosody_mean, bank.prosody_std, args.z_clip
                            )
                        )
                    elif args.prosody_mode == "slot":
                        # only the slot we just used
                        abs_p = base_p + prosody_delta[idx]
                        abs_n = normalize_to_bank(
                            abs_p, bank.prosody_mean, bank.prosody_std, args.z_clip
                        )
                        prosody_delta[idx].copy_(abs_n - base_p)
                    else:
                        abs_p = base_p + prosody_delta
                        abs_n = normalize_to_bank(
                            abs_p, bank.prosody_mean, bank.prosody_std, args.z_clip
                        )
                        prosody_delta.copy_(abs_n - base_p)

        hist = {
            "step": float(step),
            "loss": float(loss.detach().item()),
            "mel": float(loss_mel.detach().item()),
            "res": float(loss_res.detach().item()),
            "f0": float(loss_f0.detach().item()) if torch.is_tensor(loss_f0) else float(loss_f0),
            "n": float(loss_n.detach().item()) if torch.is_tensor(loss_n) else float(loss_n),
            "slm": float(loss_slm.detach().item()) if torch.is_tensor(loss_slm) else float(loss_slm),
        }
        history.append(hist)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(
                f"step={step}/{args.steps} loss={hist['loss']:.4f} mel={hist['mel']:.4f} "
                f"slm={hist['slm']:.4f} f0={hist['f0']:.4f} n={hist['n']:.4f} "
                f"res={hist['res']:.4f} skipped={skipped}",
                flush=True,
            )

    timbre_cpu = timbre.detach().cpu()
    delta_cpu = prosody_delta.detach().cpu() if prosody_delta is not None else None
    ckpt = {
        "timbre": timbre_cpu,
        "prosody_delta": delta_cpu,
        "prosody_mode": args.prosody_mode if args.train_target == "prosody" else None,
        "speaker": args.speaker,
        "baseline": args.baseline,
        "train_target": args.train_target,
    }
    torch.save(ckpt, out_dir / "style.pt")
    torch.save({"timbre": timbre_cpu, "speaker": args.speaker, "baseline": args.baseline}, out_dir / "timbre.pt")
    if delta_cpu is not None:
        torch.save(delta_cpu, out_dir / "prosody_delta.pt")
        if args.prosody_mode == "absolute":
            torch.save(
                {"prosody": delta_cpu, "speaker": args.speaker, "baseline": args.baseline},
                out_dir / "prosody.pt",
            )
            pack = assemble_pack_both(timbre_cpu, delta_cpu, baseline_pack.cpu(), mode="replace")
        else:
            pack_mode = "slot_delta" if args.prosody_mode == "slot" else "delta"
            pack = assemble_pack_both(timbre_cpu, delta_cpu, baseline_pack.cpu(), mode=pack_mode)
    else:
        pack = assemble_voice_pack(timbre_cpu, baseline_pack.cpu())
    torch.save(pack, out_dir / "voice_pack.pt")

    first_n = history[:50]
    last_n = history[-50:] if len(history) >= 50 else history

    def avg(key: str, xs: list[dict[str, float]]) -> float:
        if not xs:
            return float("nan")
        return sum(h[key] for h in xs) / len(xs)

    if args.train_target == "timbre":
        vec_l2 = float((timbre_cpu - timbre0.cpu()).pow(2).sum().sqrt())
        touched = 0
    else:
        assert delta_cpu is not None
        if args.prosody_mode == "absolute":
            touched = 1
            vec_l2 = float((delta_cpu - prosody0.detach().cpu()).pow(2).sum().sqrt())
        elif delta_cpu.dim() == 2:
            slot_norm = delta_cpu.pow(2).sum(dim=-1).sqrt()
            touched = int((slot_norm > 1e-4).sum().item())
            vec_l2 = float(slot_norm.mean())
        else:
            touched = 1
            vec_l2 = float(delta_cpu.pow(2).sum().sqrt())

    summary = {
        "speaker": args.speaker,
        "train_target": args.train_target,
        "prosody_mode": args.prosody_mode,
        "baseline": args.baseline,
        "steps": args.steps,
        "lr": args.lr,
        "lambda_slm": args.lambda_slm,
        "train_clips": len(train_rows),
        "skipped": skipped,
        "mel_first50": avg("mel", first_n),
        "mel_last50": avg("mel", last_n),
        "slm_last50": avg("slm", last_n),
        "res_last50": avg("res", last_n),
        "vec_l2_from_start": vec_l2,
        "slots_touched": touched,
    }
    print("summary", summary, flush=True)
    (out_dir / "summary.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in summary.items()) + "\n",
        encoding="utf-8",
    )

    model.eval()
    from timbre_encoder.slm_loss import WavLMFeatureLoss

    print("Loading WavLM scorer (CPU) for listen page...", flush=True)
    slm_scorer = WavLMFeatureLoss(layers=[6, 12], device="cpu")
    write_listen(
        out_dir,
        model,
        baseline_pack,
        timbre,
        prosody_delta,
        holdout,
        device,
        args.max_wav_sec,
        title=f"Speaker {args.speaker} {args.train_target} mode={args.prosody_mode} base={args.baseline} lr={args.lr}",
        prosody_mode=args.prosody_mode if args.train_target == "prosody" else "shared",
        slm_scorer=slm_scorer,
    )
    print(f"saved {out_dir} listen={out_dir / 'listen_test' / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
