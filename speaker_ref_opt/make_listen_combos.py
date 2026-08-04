"""
Rebuild listen_test for a trained run with all style combos:

  ref | timbre+base_p | base_a+prosody | opt | baseline

  .venv\\Scripts\\python.exe speaker_ref_opt/make_listen_combos.py \\
      --run-dir speaker_ref_opt/runs/speaker_ref_s_6209_f0n_accum_wavlm_bs8

  # all runs under speaker_ref_opt/runs/
  .venv\\Scripts\\python.exe speaker_ref_opt/make_listen_combos.py --all
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(HERE))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.voice_bank import US_VOICES, build_voice_bank, load_bank  # noqa: E402

from optimize_speaker_ref_s import (  # noqa: E402
    load_speaker_rows,
    load_timbre_ckpt,
    write_listen,
)


def load_style(run_dir: Path, device: torch.device):
    style_path = run_dir / "style.pt"
    if not style_path.is_file():
        raise SystemExit(f"missing {style_path}")
    obj = torch.load(style_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "timbre" in obj:
        timbre = obj["timbre"].detach().float().view(-1).to(device)
        prosody = obj.get("prosody_delta")
        if prosody is None and (run_dir / "prosody.pt").is_file():
            pobj = torch.load(run_dir / "prosody.pt", map_location=device, weights_only=False)
            if isinstance(pobj, dict) and "prosody" in pobj:
                prosody = pobj["prosody"]
            else:
                prosody = pobj
        prosody_mode = obj.get("prosody_mode") or "absolute"
        speaker = str(obj.get("speaker") or "6209")
        baseline = str(obj.get("baseline") or "mean")
        train_target = str(obj.get("train_target") or "")
    else:
        timbre = load_timbre_ckpt(style_path, device)
        prosody = None
        prosody_mode = "shared"
        speaker = "6209"
        baseline = "af_heart"
        train_target = "timbre"
    if prosody is not None:
        prosody = prosody.detach().float().to(device)
    return timbre, prosody, prosody_mode, speaker, baseline, train_target


def load_holdout(
    manifest: Path,
    speaker: str,
    holdout: int,
    max_clips: int,
    max_phonemes: int,
    seed: int,
) -> list[dict[str, str]]:
    rows = load_speaker_rows(
        manifest,
        speaker,
        max_phonemes=max_phonemes,
        max_clips=max_clips + holdout,
        seed=seed,
    )
    if len(rows) < holdout:
        raise SystemExit(f"need >= {holdout} clips for speaker {speaker}, got {len(rows)}")
    return rows[:holdout]


def rebuild_one(
    run_dir: Path,
    model: KModel,
    pipeline: KPipeline,
    bank_cache: Path | None,
    manifest: Path,
    device: torch.device,
    holdout: int,
    max_clips: int,
    max_phonemes: int,
    max_wav_sec: float,
    seed: int,
    slm_scorer,
) -> Path:
    timbre, prosody, prosody_mode, speaker, baseline, train_target = load_style(run_dir, device)
    if prosody is None:
        # timbre-only: still show combos; base_a+prosody will SKIP
        prosody_mode = "shared"

    if baseline == "mean":
        cache = bank_cache if bank_cache and bank_cache.is_file() else (run_dir / "voice_bank.pt")
        if cache.is_file():
            bank = load_bank(cache, device=device)
            baseline_pack = bank.mean_pack.to(device)
        else:
            bank = build_voice_bank(pipeline, US_VOICES).to(device)
            baseline_pack = bank.mean_pack.to(device)
    else:
        baseline_pack = pipeline.load_voice(baseline).to(device)
        if baseline_pack.dim() == 2:
            baseline_pack = baseline_pack.unsqueeze(1)

    hold = load_holdout(manifest, speaker, holdout, max_clips, max_phonemes, seed)
    title = f"{run_dir.name} · {train_target or '?'} · mode={prosody_mode} · base={baseline}"
    write_listen(
        run_dir,
        model,
        baseline_pack,
        timbre,
        prosody,
        hold,
        device,
        max_wav_sec,
        title=title,
        prosody_mode=prosody_mode if prosody is not None else "shared",
        slm_scorer=slm_scorer,
    )
    out = run_dir / "listen_test" / "index.html"
    print(f"wrote {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild listen pages with all style combos")
    ap.add_argument("--run-dir", type=str, default="", help="Single run directory")
    ap.add_argument("--all", action="store_true", help="All */style.pt under speaker_ref_opt/runs")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument("--bank-cache", type=str, default="")
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--max-clips", type=int, default=64)
    ap.add_argument("--max-phonemes", type=int, default=100)
    ap.add_argument("--max-wav-sec", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wavlm", action="store_true")
    args = ap.parse_args()

    if args.all:
        runs = sorted(
            p.parent for p in (HERE / "runs").glob("*/style.pt")
        )
    elif args.run_dir:
        runs = [Path(args.run_dir)]
    else:
        raise SystemExit("pass --run-dir PATH or --all")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} runs={len(runs)}", flush=True)
    pipeline = KPipeline(lang_code="a", model=False)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    slm_scorer = None
    if not args.no_wavlm:
        from timbre_encoder.slm_loss import WavLMFeatureLoss

        print("Loading WavLM scorer (CPU)...", flush=True)
        slm_scorer = WavLMFeatureLoss(layers=[6, 12], device="cpu")

    bank_cache = Path(args.bank_cache) if args.bank_cache else (
        HERE / "runs" / "speaker_ref_s_6209_bankmean_lr1e5" / "voice_bank.pt"
    )
    manifest = Path(args.manifest)

    for run_dir in runs:
        print(f"=== {run_dir} ===", flush=True)
        rebuild_one(
            run_dir,
            model,
            pipeline,
            bank_cache if bank_cache.is_file() else None,
            manifest,
            device,
            args.holdout,
            args.max_clips,
            args.max_phonemes,
            args.max_wav_sec,
            args.seed,
            slm_scorer,
        )


if __name__ == "__main__":
    main()
