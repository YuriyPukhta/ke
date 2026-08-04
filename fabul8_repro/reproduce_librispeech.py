"""
Fabul8-style voice embedding optimization on LibriTTS / LibriSpeech speakers.

Method (same as kokoro-experiments optimizer conceptually):
  - Optimize one ref_s frame [256] (timbre + prosody)
  - Expand to [510, 1, 256]
  - Match real speaker audio (mel L1 + forced GT duration)

Also trains a control: timbre-only + baseline length-aware prosody pack.

  .venv\\Scripts\\python.exe fabul8_repro/reproduce_librispeech.py --speaker 6209
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402
from timbre_encoder.aligner import FrozenTextAligner  # noqa: E402
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.voice_pack import assemble_voice_pack  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent / "outputs"


def load_speaker_rows(
    csv_path: Path,
    speaker: str,
    max_phonemes: int,
    seed: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("speaker_id") != speaker:
                continue
            if row.get("align_status") != "ok":
                continue
            ph = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            if not ph or not d_gt or len(d_gt) != len(ph):
                continue
            if not (2 <= len(ph) <= max_phonemes):
                continue
            if not Path(row["audio_path"]).exists():
                continue
            rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def kokoro_ids(model: KModel, phonemes: str, device: torch.device):
    ids = [model.vocab.get(p) for p in phonemes]
    if any(i is None for i in ids):
        return None
    return torch.LongTensor([[0, *ids, 0]]).to(device)


def build_alignment(d_gt: list[int], device: torch.device) -> torch.Tensor:
    d_list = [1] + [max(1, int(x)) for x in d_gt] + [1]
    return FrozenTextAligner.d_gt_to_a_gt(d_list, device=device)


def load_wav(path: str, max_sec: float) -> torch.Tensor:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    max_n = int(max_sec * SAMPLE_RATE)
    if wav.shape[0] > max_n:
        wav = wav[:max_n]
    return torch.from_numpy(wav.astype("float32"))


def assemble_fabul8(frame256: torch.Tensor) -> torch.Tensor:
    if frame256.dim() == 2:
        frame256 = frame256.squeeze(0)
    return frame256.view(1, 1, 256).expand(510, 1, 256).contiguous().clone()


def prepare_clip(model: KModel, row: dict, device: torch.device, max_sec: float):
    phonemes = row["phonemes_st2"].strip()
    d_gt = parse_d_gt(row["d_gt"])
    assert d_gt is not None
    input_ids = kokoro_ids(model, phonemes, device)
    if input_ids is None:
        return None
    aln = build_alignment(d_gt, device)
    if aln.shape[1] != input_ids.shape[1]:
        return None
    wav = load_wav(row["audio_path"], max_sec).to(device)
    if wav.numel() < SAMPLE_RATE // 4:
        return None
    target_mel = waveform_to_mel(wav.cpu()).to(device)  # [1, 80, T]
    return {
        "row": row,
        "phonemes": phonemes,
        "input_ids": input_ids,
        "aln": aln,
        "wav": wav,
        "target_mel": target_mel,
        "text": row["text"],
    }


def mel_loss(pred_audio: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
    if pred_audio.dim() == 1:
        pred_audio = pred_audio.unsqueeze(0)
    pred_mel = waveform_to_mel(pred_audio, device=pred_audio.device)
    t = min(pred_mel.size(-1), target_mel.size(-1))
    return F.l1_loss(pred_mel[..., :t], target_mel[..., :t])


def optimize_frame(
    model: KModel,
    clips: list[dict],
    init_frame: torch.Tensor,
    epochs: int,
    lr: float,
    mode: str,
    baseline_pack: torch.Tensor | None,
) -> tuple[torch.Tensor, list[float]]:
    """
    mode=fabul8: optimize full [1,256]
    mode=timbre: optimize [:128], freeze prosody from baseline slot of first clip length
    """
    device = init_frame.device
    if mode == "fabul8":
        ref = init_frame.clone().detach().requires_grad_(True)
        params = [ref]
    else:
        assert baseline_pack is not None
        timbre = init_frame[:, :128].clone().detach().requires_grad_(True)
        # freeze prosody from baseline at a mid slot; slot-specific applied at forward
        params = [timbre]
        ref = None  # assembled per step

    opt = torch.optim.Adam(params, lr=lr)
    hist: list[float] = []

    for ep in range(epochs):
        opt.zero_grad()
        losses = []
        for clip in clips:
            if mode == "fabul8":
                ref_s = ref
            else:
                # use length-appropriate baseline prosody for this clip
                idx = max(0, min(len(clip["phonemes"]) - 1, baseline_pack.shape[0] - 1))
                prosody = baseline_pack[idx, :, 128:].to(device)
                if prosody.dim() == 1:
                    prosody = prosody.unsqueeze(0)
                ref_s = torch.cat([timbre, prosody], dim=-1)

            audio, _, _, _ = model.forward_with_alignment(
                clip["input_ids"], ref_s, clip["aln"]
            )
            losses.append(mel_loss(audio.unsqueeze(0) if audio.dim() == 1 else audio, clip["target_mel"]))

        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[{mode}] epoch {ep:03d} loss={loss.item():.4f}", flush=True)

    if mode == "fabul8":
        return ref.detach(), hist
    # return full frame = timbre + prosody from heart mid slot for saving single frame
    idx = 50
    prosody = baseline_pack[idx, :, 128:].to(device)
    if prosody.dim() == 1:
        prosody = prosody.unsqueeze(0)
    return torch.cat([timbre.detach(), prosody], dim=-1), hist


@torch.no_grad()
def synth_forced(model: KModel, clip: dict, ref_s: torch.Tensor) -> np.ndarray:
    audio, _, _, _ = model.forward_with_alignment(
        clip["input_ids"], ref_s, clip["aln"]
    )
    return audio.detach().cpu().numpy()


@torch.no_grad()
def synth_pipeline(pipeline: KPipeline, text: str, voice) -> np.ndarray:
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=1):
        if audio is not None:
            a = audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)
            chunks.append(a)
    return np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE // 10, dtype=np.float32)


def write_listen(
    out_dir: Path,
    model: KModel,
    pipeline: KPipeline,
    holdout: list[dict],
    pack_fabul8: torch.Tensor,
    pack_control: torch.Tensor,
    baseline_name: str,
    speaker: str,
    meta: dict,
):
    listen = out_dir / "listen"
    listen.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    cards = []
    for i, clip in enumerate(holdout):
        n = len(clip["phonemes"])
        idx = max(0, min(n - 1, 509))
        ref_wav = clip["wav"].detach().cpu().numpy()
        sf.write(listen / f"{i:02d}_ref.wav", ref_wav, SAMPLE_RATE)

        fab = synth_forced(model, clip, pack_fabul8[idx].to(device))
        ctrl = synth_forced(model, clip, pack_control[idx].to(device))
        base = synth_forced(model, clip, pipeline.load_voice(baseline_name)[idx].to(device))
        sf.write(listen / f"{i:02d}_fabul8_expand.wav", fab, SAMPLE_RATE)
        sf.write(listen / f"{i:02d}_timbre_plus_baseline_p.wav", ctrl, SAMPLE_RATE)
        sf.write(listen / f"{i:02d}_baseline_{baseline_name}.wav", base, SAMPLE_RATE)

        cards.append(
            f"<div class='card'><h3>#{i} holdout forced · ph={n} · slot={idx}</h3>"
            f"<p>{html_lib.escape(clip['text'])}</p>"
            f"<p><b>ref (LibriTTS)</b></p><audio controls src='{i:02d}_ref.wav'></audio>"
            f"<p><b>fabul8 expand</b></p><audio controls src='{i:02d}_fabul8_expand.wav'></audio>"
            f"<p><b>timbre + baseline prosody</b></p>"
            f"<audio controls src='{i:02d}_timbre_plus_baseline_p.wav'></audio>"
            f"<p><b>baseline {baseline_name}</b></p>"
            f"<audio controls src='{i:02d}_baseline_{baseline_name}.wav'></audio></div>"
        )

    # Free synth (no forced align) — short holdout text + longer unseen sentence
    long_text = (
        "The sky above the port was the color of television, tuned to a dead channel. "
        "It was a Sprawl voice and a Sprawl joke."
    )
    free_items = [
        ("free_short", holdout[0]["text"] if holdout else "Hello there."),
        ("free_long", long_text),
    ]
    for tag, text in free_items:
        sf.write(listen / f"{tag}_fabul8.wav", synth_pipeline(pipeline, text, pack_fabul8), SAMPLE_RATE)
        sf.write(
            listen / f"{tag}_timbre_baseline_p.wav",
            synth_pipeline(pipeline, text, pack_control),
            SAMPLE_RATE,
        )
        sf.write(
            listen / f"{tag}_baseline.wav",
            synth_pipeline(pipeline, text, baseline_name),
            SAMPLE_RATE,
        )
        cards.append(
            f"<div class='card'><h3>{tag} (pipeline, no forced align)</h3>"
            f"<p>{html_lib.escape(text)}</p>"
            f"<p><b>fabul8 expand</b></p><audio controls src='{tag}_fabul8.wav'></audio>"
            f"<p><b>timbre + baseline prosody</b></p>"
            f"<audio controls src='{tag}_timbre_baseline_p.wav'></audio>"
            f"<p><b>baseline</b></p><audio controls src='{tag}_baseline.wav'></audio></div>"
        )

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Fabul8-style on LibriTTS spk {speaker}</title>",
        "<style>body{font-family:sans-serif;max-width:920px;margin:1.5rem auto}"
        "audio{width:100%;margin:.2rem 0}.card{margin:1.2rem 0;padding:1rem;border:1px solid #ccc}"
        "code{background:#f4f4f4;padding:0 .25rem}</style></head><body>",
        f"<h1>Fabul8-style on LibriTTS speaker {html_lib.escape(speaker)}</h1>",
        "<p>Optimize full <code>ref_s[256]</code> on real clips, then "
        "<code>expand(510)</code>. Control: timbre-only + length-aware baseline prosody.</p>",
        f"<p>train_clips={meta['n_train']} · holdout={meta['n_holdout']} · "
        f"epochs={meta['epochs']} · baseline={html_lib.escape(baseline_name)} · "
        f"final loss fabul8={meta['final_fabul8']:.4f} · "
        f"timbre={meta['final_timbre']:.4f}</p>",
        *cards,
        "</body></html>",
    ]
    (listen / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"Listen: {listen / 'index.html'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default="6209")
    ap.add_argument(
        "--manifest",
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument("--baseline", default="af_heart", help="Kokoro voice for prosody baseline / start")
    ap.add_argument("--train-clips", type=int, default=6)
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--max-phonemes", type=int, default=80)
    ap.add_argument("--max-wav-sec", type=float, default=6.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_ROOT / f"librispeech_{args.speaker}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} speaker={args.speaker} out={out_dir}", flush=True)

    rows = load_speaker_rows(Path(args.manifest), args.speaker, args.max_phonemes, args.seed)
    need = args.train_clips + args.holdout
    if len(rows) < need:
        raise SystemExit(f"Need {need} clips for speaker {args.speaker}, got {len(rows)}")

    pipeline = KPipeline(lang_code="a", model=False)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).train()
    for p in model.parameters():
        p.requires_grad_(False)

    baseline_pack = pipeline.load_voice(args.baseline).to(device)
    prepared = []
    for row in rows:
        clip = prepare_clip(model, row, device, args.max_wav_sec)
        if clip is not None:
            prepared.append(clip)
        if len(prepared) >= need:
            break
    if len(prepared) < need:
        raise SystemExit(f"Only prepared {len(prepared)}/{need} clips")

    train = prepared[: args.train_clips]
    holdout = prepared[args.train_clips : args.train_clips + args.holdout]
    print(
        f"train={len(train)} holdout={len(holdout)} "
        f"ph_lens={[len(c['phonemes']) for c in train]}",
        flush=True,
    )

    # init from baseline mid-slot (Fabul8 uses median centroid; heart is fine start)
    init = baseline_pack[50].detach()
    if init.dim() == 1:
        init = init.unsqueeze(0)

    print("\n=== Fabul8: full 256 + later expand ===", flush=True)
    frame_fab, hist_fab = optimize_frame(
        model, train, init, args.epochs, args.lr, mode="fabul8", baseline_pack=None
    )
    print("\n=== Control: timbre-only + baseline prosody ===", flush=True)
    frame_ctrl, hist_ctrl = optimize_frame(
        model, train, init, args.epochs, args.lr, mode="timbre", baseline_pack=baseline_pack
    )

    pack_fab = assemble_fabul8(frame_fab.cpu())
    pack_ctrl = assemble_voice_pack(frame_ctrl[:, :128].cpu(), baseline_pack.cpu())

    torch.save(pack_fab, out_dir / "voice_fabul8_expand.pt")
    torch.save(pack_ctrl, out_dir / "voice_timbre_plus_baseline_p.pt")
    torch.save(frame_fab.cpu(), out_dir / "frame_fabul8_256.pt")

    meta = {
        "speaker": args.speaker,
        "baseline": args.baseline,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "epochs": args.epochs,
        "lr": args.lr,
        "final_fabul8": hist_fab[-1],
        "final_timbre": hist_ctrl[-1],
        "hist_fabul8": hist_fab,
        "hist_timbre": hist_ctrl,
        "train_texts": [c["text"] for c in train],
        "holdout_texts": [c["text"] for c in holdout],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # pipeline with model for free synth listen
    pipe_full = KPipeline(lang_code="a")
    write_listen(
        out_dir,
        model,
        pipe_full,
        holdout,
        pack_fab,
        pack_ctrl,
        args.baseline,
        args.speaker,
        meta,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
