"""
Reproduce Fabul8/kokoro-experiments voice-embedding approach and evaluate prosody.

Their method (VOICE_EMBEDDING_OPTIMIZER.md):
  - Optimize a single frame of ref_s [256] (decoder[:128] + prosody[128:])
  - Replicate that frame to all 510 slots (claim: frames ~identical)
  - Match target audio via speaker embedding (WavLM) / we use mel L1 + autodiff

We add a control that matches our docs: timbre-only + baseline length-aware prosody.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
VOICES_CACHE = Path(__file__).resolve().parent / "voices"
SAMPLE_RATE = 24000

US_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

SHORT_TEXT = "Hello, this is a test of the voice optimization system."
LONG_TEXT = (
    "The sky above the port was the color of television, tuned to a dead channel. "
    "It was a Sprawl voice and a Sprawl joke. You could drink there for a week "
    "and never hear two words in Japanese. These decisions were to have an enormous "
    "impact for centuries to come, and the basilica aptly filled that need."
)


def load_voice(name: str) -> torch.Tensor:
    VOICES_CACHE.mkdir(parents=True, exist_ok=True)
    local = VOICES_CACHE / f"{name}.pt"
    if not local.exists():
        path = hf_hub_download("hexgrad/Kokoro-82M", f"voices/{name}.pt")
        torch.save(torch.load(path, weights_only=True), local)
    return torch.load(local, weights_only=True)


def mel_spec(waveform: torch.Tensor) -> torch.Tensor:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=80
    ).to(waveform.device)
    return torch.log(torch.clamp(mel(waveform), min=1e-5))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def analyze_claims(voices: list[str], frame_idx: int = 50) -> dict:
    """Reproduce Fabul8 variance + within-voice frame similarity stats."""
    packs = {n: load_voice(n) for n in voices}
    emb = []
    for v in packs.values():
        v = v.squeeze(1) if v.dim() == 3 else v
        emb.append(v[min(frame_idx, v.shape[0] - 1)].numpy())
    emb = np.stack(emb)
    var = emb.var(axis=0)
    decoder_var = float(var[:128].mean())
    prosody_var = float(var[128:].mean())

    # Within-voice consecutive-frame similarity (their ~99.9978% claim)
    consec = []
    slot_prosody = []
    slot_timbre = []
    for v in packs.values():
        v = v.squeeze(1) if v.dim() == 3 else v  # [510, 256]
        for i in range(min(100, v.shape[0] - 1)):
            consec.append(cosine(v[i], v[i + 1]))
        # Length-slot drift: frame 20 vs 200 vs 400
        for a, b in [(20, 200), (50, 400), (20, 400)]:
            if v.shape[0] > b:
                slot_timbre.append(cosine(v[a, :128], v[b, :128]))
                slot_prosody.append(cosine(v[a, 128:], v[b, 128:]))

    report = {
        "n_voices": len(packs),
        "frame_idx": frame_idx,
        "decoder_mean_var": decoder_var,
        "prosody_mean_var": prosody_var,
        "prosody_decoder_ratio": prosody_var / max(decoder_var, 1e-12),
        "consec_frame_mean_cosine": float(np.mean(consec)),
        "slot_timbre_mean_cosine": float(np.mean(slot_timbre)),
        "slot_prosody_mean_cosine": float(np.mean(slot_prosody)),
        "top10_dims": np.argsort(var)[::-1][:10].tolist(),
        "top10_in_prosody": int(sum(i >= 128 for i in np.argsort(var)[::-1][:10])),
    }
    return report


def phonemes_and_ids(pipeline: KPipeline, model: KModel, text: str, device: str):
    phonemes = ""
    for _, ps, _ in pipeline.en_tokenize(pipeline.g2p(text)[1]):
        phonemes += ps
    phonemes = phonemes[:510]
    ids = [i for i in map(model.vocab.get, phonemes) if i is not None]
    input_ids = torch.LongTensor([[0, *ids, 0]]).to(device)
    return phonemes, input_ids


def assemble_fabul8(frame256: torch.Tensor) -> torch.Tensor:
    """Their voice_to_full_sequence: replicate one frame to 510."""
    if frame256.dim() == 2:
        frame256 = frame256.squeeze(0)
    return frame256.view(1, 1, 256).expand(510, 1, 256).contiguous()


def assemble_timbre_plus_baseline(timbre128: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    if timbre128.dim() == 2:
        timbre128 = timbre128[0]
    pack = baseline.clone()
    pack[:, :, :128] = timbre128[:128].view(1, 1, 128).expand(510, 1, 128)
    return pack


@torch.no_grad()
def synth(pipeline: KPipeline, text: str, voice, out_wav: Path):
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=1):
        if audio is not None:
            chunks.append(audio.detach().cpu().numpy() if torch.is_tensor(audio) else audio)
    if not chunks:
        raise RuntimeError(f"No audio for {out_wav}")
    wav = np.concatenate(chunks)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, wav, SAMPLE_RATE)
    return wav


def optimize_fabul8_style(
    device: str,
    target_voice_name: str = "af_bella",
    start_voice_name: str = "am_michael",
    epochs: int = 40,
    lr: float = 0.05,
) -> dict:
    """
    Fabul8 conceptual match with autodiff (faster than their finite-diff WavLM):
    optimize full 256 from start voice to match target-generated audio via mel L1,
    then expand to 510.
    """
    pipeline = KPipeline(lang_code="a", model=False)
    # train() needed so cuDNN LSTM backward works; weights stay frozen
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).train()
    for p in model.parameters():
        p.requires_grad_(False)

    text = SHORT_TEXT
    phonemes, input_ids = phonemes_and_ids(pipeline, model, text, device)
    idx = max(len(phonemes) - 1, 0)

    target_pack = pipeline.load_voice(target_voice_name).to(device)
    start_pack = pipeline.load_voice(start_voice_name).to(device)
    baseline_heart = pipeline.load_voice("af_heart").to(device)

    with torch.no_grad():
        target_audio, target_dur = model.forward_with_tokens_grad(
            input_ids, target_pack[idx], target_dur=None
        )
        target_mel = mel_spec(target_audio.unsqueeze(0))

    # --- A) Fabul8: full 256 ---
    ref_full = start_pack[idx].clone().detach().requires_grad_(True)
    opt_full = torch.optim.Adam([ref_full], lr=lr)
    hist_full = []
    for ep in range(epochs):
        opt_full.zero_grad()
        pred, _ = model.forward_with_tokens_grad(input_ids, ref_full, target_dur=target_dur)
        pred_mel = mel_spec(pred.unsqueeze(0))
        t = target_mel.size(-1)
        pred_mel = pred_mel[..., :t] if pred_mel.size(-1) >= t else F.pad(pred_mel, (0, t - pred_mel.size(-1)))
        loss = F.l1_loss(pred_mel, target_mel)
        loss.backward()
        opt_full.step()
        hist_full.append(float(loss.item()))
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[fabul8-full] epoch {ep:03d} loss={loss.item():.4f}", flush=True)

    # --- B) Control: timbre-only, freeze prosody from heart ---
    timbre = start_pack[idx, :, :128].clone().detach().requires_grad_(True)
    prosody_fixed = baseline_heart[idx, :, 128:].detach()
    opt_t = torch.optim.Adam([timbre], lr=lr)
    hist_t = []
    for ep in range(epochs):
        opt_t.zero_grad()
        ref_s = torch.cat([timbre, prosody_fixed], dim=-1)
        pred, _ = model.forward_with_tokens_grad(input_ids, ref_s, target_dur=target_dur)
        pred_mel = mel_spec(pred.unsqueeze(0))
        t = target_mel.size(-1)
        pred_mel = pred_mel[..., :t] if pred_mel.size(-1) >= t else F.pad(pred_mel, (0, t - pred_mel.size(-1)))
        loss = F.l1_loss(pred_mel, target_mel)
        loss.backward()
        opt_t.step()
        hist_t.append(float(loss.item()))
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[timbre-only]  epoch {ep:03d} loss={loss.item():.4f}", flush=True)

    pack_fabul8 = assemble_fabul8(ref_full.detach().cpu())
    pack_control = assemble_timbre_plus_baseline(timbre.detach().cpu(), baseline_heart.detach().cpu())

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(pack_fabul8, OUT / "voice_fabul8_expand.pt")
    torch.save(pack_control, OUT / "voice_timbre_plus_heart_prosody.pt")
    torch.save(ref_full.detach().cpu(), OUT / "frame_fabul8_256.pt")

    # Slot drift of optimized Fabul8 pack vs official target pack
    tgt = target_pack.detach().cpu().squeeze(1)
    fab = pack_fabul8.squeeze(1)
    ctrl = pack_control.squeeze(1)
    slot_stats = {}
    for slot in [20, 50, 100, 200, 400]:
        slot_stats[str(slot)] = {
            "fabul8_vs_target_prosody_cos": cosine(fab[slot, 128:], tgt[slot, 128:]),
            "control_vs_target_prosody_cos": cosine(ctrl[slot, 128:], tgt[slot, 128:]),
            "fabul8_vs_target_timbre_cos": cosine(fab[slot, :128], tgt[slot, :128]),
            "control_vs_target_timbre_cos": cosine(ctrl[slot, :128], tgt[slot, :128]),
            "fabul8_slot_vs_50_prosody_cos": cosine(fab[slot, 128:], fab[50, 128:]),
            "official_slot_vs_50_prosody_cos": cosine(tgt[slot, 128:], tgt[50, 128:]),
        }

    return {
        "target_voice": target_voice_name,
        "start_voice": start_voice_name,
        "epochs": epochs,
        "hist_fabul8": hist_full,
        "hist_timbre_only": hist_t,
        "final_loss_fabul8": hist_full[-1],
        "final_loss_timbre_only": hist_t[-1],
        "slot_stats": slot_stats,
        "train_phoneme_len": len(phonemes),
    }


def make_listen(pipeline: KPipeline, meta: dict):
    listen = OUT / "listen"
    listen.mkdir(parents=True, exist_ok=True)
    pack_fab = torch.load(OUT / "voice_fabul8_expand.pt", weights_only=True)
    pack_ctrl = torch.load(OUT / "voice_timbre_plus_heart_prosody.pt", weights_only=True)
    target = meta["target_voice"]

    rows = []
    for label, text in [("short", SHORT_TEXT), ("long", LONG_TEXT)]:
        paths = {
            "target": listen / f"{label}_target_{target}.wav",
            "fabul8_expand": listen / f"{label}_fabul8_expand.wav",
            "timbre_heart_p": listen / f"{label}_timbre_plus_heart_prosody.wav",
            "start_michael": listen / f"{label}_start_am_michael.wav",
        }
        synth(pipeline, text, target, paths["target"])
        synth(pipeline, text, pack_fab, paths["fabul8_expand"])
        synth(pipeline, text, pack_ctrl, paths["timbre_heart_p"])
        synth(pipeline, text, "am_michael", paths["start_michael"])
        rows.append((label, text, paths))

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Fabul8-style prosody reproduction</title>",
        "<style>body{font-family:sans-serif;max-width:920px;margin:1.5rem auto}",
        "audio{width:100%;margin:.2rem 0}.card{margin:1.2rem 0;padding:1rem;border:1px solid #ccc}",
        "code{background:#f4f4f4;padding:0 .25rem}</style></head><body>",
        "<h1>Fabul8-style voice opt — prosody check</h1>",
        "<p>Match: optimize full <code>ref_s[256]</code>, then <code>expand(510)</code> "
        "(their assumption that frames are interchangeable). "
        "Control: timbre-only + <code>af_heart</code> length-aware prosody.</p>",
        f"<p>Train text (short): <i>{SHORT_TEXT}</i><br>"
        f"Target={meta['target_voice']} · start={meta['start_voice']} · "
        f"epochs={meta['epochs']} · final mel L1: "
        f"fabul8={meta['final_loss_fabul8']:.4f}, "
        f"timbre-only={meta['final_loss_timbre_only']:.4f}</p>",
    ]
    for label, text, paths in rows:
        html.append(f"<div class='card'><h2>{label}</h2><p>{text}</p>")
        for k, p in paths.items():
            html.append(f"<p><b>{k}</b></p><audio controls src='{p.name}'></audio>")
        html.append("</div>")
    html.append("</body></html>")
    (listen / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"Listen page: {listen / 'index.html'}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    OUT.mkdir(parents=True, exist_ok=True)

    print("\n=== 1) Analyze Fabul8 claims on official packs ===")
    report = analyze_claims(US_VOICES)
    (OUT / "claims_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    print("\n=== 2) Optimize Fabul8-style vs timbre-only control ===")
    meta = optimize_fabul8_style(device=device, epochs=40, lr=0.05)
    (OUT / "opt_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("slot_stats:", json.dumps(meta["slot_stats"], indent=2))

    print("\n=== 3) Listen synthesis (short + long) ===")
    pipeline = KPipeline(lang_code="a")
    make_listen(pipeline, meta)
    print("Done.")


if __name__ == "__main__":
    main()
