"""
Clean-path experiments: why LibriTTS×Kokoro fails vs StyleTTS2 manifold.

Exp1: ST2 reconstruct with official style_encoder + GT alignment
Exp2: pretrained style vs random style mel/wave quality under ST2
Exp3: ST2 style stats vs Kokoro ref_s; try ST2→Kokoro (expect fail)

  python -m timbre_encoder.experiments.run_clean_path_exps
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from munch import Munch

ROOT = Path(__file__).resolve().parents[2]
ST2_ROOT = ROOT / "third_party" / "StyleTTS2"
CKPT_DIR = ROOT / "timbre_encoder" / "checkpoints" / "styletts2_libritts" / "Models" / "LibriTTS"
OUT = ROOT / "timbre_encoder" / "runs" / "clean_path_exps"

# stubs + StyleTTS2 on path
sys.path.insert(0, str(ROOT / "third_party"))
sys.path.insert(0, str(ST2_ROOT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    if isinstance(d, list):
        return [recursive_munch(v) for v in d]
    return d


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def load_st2(device: torch.device):
    # PyTorch>=2.6 defaults weights_only=True; ST2 ckpts need full unpickle.
    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load  # type: ignore[assignment]

    from models import build_model, load_ASR_models, load_F0_models
    from Utils.PLBERT.util import load_plbert

    config_path = CKPT_DIR / "config.yml"
    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    # paths relative to ST2_ROOT
    asr_path = str(ST2_ROOT / config["ASR_path"])
    asr_cfg = str(ST2_ROOT / config["ASR_config"])
    f0_path = str(ST2_ROOT / config["F0_path"])
    plbert_dir = str(ST2_ROOT / config["PLBERT_dir"])

    text_aligner = load_ASR_models(asr_path, asr_cfg)
    pitch_extractor = load_F0_models(f0_path)
    plbert = load_plbert(plbert_dir)

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    params = torch.load(CKPT_DIR / "epochs_2nd_00020.pth", map_location="cpu", weights_only=False)
    params = params["net"] if "net" in params else params
    for key in model:
        if key not in params:
            continue
        try:
            model[key].load_state_dict(params[key])
        except Exception:
            state = params[key]
            new_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
            model[key].load_state_dict(new_state, strict=False)
        model[key].eval().to(device)
        for p in model[key].parameters():
            p.requires_grad = False
    return model, config


def preprocess_wave(wave: np.ndarray, device: torch.device) -> torch.Tensor:
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300
    )
    mean, std = -4.0, 4.0
    wave_tensor = torch.from_numpy(wave).float()
    mel = torch.log(1e-5 + to_mel(wave_tensor).unsqueeze(0))
    mel = (mel - mean) / std
    return mel.to(device)


def text_to_st2_ids(text: str, device: torch.device) -> torch.Tensor:
    """Kokoro/misaki phonemes → StyleTTS2 TextCleaner ids (no espeak)."""
    from text_utils import TextCleaner
    from kokoro.pipeline import KPipeline
    from timbre_encoder.librispeech_dataset import text_to_kokoro_phonemes

    pipe = KPipeline(lang_code="a")
    phonemes = text_to_kokoro_phonemes(text, pipe)
    cleaner = TextCleaner()
    ids = cleaner(phonemes)
    if not ids:
        ids = [0]
    return torch.LongTensor(ids).unsqueeze(0).to(device), phonemes


@torch.no_grad()
def st2_reconstruct(model, mel: torch.Tensor, text_ids: torch.Tensor, style_mel: torch.Tensor | None = None):
    """
    Teacher-forced reconstruct: aligner + text_encoder + style_encoder + pitch from mel + decoder.
    style_mel: mel used for style (default: same mel). Returns waveform, metrics dict.
    """
    device = mel.device
    # ASRCNN stride-2: odd mel frames → mask length mel//2 mismatches encoder T (82 vs 81).
    if mel.shape[-1] % 2 == 1:
        mel = mel[..., :-1]
    if style_mel is None:
        style_mel = mel
    elif style_mel.shape[-1] % 2 == 1:
        style_mel = style_mel[..., :-1]

    texts = text_ids
    input_lengths = torch.LongTensor([texts.shape[1]]).to(device)
    mel_input_length = torch.LongTensor([mel.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    n_down = model.text_aligner.n_down
    mel_down = torch.clamp(mel_input_length // (2**n_down), min=1)
    mel_mask = length_to_mask(mel_down).to(device)

    # Match StyleTTS2 train_second.py attention post-process
    _, _, s2s_attn = model.text_aligner(mel, mel_mask, texts)
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)

    from monotonic_align import mask_from_lens, maximum_path

    mask_st = mask_from_lens(s2s_attn, input_lengths, mel_down)
    s2s_attn_mono = maximum_path(s2s_attn, mask_st)

    t_en = model.text_encoder(texts, input_lengths, text_mask)
    asr = t_en @ s2s_attn_mono

    F0_real, _, _ = model.pitch_extractor(mel.unsqueeze(1))
    from utils import log_norm

    real_norm = log_norm(mel.unsqueeze(1)).squeeze(1)
    s = model.style_encoder(style_mel.unsqueeze(1))

    # Decoder F0_conv/N_conv are stride-2: pass full-mel-rate curves (≈2 * asr_T).
    if not torch.is_tensor(F0_real):
        raise TypeError(f"unexpected F0 type {type(F0_real)}")
    if F0_real.dim() == 1:
        F0_real = F0_real.unsqueeze(0)
    if real_norm.dim() == 1:
        real_norm = real_norm.unsqueeze(0)

    T_asr = asr.shape[-1]
    T_f0 = 2 * T_asr

    def _match_full(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == T_f0:
            return x
        return F.interpolate(x.unsqueeze(1), size=T_f0, mode="linear", align_corners=False).squeeze(1)

    F0_use = _match_full(F0_real)
    N_use = _match_full(real_norm)

    y = model.decoder(asr, F0_use, N_use, s)
    return y.squeeze().cpu(), {
        "s_norm": float(s.norm()),
        "s_mean": float(s.mean()),
        "s_std": float(s.std()),
        "asr_T": int(T_asr),
        "mel_T": int(mel.shape[-1]),
    }


def wave_mel_l1(a: np.ndarray, b: np.ndarray, device: torch.device) -> float:
    from timbre_encoder.mel_preprocess import waveform_to_mel

    ta = torch.from_numpy(a).float()
    tb = torch.from_numpy(b).float()
    ma = waveform_to_mel(ta, device=device)
    mb = waveform_to_mel(tb, device=device)
    t = min(ma.shape[-1], mb.shape[-1])
    return float(F.l1_loss(ma[..., :t], mb[..., :t]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # Sample from filtered listen_test / LibriTTS
    wav_path = ROOT / "timbre_encoder" / "runs" / "dual_snr22" / "listen_test" / "01_spk6345_original.wav"
    text = "He hurriedly cut cake and pressed it upon her."
    if not wav_path.is_file():
        # fallback first filtered row
        import csv

        row = next(csv.DictReader(open(ROOT / "timbre_encoder" / "manifests" / "dev-clean.filtered.csv", encoding="utf-8")))
        wav_path = Path(row["audio_path"])
        text = row["text"]

    wave, _ = librosa.load(str(wav_path), sr=24000, mono=True)
    wave, _ = librosa.effects.trim(wave, top_db=30)
    sf.write(OUT / "00_original.wav", wave, 24000)
    print(f"sample={wav_path.name} sec={len(wave)/24000:.2f}", flush=True)

    results = {"sample": str(wav_path), "text": text, "device": str(device)}

    # ----- Exp0 reminder: Kokoro vs LibriTTS (known fail) -----
    print("\n=== Exp0: Kokoro af_heart vs LibriTTS (baseline fail) ===", flush=True)
    from kokoro.pipeline import KPipeline

    pipe = KPipeline(lang_code="a")
    chunks = []
    for r in pipe(text, voice="af_heart", speed=1):
        if r.audio is not None:
            chunks.append(r.audio.cpu().numpy())
    k_wav = np.concatenate(chunks)
    sf.write(OUT / "exp0_kokoro_af_heart.wav", k_wav, 24000)
    results["exp0_mel_l1_kokoro_vs_real"] = wave_mel_l1(k_wav, wave, device)
    print(f"mel L1 Kokoro vs real = {results['exp0_mel_l1_kokoro_vs_real']:.4f}", flush=True)

    # ----- Load StyleTTS2 -----
    print("\n=== Loading StyleTTS2-LibriTTS ===", flush=True)
    model, config = load_st2(device)
    mel = preprocess_wave(wave, device)
    text_ids, phonemes = text_to_st2_ids(text, device)
    print(f"mel={tuple(mel.shape)} text_ids={text_ids.shape}", flush=True)

    # ----- Exp1: ST2 reconstruct with official style -----
    print("\n=== Exp1: ST2 reconstruct (official style_encoder + a_gt) ===", flush=True)
    try:
        y1, meta1 = st2_reconstruct(model, mel, text_ids)
        y1n = y1.numpy()
        sf.write(OUT / "exp1_st2_recon_official_style.wav", y1n, 24000)
        results["exp1"] = meta1
        results["exp1_mel_l1_vs_real"] = wave_mel_l1(y1n, wave, device)
        print(f"OK mel L1={results['exp1_mel_l1_vs_real']:.4f} s_norm={meta1['s_norm']:.3f}", flush=True)
    except Exception as e:
        results["exp1_error"] = repr(e)
        print(f"FAIL Exp1: {e}", flush=True)
        y1n = None

    # ----- Exp2: random style vs official -----
    print("\n=== Exp2: random style vs pretrained style under ST2 ===", flush=True)
    try:
        # official already in exp1; inject random style by monkeypatching path
        # reconstruct with style from noise mel
        noise = np.random.randn(len(wave)).astype(np.float32) * 0.01
        mel_noise = preprocess_wave(noise, device)
        y2, meta2 = st2_reconstruct(model, mel, text_ids, style_mel=mel_noise)
        y2n = y2.numpy()
        sf.write(OUT / "exp2_st2_recon_random_style.wav", y2n, 24000)
        results["exp2"] = meta2
        results["exp2_mel_l1_vs_real"] = wave_mel_l1(y2n, wave, device)
        if y1n is not None:
            results["exp2_mel_l1_vs_exp1"] = wave_mel_l1(y2n, y1n, device)
        print(
            f"random-style mel L1 vs real={results['exp2_mel_l1_vs_real']:.4f} "
            f"s_norm={meta2['s_norm']:.3f}",
            flush=True,
        )
    except Exception as e:
        results["exp2_error"] = repr(e)
        print(f"FAIL Exp2: {e}", flush=True)

    # ----- Exp3: style space bridge -----
    print("\n=== Exp3: ST2 s vs Kokoro ref_s stats + ST2→Kokoro attempt ===", flush=True)
    with torch.no_grad():
        s_a = model.style_encoder(mel.unsqueeze(1)).cpu().squeeze(0)
        s_p = model.predictor_encoder(mel.unsqueeze(1)).cpu().squeeze(0)
    heart = pipe.load_voice("af_heart")[100, 0].cpu()
    results["exp3"] = {
        "st2_sa_norm": float(s_a.norm()),
        "st2_sa_std": float(s_a.std()),
        "st2_sp_norm": float(s_p.norm()),
        "kokoro_sa_norm": float(heart[:128].norm()),
        "kokoro_sa_std": float(heart[:128].std()),
        "l1_st2_sa_vs_heart": float((s_a - heart[:128]).abs().mean()),
        "cos_st2_sa_vs_heart": float(F.cosine_similarity(s_a.unsqueeze(0), heart[:128].unsqueeze(0))),
    }
    print(json.dumps(results["exp3"], indent=2), flush=True)

    # Force ST2 acoustic style into Kokoro pack (expect bad)
    from timbre_encoder.voice_pack import assemble_voice_pack

    pack = assemble_voice_pack(s_a, pipe.load_voice("af_heart"))
    chunks = []
    for r in pipe(text, voice=pack, speed=1):
        if r.audio is not None:
            chunks.append(r.audio.cpu().numpy())
    bad = np.concatenate(chunks)
    sf.write(OUT / "exp3_st2_style_into_kokoro.wav", bad, 24000)
    results["exp3_mel_l1_st2style_kokoro_vs_real"] = wave_mel_l1(bad, wave, device)
    results["exp3_mel_l1_st2style_kokoro_vs_afheart"] = wave_mel_l1(bad, k_wav, device)
    print(
        f"ST2→Kokoro mel L1 vs real={results['exp3_mel_l1_st2style_kokoro_vs_real']:.4f} "
        f"vs af_heart={results['exp3_mel_l1_st2style_kokoro_vs_afheart']:.4f}",
        flush=True,
    )

    # HTML report
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Clean path exps</title></head><body>",
        "<h1>Clean-path experiments</h1>",
        f"<p>{text}</p><pre>{json.dumps(results, indent=2)}</pre>",
        "<h3>00 original LibriTTS</h3><audio controls src='00_original.wav'></audio>",
        "<h3>Exp0 Kokoro af_heart</h3><audio controls src='exp0_kokoro_af_heart.wav'></audio>",
        "<h3>Exp1 ST2 recon (official style)</h3><audio controls src='exp1_st2_recon_official_style.wav'></audio>",
        "<h3>Exp2 ST2 recon (random style mel)</h3><audio controls src='exp2_st2_recon_random_style.wav'></audio>",
        "<h3>Exp3 ST2 style forced into Kokoro</h3><audio controls src='exp3_st2_style_into_kokoro.wav'></audio>",
        "</body></html>",
    ]
    (OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
