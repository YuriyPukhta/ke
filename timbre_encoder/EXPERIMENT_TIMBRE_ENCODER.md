# Timbre Encoder — MEL = MEL reconstruction

Train `TimbreEncoder` so encoding a reference and synthesizing through frozen
Kokoro recovers **that same utterance’s mel**.

## Pipeline

```text
LibriTTS utt (align_status=ok, speakers A→Z)
  → mel_ref
  → multi-crop (4 × 192) → TimbreEncoder → mean → absolute timbre[128]
  → ref_s = [timbre | baseline_prosody]
  → frozen Kokoro + forced d_gt
  → F0/N baseline (no_grad overrides)
  → L1(mel_synth, mel_ref) only   # NO λ_res
  → Adam, grad-accum batch
```

## Overnight (all ok speakers)

All-GPU, `max-phonemes=50` (peak ~2.5GB on 4GB card). **`λ_res=0`**.

```powershell
.\.venv\Scripts\python.exe -m timbre_encoder.train_encoder_recon `
  --epochs 8 --batch-size 8 --n-crops 4 --lr 1e-3 `
  --max-clips 0 --max-phonemes 50 --max-wav-sec 4.0 --baseline mean `
  --lambda-res 0 `
  --out-dir timbre_encoder/runs/encoder_recon_all_ok
```

(`--speaker` omitted = all `align_status=ok`, sorted A→Z.)

**4GB:** peak VRAM tracks phoneme length, not batch-size.
`ph≤100` → ~3.8GB + Shared thrash. `ph≤50` → ~2.5GB.

Log: `runs/encoder_recon_all_ok/train.log`  
Ckpts: `encoder_epochN.pt`, `encoder_step*.pt`, `encoder_final.pt`  
Listen after finish: `listen_test/index.html`

## Defaults

| Knob | Value |
|------|--------|
| batch-size | 8 (grad accum; peak ≈ 1 utt) |
| n-crops | 4 |
| max-phonemes | **50** |
| max-wav-sec | **4.0** |
| epochs | **8** |
| baseline | voice-bank mean |
| λ_mel / λ_res | 1.0 / **0** |
| lr | 1e-3 |

VRAM peak ~2.5GB with `max-phonemes=50` on 4GB GPU.
