# JDC ← Kokoro F0 distillation

Fine-tune StyleTTS2 **JDCNet** so that mel from Kokoro-synthesized audio
matches internal **F0Ntrain** (Kokoro units) — no post-hoc scale/shift.

## Idea

```text
text + voice pack
  → Kokoro forward_with_tokens → wav
  → mel(wav)                          # input
  → F0Ntrain (same utterance)         # target [T_mel]
  → train JDCNet(mel) ≈ target
```

## Layout

```text
gt_cache/jdc_finetune/
  build_dataset.py
  train.py
  eval_compare.py
  dataset.py
  common.py
  data/            # gitignored
  checkpoints/     # gitignored
  results/
```

## Run

From repo root:

```powershell
# Full Kokoro-82M set: all voices, 10 held-out for val, many LibriTTS texts
.\.venv\Scripts\python.exe -m gt_cache.jdc_finetune.build_dataset `
  --all-voices --n-val-voices 10 --limit-texts 45 --force `
  --texts-csv timbre_encoder/manifests/dev-clean.filtered.csv

# Large effective batch on 4GB: bs=12 * accum=2 (=24), length-bucketed + max_frames=400
.\.venv\Scripts\python.exe -m gt_cache.jdc_finetune.train `
  --epochs 25 --batch-size 12 --accum-steps 2 --lr 1e-4 --max-frames 400

# Eval on held-out voices
.\.venv\Scripts\python.exe -m gt_cache.jdc_finetune.eval_compare `
  --weights gt_cache/jdc_finetune/checkpoints/jdc_kokoro_best.t7 `
  --voices af_nicole,af_jessica,af_nova,am_adam,am_echo,am_onyx,bf_alice,bf_lily,bm_fable,bm_lewis
```

Checkpoints are saved in the same layout as `bst.t7` (`{"net": state_dict}`),
so they drop into `load_F0_models` / `gt_cache.utils.load_pitch_extractor`
if you point `F0_path` / `--init-weights` at them.

## Notes

- Target F0 is resized to mel length (hop 300 @ 24 kHz).
- Loss: Smooth L1, voiced frames (`F0 > 50`) weighted ×2.
- Metric: Pearson corr on Kokoro-voiced frames (same as `compare_kokoro_voices`).
- GTX 1650 Ti: batch 2–4 is safe; JDC is small (~21 MB).
