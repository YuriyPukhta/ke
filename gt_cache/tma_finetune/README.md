# TMA ← Kokoro duration distillation

Fine-tune StyleTTS2 **ASRCNN / TMA** so that mel from Kokoro-synthesized audio
aligns to **`duration_proj`** (the lengths Kokoro used to generate the wav) —
same idea as `jdc_finetune` for F0.

## Idea

```text
text + voice pack
  → Kokoro forward_with_tokens → wav + pred_dur
  → mel(wav)                                   # input
  → rescale pred_dur[1:-1] → T_mel_down         # hard a_gt target
  → train ASRCNN soft_attn ≈ hard path (NLL)
```

## Layout

```text
gt_cache/tma_finetune/
  build_dataset.py
  train.py
  eval_compare.py
  dataset.py
  common.py
  data/            # gitignored
  checkpoints/     # gitignored
  results/
```

Checkpoint layout matches `asr_epoch_00080.pth` (`{"model": state_dict}`),
so `FrozenTextAligner(checkpoint=...)` can load it directly.

## Run

From repo root:

```powershell
# Small starter set
.\.venv\Scripts\python.exe -m gt_cache.tma_finetune.build_dataset `
  --train-voices af_heart,am_michael,af_bella,bm_george,af_sarah,am_fenrir `
  --val-voices af_nicole,am_adam,bf_emma,bm_lewis `
  --limit-texts 12 --force

.\.venv\Scripts\python.exe -m gt_cache.tma_finetune.train `
  --epochs 10 --lr 1e-4 --max-mel 600

# Held-out voices
.\.venv\Scripts\python.exe -m gt_cache.tma_finetune.eval_compare `
  --weights gt_cache/tma_finetune/checkpoints/tma_kokoro_best.pth `
  --voices af_nicole,am_adam,bf_emma,bm_lewis
```

Larger set (23 train / 5 val voices, clip targets):

```powershell
.\.venv\Scripts\python.exe -m gt_cache.tma_finetune.build_dataset `
  --all-voices --n-val-voices 5 `
  --train-clips 1000 --val-clips 2500 --force `
  --texts-csv timbre_encoder/manifests/train-clean-100.filtered.csv
```

(~44 texts × 23 train voices → trim to 1000; 500 texts × 5 val → 2500)
## Notes

- Batch size is **1** (S2S aligner is sequential / variable-length).
- Target durations are rescaled so `sum(d) == T_mel // 2` (`n_down=1`).
- Loss: NLL of soft attention on the hard MAS path from Kokoro durs.
- Metric: Pearson corr of MAS durations vs Kokoro (rescaled) durs.
- Use fine-tuned weights: `FrozenTextAligner(checkpoint=".../tma_kokoro_best.pth")`.
- Rebuild LibriTTS `d_gt` cache (new columns, baseline untouched):

```powershell
.\.venv\Scripts\python.exe -m gt_cache.cache_d_gt_tma_ft `
  --csv timbre_encoder/manifests/dev-clean.filtered.csv
.\.venv\Scripts\python.exe -m gt_cache.cache_d_gt_tma_ft `
  --csv timbre_encoder/manifests/train-clean-100.filtered.csv
```
