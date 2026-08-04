# LibriTTS GT cache: F0 / N / d_gt

Root package for ground-truth feature caching used by timbre / speaker_ref training.

## Layout

```text
gt_cache/
  cache_f0.py         # JDCNet → features/f0/
  cache_f0_jdc_ft.py  # fine-tuned JDC (Kokoro units) → features/f0_jdc_ft/
  cache_f0_rmvpe.py   # RMVPE  → features/f0_rmvpe/
  cache_n.py          # log_norm → features/n/
  cache_d_gt.py       # TMA alignment → d_gt columns in manifest CSV
  cache_d_gt_tma_ft.py # fine-tuned TMA → d_gt_tma_ft columns (keeps baseline d_gt)
  rmvpe_backend.py
  utils.py
  jdc_finetune/       # distill JDC ← Kokoro F0Ntrain (see jdc_finetune/README.md)
  tma_finetune/       # distill TMA ← Kokoro duration_proj (see tma_finetune/README.md)
  features/           # gitignored .npy store
  README.md
```

| Kind | On disk | Manifest columns |
|------|---------|------------------|
| F0 (JDC) | `features/f0/**/*.npy` | `f0_status`, `f0_len` |
| F0 (JDC ft) | `features/f0_jdc_ft/**/*.npy` | `f0_jdc_ft_status`, `f0_jdc_ft_len` |
| F0 (RMVPE) | `features/f0_rmvpe/**/*.npy` | `f0_rmvpe_status`, `f0_rmvpe_len` |
| N | `features/n/**/*.npy` | `n_status`, `n_len` |
| d_gt | *(in CSV)* | `d_gt`, `align_status`, `phonemes_st2`, … |
| d_gt (TMA ft) | *(in CSV)* | `d_gt_tma_ft`, `align_status_tma_ft`, `mel_down_len_tma_ft` |

Curves: float32 `[T_mel]`, hop 300 @ 24 kHz (`max_wav_sec=8`, trim 30 dB).

Manifests stay in `timbre_encoder/manifests/` (updated in place by these scripts).

RMVPE weights: `timbre_encoder/checkpoints/rmvpe/rmvpe.pt` (~181 MB).

## Run

```powershell
# d_gt (alignment)
.\.venv\Scripts\python.exe -m gt_cache.cache_d_gt `
  --csv timbre_encoder/manifests/train-clean-100.filtered.csv

# N
.\.venv\Scripts\python.exe -m gt_cache.cache_n `
  --csv timbre_encoder/manifests/train-clean-100.filtered.csv

# F0 JDC
.\.venv\Scripts\python.exe -m gt_cache.cache_f0 `
  --csv timbre_encoder/manifests/train-clean-100.filtered.csv

# F0 JDC fine-tuned (Kokoro units) — default weights: jdc_finetune/checkpoints/jdc_kokoro_best.t7
.\.venv\Scripts\python.exe -m gt_cache.cache_f0_jdc_ft `
  --csv timbre_encoder/manifests/dev-clean.filtered.csv

# d_gt fine-tuned TMA (Kokoro duration) — columns d_gt_tma_ft / align_status_tma_ft
.\.venv\Scripts\python.exe -m gt_cache.cache_d_gt_tma_ft `
  --csv timbre_encoder/manifests/train-clean-100.filtered.csv

# F0 RMVPE
.\.venv\Scripts\python.exe -m gt_cache.cache_f0_rmvpe `
  --csv timbre_encoder/manifests/train-clean-100.filtered.csv --speaker 6209
```

`--limit N`, `--force` as needed.

## Python API

```python
from gt_cache import DEFAULT_FEATURES_ROOT, feature_path, load_curve
```

Related: `python -m gt_cache.compare_to_kokoro --speaker 6209 --n 6` → `gt_cache/results/compare_to_kokoro/index.html`.

JDC fine-tune on Kokoro F0: `python -m gt_cache.jdc_finetune.build_dataset` then `…train` — see `jdc_finetune/README.md`.
