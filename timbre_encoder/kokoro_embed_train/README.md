# Kokoro 2-step embedding training

Train only Kokoro style residuals (`DualStyleEncoder`) through **frozen** Kokoro.

| Stage | Train | Loss | Decode? |
|-------|--------|------|---------|
| 1 | `E_a` (timbre) | mel L1 + residual; baseline `[128:]` frozen | yes + forced `d_gt` |
| 2 | `E_p` (prosody) | F0/N (ST2 cache cal→Kokoro) + duration vs `d_gt` | **no** |

**v0 baseline:** `af_heart` only.

Listen always compares: **original** | **style+baseline** | **full vector**.

## Layout

```
kokoro_embed_train/
  common.py
  report_f0n_calibration.py   # multi-voice cal quality (before train)
  train_stage1.py
  train_stage2.py
  make_listen.py
  results/                    # gitignored
  README.md
```

Needs: filtered manifests with `align_status=ok`, F0/N cache under `f0n_cache/features/`.

## Commands

```powershell
# 1) Calibration report (multi-voice mean vs single)
.\.venv\Scripts\python.exe -m timbre_encoder.kokoro_embed_train.report_f0n_calibration `
  --manifest timbre_encoder/manifests/dev-clean.filtered.csv --limit 40

# 2) Stage 1 — single speaker sanity (same voice for both stages)
.\.venv\Scripts\python.exe -m timbre_encoder.kokoro_embed_train.train_stage1 `
  --manifest timbre_encoder/manifests/train-clean-100.filtered.csv `
  --speaker 4195 --no-f0n-override --steps 300 `
  --out-dir timbre_encoder/runs/kokoro_embed/spk4195/stage1 `
  --baseline af_heart

# 3) Stage 2 — same speaker
.\.venv\Scripts\python.exe -m timbre_encoder.kokoro_embed_train.train_stage2 `
  --stage1-ckpt timbre_encoder/runs/kokoro_embed/spk4195/stage1/stage1_final.pt `
  --manifest timbre_encoder/manifests/train-clean-100.filtered.csv `
  --speaker 4195 --steps 300 `
  --out-dir timbre_encoder/runs/kokoro_embed/spk4195/stage2

# 4) Listen (same speaker)
.\.venv\Scripts\python.exe -m timbre_encoder.kokoro_embed_train.make_listen `
  --ckpt timbre_encoder/runs/kokoro_embed/spk4195/stage2/stage2_final.pt `
  --manifest timbre_encoder/manifests/train-clean-100.filtered.csv `
  --speaker 4195 --limit 8 --no-f0n-override
```

`max_wav_sec` defaults to **8** (matches F0/N cache). Stage1 uses F0/N override (calibrated to baseline) unless `--no-f0n-override`. Use `--speaker` for single-voice sanity (recommended before multi-speaker).
