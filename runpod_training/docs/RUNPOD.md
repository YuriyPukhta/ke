# RunPod LibriTTS Kokoro-82M training

## Rules

- Do **not** edit `third_party/` (including kikiri-tts / StyleTTS2).
- Runnable code is only under `runpod_training/`.
- StyleTTS2 is a dependency via `sys.path` (`third_party/kikiri-tts/StyleTTS2`).

## Data (first production run)

- Corpus: **LibriTTS_R** `train-clean-100` + `dev-clean` (not raw LibriSpeech).
- Cache: `kokoro_targets_v1/**/*.npz` + `manifest.jsonl`.
- Phonemes map: `gt_cache/phonemes_libritts_filtered.jsonl`.
- Manifests: `timbre_encoder/manifests/*.filtered.csv`.
- QC keeps `align_status=ok`, complete NPZ, `align_score >= 0.25`, `phoneme_ids` length ≤ 510.
- **No** phoneme remaps (`ʏ→y`, `Y→y`, etc.). IDs come from NPZ.

Expected accepted count ≈ 23 459 after QC.

## Persistent volume

```text
/workspace/kokoro-run/
  checkpoints/{stage1,stage2}/   # latest, best_*, epoch_*, first_stage.pth, final.pth
  logs/tensorboard/{stage1,stage2}/
  logs/console/                  # stage*.log, launch tee, tensorboard.log
  logs/wandb/                    # local W&B dir when API key set
  metrics/{stage1,stage2}_*.jsonl, summary.json
  samples/
  exports/{kmodel,voicepacks}/
  indexes/{train,dev}.jsonl, *_list.txt, OOD_texts.txt, audit_summary.json
  weights/kokoro_base.pth, config.json
  configs/runtime.yml
  state/latest.json, latest_stage1.json, latest_stage2.json, preflight.json
  cache/{huggingface,torch,xdg}/ # HF/torch hubs (not container disk)
```

Checkpoints are written atomically (`.tmp` → rename). Retention keeps latest/best + last 3 epoch files.

**Pod recreate:** see repo-root [`POD_RESTORE.md`](../POD_RESTORE.md) and `bash runpod_training/scripts/restore_pod.sh`.

## Initialization

| Module | Source |
|--------|--------|
| bert, bert_encoder, predictor, text_encoder, decoder | English `kokoro-v1_0` → `kokoro_base.pth` |
| text_aligner | StyleTTS2 English `Utils/ASR/epoch_00080.pth` (no token remap; warm-up adapts) |
| pitch_extractor | JDC `bst.t7` (frozen teacher) |
| style_encoder.shared | `dida-80b/kokoro-deutsch-hui-base` (optional); `unshared` re-init |
| predictor_encoder | deepcopy(style_encoder) at Stage 2 start |

## Schedule (A40 48GB)

- Micro-batch 12, grad accum 2 → effective 24, BF16 preferred.
- LR: 5% linear warm-up + cosine to 10% peak.

| Phase | Epochs | Trainable | Peak LRs |
|-------|--------|-----------|----------|
| Aligner warm-up | 1 (max 2) | text_aligner | heads ~2e-4 |
| Stage 1 acoustic | 9 | aligner, text_encoder, style_encoder, decoder, disc | decoder/style 2e-5, text_enc 1e-5 |
| Stage 2 pre-joint | 0–2 | bert_encoder, predictor, predictor_encoder | bert_enc 5e-6, pred_enc 1e-4 |
| Stage 2 joint | 3–9 | + style/decoder/disc/wd | decoder/style lower |

PLBERT stays **frozen**. Diffusion disabled (`diff_epoch: 999`).

## Monitoring

- TensorBoard: port **6006** (`logs/tensorboard`).
- W&B: set `WANDB_API_KEY` as RunPod secret (never commit). Falls back to JSONL/TB.
- Metrics JSONL under `metrics/`.

## Commands

```bash
export PYTHONPATH=/workspace/kokoro
bash runpod_training/scripts/launch_runpod.sh

# Smoke after indexes + weights exist
python -m runpod_training.scripts.smoke_test --config /workspace/kokoro-run/configs/runtime.yml
```

## Alignment note

ASR token rows start with StyleTTS2 index semantics while data uses Kokoro IDs (same as kikiri). Warm-up + `TMA_epoch: 0` adapts the aligner. Cached CTC `d_gt` is QC-only, not the Stage 1/2 duration target.
