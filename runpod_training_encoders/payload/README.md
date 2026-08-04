# RunPod data payload (no WAV)

Ships with `runpod_training/` to the pod. **Audio is separate.**

```text
payload/
  kokoro_targets_v1/     # NPZ f0/n/d_gt/phoneme_ids + manifest.jsonl  (~185 MB)
  phonemes_libritts_filtered.jsonl
  manifests/
    train-clean-100.filtered.csv
    dev-clean.filtered.csv
  indexes/
    train.jsonl  (~19.7k)
    dev.jsonl    (~3.6k)
    *_list.txt, OOD_texts.txt, speaker_map.json, audit_summary.json
```

Paths inside indexes are **relative**:
- `audio` / `audio_path` → under `DATA_ROOT` (LibriTTS_R on the pod)
- `cache` / `npz_path` → under `payload/kokoro_targets_v1`

On RunPod you still need only:

```bash
# audio (~10 GB), not in this folder
/workspace/data/LibriTTS_R/{train-clean-100,dev-clean}/...
```
