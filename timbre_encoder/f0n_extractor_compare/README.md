# F0 / N extractor compare (Kokoro bypass)

Експерименти: чи можна **обійти** внутрішнє `F0Ntrain` у Kokoro, взявши F0/N з того ж згенерованого аудіо (як GT duration через `d_gt`).

## Ідея

```text
текст → Kokoro native voice → wav + внутрішні F0_pred, N_pred
wav  → зовнішній extractor → F0, N
     → (опційно) калібрування під одиниці Kokoro
     → forward_with_alignment(..., f0_override, n_override)
```

Порівнюємо контури (corr, графіки) і слухаємо resynth.

## Структура

```text
f0n_extractor_compare/
  compare_yin_f0n_kokoro.py   # librosa YIN + STFT log-energy
  compare_st2_f0n_kokoro.py   # StyleTTS2 JDCNet + log_norm(mel)
  results/
    yin_f0n_kokoro_compare/   # index.html + wav + curves
    st2_f0n_kokoro_compare/
  README.md
```

## Запуск

З кореня репо:

```bash
.\.venv\Scripts\python.exe -m timbre_encoder.f0n_extractor_compare.compare_yin_f0n_kokoro
.\.venv\Scripts\python.exe -m timbre_encoder.f0n_extractor_compare.compare_st2_f0n_kokoro
```

Опції: `--voices af_heart,am_michael`, `--texts "..." "..."`, `--out <dir>`.

Потрібно: CUDA (бажано), Kokoro weights (HF), для ST2-скрипа — `timbre_encoder/checkpoints/styletts2_libritts` і `third_party/StyleTTS2/Utils/JDC/bst.t7` (у корені репо).

## Хто за що відповідає

| Джерело | F0 | N |
|---------|----|---|
| Kokoro (еталон) | `ProsodyPredictor.F0Ntrain` | те саме |
| YIN-скрипт | `librosa.yin` (Hz) | STFT mean → log-energy |
| ST2-скрипт | JDCNet `pitch_extractor(mel)` | `log_norm(mel)` (як у StyleTTS2) |

Калібрування: mean/std під Kokoro; для F0 — voiced-aware + UV→0.

## Як слухати `index.html`

У кожній картці:

1. **native** — еталон Kokoro  
2. **N cal only** — чи N з extractor ≈ native  
3. **F0 cal only** — якість висоти  
4. **F0+N cal** — повний bypass  
5. **raw** — без калібрування (контроль одиниць)

## Результати (коротко, 2026-07-28)

На тих самих текстах / `af_heart` + `am_michael`:

| Extractor | F0 corr (Kokoro-voiced) | N corr |
|-----------|-------------------------|--------|
| YIN + log-E | ~0.02–0.71 (слабко на male UV) | ~0.96–0.97 |
| **ST2 JDC + log_norm** | **~0.65–0.89** | **~0.99** |

**Висновок:** N можна брати з energy / `log_norm`. Для F0 краще **ST2 pitch_extractor**, не YIN. Bypass осмислений для **того ж тексту**, що в reference wav; для нового тексту лишається text-conditioned `F0Ntrain`.

## Пов’язаний код в репо

- `demo/kokoro_revers/model.py` — `forward_with_alignment(..., f0_override, n_override)`
- `timbre_encoder/st2_runtime.py` — той самий F0/N шлях у StyleTTS2
- `timbre_encoder/optimize_speaker_ref_s.py` — `extract_f0_n` (YIN)
