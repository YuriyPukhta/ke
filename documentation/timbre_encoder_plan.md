# План: Timbre Encoder для Kokoro

Контекст архітектури та просодії: [`architecture_and_prosody_insights.md`](architecture_and_prosody_insights.md).  
Операційний README: [`../timbre_encoder/README.md`](../timbre_encoder/README.md).  
Рапорт експерименту A (ST2 encoders): [`st2_encoder_experiment_report.md`](st2_encoder_experiment_report.md).

Просодія `[128:]` — **стала**, з офіційного voice pack baseline.  
Енкодер вчиться лише **тембр `[:128]`** з reference audio.

---

## Пріоритет шляхів

| Пріоритет | Шлях | Суть |
|-----------|------|------|
| **A (основний)** | Чисте навчання | StyleTTS2 decoder + Aligner + WavLM на LibriTTS_R → map/distill у Kokoro |
| **B (запасний)** | Teachers / RVC | Timbre-only opt (+ опційно RVC) → supervised на Kokoro teachers |

**Не робити:** mel L1 LibriTTS через **frozen Kokoro** (OOD `ref_s` → шум; перевірено емпірично).

---

## A — Чисте навчання (бажаний)

1. Train `E_a` (і за потреби `E_p`) з **frozen StyleTTS2** decoder/aligner на filtered LibriTTS_R (`SNR≥22`, EN).
2. Losses: mel @ `a_gt`, F0/N, `L_slm` (WavLM not tuned).
3. Міст у Kokoro: distill / mapper `s_a^{ST2} → [:128]^{Kokoro}` або fine-tune head на синтетичних Kokoro-парах.
4. Infer: `assemble_voice_pack(timbre, baseline_prosody)`.

Деталі: `timbre_encoder/README.md` §A.

---

## B — Запасний варіант (teachers / RVC)

Використовувати, якщо A ще не піднятий (нема повного StyleTTS2 decoder stack).

### Етап 0 — Teacher dataset

Згенерувати пари **(reference_wav, timbre_teacher)** без енкодера.

**Джерела:**

1. Існуючі `optimized_ref_s.pt` + target wav з `cloned_voice_*`
2. Нові пари через timbre-only `kokoro_revers` (див. `TODO_prosody.md`)
3. Official packs + synthetic audio — warm-start / sanity (`train_supervised_kokoro.py`)

**Формат `dataset_timbre.csv`:**

```csv
audio_path,baseline_voice,timbre_teacher_path,phonemes,text,speaker_id
chunks/spk01_ref.wav,af_heart,teachers/spk01_timbre.pt,,,spk01
```

`timbre_teacher` = `ref_s[:, :128]` після timbre-only gradient opt на reference.

**Мінімум для v0:** 200–500 reference clips, 20–50 «спікерів» (RVC/FreeVC targets).

### Етап 1 — Supervised під teacher

```python
loss = L1(timbre_pred, timbre_teacher) + 0.1 * (1 - cosine_sim(...))
```

Warm-start з `runs/supervised_kokoro/`.

### Етап 2 — Distillation через frozen Kokoro

Лише коли target уже в Kokoro-manifold (VC / teacher opt), з **forced duration** baseline.

### Етап 3 — (опційно) Multi-length

Тембр інваріантний до `baseline[idx, 128:]`.

---

## Мета (інференс, спільна для A і B)

```
reference.wav (3–10 с)
    → TimbreEncoder
    → timbre [128]
    → assemble_voice_pack(timbre, baseline_voice)
    → [510, 1, 256]
    → Kokoro inference
```

**Не робимо на інференсі:** diffusion, gradient opt `ref_s`.

---

## Архітектура енкодера

Копія `StyleEncoder` з StyleTTS2 (CNN → 128-D).

Для Kokoro: **residual** `s = baseline + scale * tanh(delta)` (zero-init head).

| Параметр | Значення |
|----------|----------|
| `dim_in` | 64 |
| `style_dim` | **128** |
| `max_conv_dim` | 512 |
| Sample rate | 24000 |

**Warm start (опційно):** `style_encoder` з StyleTTS2-LibriTTS / LJSpeech.

---

## Просодія — стала з baseline

| Цільовий голос | Baseline prosody |
|----------------|------------------|
| Female | `af_heart` / `af_bella` |
| Male | `am_michael` / `am_puck` |

Реалізація: `timbre_encoder/voice_pack.py`.

---

## Чеклист

### Чисте (A)
- [x] Підняти StyleTTS2 decoder + pretrained weights (LibriTTS)
- [x] Clean-path Exp0–3 (`runs/clean_path_exps/`) — Exp1 OK
- [x] `train_encoders_styletts2.py` — encoder-only на LibriTTS_R filtered
- [ ] Mapper / distill → Kokoro `[:128]`
- [ ] Val: clone held-out speaker → listen

### Запасний (B)
- [x] `train_supervised_kokoro.py` — sanity в Kokoro space
- [ ] `build_teacher_dataset.py` — timbre-only teachers
- [ ] Fine-tune encoder на teachers
- [ ] Distill + listen на real targets

### Не робити
- [x] ~~mel-match LibriTTS × frozen Kokoro~~ (відхилено)
