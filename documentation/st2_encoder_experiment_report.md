# Рапорт: Clean-path StyleTTS2 style encoders (шлях A)

**Дата:** 2026-07-26  
**Середовище:** Windows, GTX 1650 Ti 4 GB, `.venv` (`torch 2.13.0+cu126`)  
**Дані:** LibriTTS_R, EN + SNR≥22 (`train-clean-100.filtered.csv` ≈19867, `dev-clean.filtered.csv` ≈3627)

---

## 1. Мета

Перевірити, чи можна навчити акустичний style encoder `E_a` (і допоміжно `E_p`) **у manifold StyleTTS2** на реальному LibriTTS так, щоб реконструкція через **frozen StyleTTS2 decoder** давала впізнаваний і прийнятно чистий голос.

Далі (поза цим прогоном): map/distill `s_a^{ST2} → Kokoro[:128]` + `assemble_voice_pack` з baseline-просодією по length-слотах `[510]`.

---

## 2. Контекст і гіпотеза

### Відхилений підхід
Mel-match LibriTTS через **frozen Kokoro** → `ref_s` OOD → шум (перевірено раніше; listen `dual_snr22`).

### Гіпотеза шляху A
Kokoro не вміє «довільного» LibriTTS-спікера. StyleTTS2-LibriTTS — так. Тому:

1. тренувати `E_a` під **ST2 decoder + GT alignment + GT F0/N**;
2. тримати `s_a` біля official ST2 `style_encoder` (якір);
3. у Kokoro переносити лише після окремого mapper’а, не напряму.

---

## 3. Підготовчі експерименти (sanity)

Скрипт: `timbre_encoder/experiments/run_clean_path_exps.py`  
Артефакти: `timbre_encoder/runs/clean_path_exps/`

| Exp | Опис | Mel L1 vs real | Висновок |
|-----|------|----------------|----------|
| **0** | Kokoro `af_heart` vs LibriTTS | **1.31** | Kokoro ≠ цей спікер |
| **1** | ST2 recon + official style + `a_gt` | **0.43** | ST2 stack живий, слух ок |
| **2** | ST2 recon + random style mel | **0.89** | без правильного `s` — шипіння/скрегіт |
| **3** | ST2 `s_a` → Kokoro pack | **1.21** | прямий міст не працює (cos vs `af_heart` ≈ 0.05) |

**Вердикт підготови:** decoder+aligner ST2 валідні; style encoder критичний; міст у Kokoro потребує окремого етапу.

---

## 4. Метод навчання

### Freeze
StyleTTS2-LibriTTS (`epochs_2nd_00020.pth`): decoder, text_aligner, text_encoder, pitch_extractor, PLBERT, official style/predictor encoders (як teacher).

### Train
`AbsoluteDualStyleEncoder` = deepcopy ST2 `style_encoder` + `predictor_encoder` (warm-start; сумісний `spectral_norm`).

Скрипт: `timbre_encoder/train_encoders_styletts2.py`  
Рантайм: `timbre_encoder/st2_runtime.py`

### Forward (teacher-forced)
```
mel, text → frozen aligner (MAS) → asr
         → frozen pitch/N з mel
mel_crop → E_a → s_a → frozen decoder → wave
         → E_p → s_p (лише teacher L1; у recon decoder не входить)
```

### Losses

| Символ | Зміст |
|--------|--------|
| `mel` | L1(mel(pred_wav), mel_gt) |
| `sa` | L1(`s_a`, frozen ST2 `style_encoder`) |
| `cos` | `1 − cos(s_a, teacher)` (з phase2) |
| `sp` | L1(`s_p`, frozen `predictor_encoder`) |
| `slm` | WavLM feature L1 (спроба phase3 — **провалена на 4 GB**) |

Kokoro length-pack `[510,1,256]` на цьому етапі **не використовується**: один `s_a` у ST2-просторі. Для Kokoro пізніше: один timbre на всі слоти, просодія — з baseline по індексу довжини.

---

## 5. Прогони

### Phase 1 — `st2_encoders` (500 кроків)
- LR `1e-4`, `λ_mel=1`, `λ_sa=0.1`, `λ_sp=1`, **без SLM**
- Warm-start з ST2

| | mel (avg) | sa (avg) | s_a_norm |
|--|-----------|----------|----------|
| first 50 | 0.58 | 0.040 | 2.17 |
| last 50 | 0.56 | **0.088** | 2.19 |

`sa` зріс (0.001 → 0.09): mel штовхнув `E_a` **від** teacher → AdaIN «брудний».  
Listen: голос **впізнаваний**, але сильно грязний (`st2_encoders/listen_test/`).

### Phase 2 — `st2_encoders_v2` (800 кроків, resume @500 → 1300)
- LR `3e-5`, `λ_mel=0.5`, `λ_sa=3`, `λ_sa_cos=1`, `λ_sp=1`
- Мета: повернути `s_a` на manifold teacher

| | mel | sa | cos |
|--|-----|-----|-----|
| step 501 | 0.54 | 0.095 | 0.149 |
| last 50 avg | ~0.61 | **0.006** | ~0.0005 |
| step 1300 | 0.54 | **0.0056** | **0.0004** |

Listen: **дуже схожі / чистіші голоси** (`st2_encoders_v2/listen_test/`). Якір важливіший за голий mel L1.

### Phase 3 — WavLM (невдача)
- `wavlm-base-plus` ≈94.4 M; сам по собі ~0.8 GB peak на 2 с
- Разом зі ST2 decoder на 4 GB → **система впала**
- Висновок: на 1650 Ti SLM відкласти (інша GPU / CPU WavLM / коротші кліпи + жорсткий offload — окремо)

---

## 6. Висновки

1. **Гіпотеза шляху A підтверджена слухом на v2:** `E_a` під frozen ST2 дає впізнаваний клон LibriTTS без RVC і без gradient-opt teachers у Kokoro.
2. **Mel L1 без сильного teacher-якоря шкідливий** — оптимізатор знаходить брудні `s_a` з кращим спектральним match.
3. **Прямий ST2→Kokoro style не працює** (Exp3); потрібен mapper/distill.
4. **WavLM бажаний для якості**, але на 4 GB зараз непрактичний у зв’язці з повним ST2.
5. Просодія Kokoro length-slots — не задача цього етапу: один timbre + baseline `[128:]` по слоту.

---

## 7. Артефакти

| Шлях | Що |
|------|-----|
| `timbre_encoder/runs/clean_path_exps/` | Exp0–3 + `results.json` |
| `timbre_encoder/runs/st2_encoders/` | Phase1 ckpt + брудний listen |
| `timbre_encoder/runs/st2_encoders_v2/` | Phase2 ckpt (**робочий**) + listen |
| `timbre_encoder/train_encoders_styletts2.py` | Train loop |
| `timbre_encoder/make_listen_st2_encoders.py` | Listen-test |
| `documentation/timbre_encoder_plan.md` | План A/B |

---

## 8. Наступні кроки (рекомендації)

1. [x] **Mapper** `s_a^{ST2} → Kokoro[:128]` — `train_st2_to_kokoro_mapper.py`, ckpt `runs/st2_kokoro_mapper/mapper_final.pt` (2000 steps, L1~0.01 на train voices).
2. Listen: `runs/st2_kokoro_mapper/listen_test/` — original / ST2+E_a / Kokoro mapped / baseline-only.
3. Val: held-out спікер, короткі/довгі тексти (перевірка length-slots просодії); якщо mapped ≈ baseline — потрібні сильніші teachers (path B).
4. SLM — лише на GPU ≥8–12 GB або з offload.

---

## 9. One-liner

> Під frozen StyleTTS2 + сильний якір до official `style_encoder` енкодер тембру на LibriTTS працює; без якоря — бруд; у Kokoro — тільки через окремий map, не напряму.
