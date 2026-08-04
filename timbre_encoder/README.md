# Dual / Timbre Encoder — шляхи навчання

## Що вже працює

Supervised у **просторі Kokoro** (офіційні голоси → mel → teacher timbre):

```bash
python -m timbre_encoder.train_supervised_kokoro --steps 500
python -m timbre_encoder.make_listen_supervised
# listen: timbre_encoder/runs/supervised_kokoro/listen_test/
```

Residual: `s = baseline + scale * tanh(delta)` (тримати AdaIN in-distribution).

---

## A. Чисте навчання (основний напрям)

Мета: клон з reference wav **без** RVC і **без** gradient-opt teachers.

### Чому не «LibriTTS mel → frozen Kokoro»

Kokoro не покриває довільних спікерів LibriTTS. Mel L1 навіть для `af_heart` ≈ **1.2** vs real → оптимізатор штовхає `ref_s` OOD → шум. Це вже перевірено.

### Правильна схема (StyleTTS2-native, потім міст у Kokoro)

```text
LibriTTS_R (SNR≥22, EN)
    → frozen TMA (a_gt)
    → train E_a / E_p  через frozen StyleTTS2 decoder (+ WavLM)
       ^^^ той самий manifold, що й style encoders
    → (опційно) distill / linear map  StyleTTS2 s_a  →  Kokoro [:128]
    → assemble_voice_pack + baseline prosody → Kokoro infer
```

| Крок | Що freeze | Що train | Loss |
|------|-----------|----------|------|
| 1 | StyleTTS2 decoder, aligner, text encoders, WavLM | `E_a`, `E_p` | mel @ `a_gt`, F0/N, `L_slm` |
| 2 | Kokoro повністю | маленький mapper або fine-tune `E` head | L1/cos до Kokoro teachers **лише з офіційних / mapped** голосів, або synth-in-Kokoro pairs |

Критично: на кроці 1 decoder = **StyleTTS2**, не Kokoro. Тоді mel-match на LibriTTS — валідна ціль.

```bash
# GPU: use .venv (torch cu126). 4GB: --no-slm
.\.venv\Scripts\python.exe -m timbre_encoder.train_encoders_styletts2 \
  --manifest timbre_encoder/manifests/train-clean-100.filtered.csv \
  --out-dir timbre_encoder/runs/st2_encoders \
  --steps 2000 --no-slm --save-every 200
```

Smoke / val: `dev-clean.filtered.csv`. Warm-start з ST2 `style_encoder` / `predictor_encoder`.  
Ckpt: `timbre_encoder/checkpoints/styletts2_libritts/Models/LibriTTS/epochs_2nd_00020.pth`.

Дані: `dataset/LibriTTS_R` + `filter_manifest --min-snr 22`.

### Інференс після чистого шляху

```text
ref.wav → E_a → (map) → Kokoro timbre[:128]
        → assemble_voice_pack(timbre, af_heart|am_michael)
        → KPipeline
```

Просодію з reference **не** беремо (або окремий етап пізніше).

---

## B. Запасний варіант (teachers / RVC) — не основний

Якщо чистий StyleTTS2-шлях відкладається: teachers у просторі Kokoro вручну.

```text
target wav
  → (опційно) RVC/FreeVC з Kokoro source, 0 pitch shift, same duration
  → timbre-only gradient opt (freeze ref_s[:,128:])
  → teacher [128] + ref wav
  → supervised fine-tune encoder (warm-start з supervised_kokoro)
  → optional distill через frozen Kokoro на matched duration
```

Деталі: [`documentation/timbre_encoder_plan.md`](../documentation/timbre_encoder_plan.md), [`TODO_prosody.md`](../documentation/TODO_prosody.md).

Мінуси: залежність від VC/opt, шум teachers, більше ручної роботи.  
Плюси: швидший шлях до конкретного клону без підняття StyleTTS2 decoder.

---

## Дані / фільтр

`*.book.tsv` останній стовпець = **SNR**. Clean уже ≥20; для якості ≥22:

```bash
python -m timbre_encoder.build_librispeech_manifest
python -m timbre_encoder.filter_manifest --all-defaults --min-snr 22
```

### Кеш GT alignment у filtered CSV

| Колонка | Зміст |
|---------|--------|
| `phonemes` | сирий Misaki/Kokoro |
| `phonemes_st2` | **нова**: після норм під ST2 aligner |
| `norm_applied` | `1` якщо `phonemes ≠ phonemes_st2` |
| `norm_ops` | `repl=2\|drop=1\|dlen=-1` |
| `len_phonemes` / `len_phonemes_st2` / `delta_len` | довжини та різниця |
| `d_gt` | кадри на **`phonemes_st2`** |

Після норм aligner завжди на `phonemes_st2` (навіть якщо `delta_len=0`).

```bash
.\.venv\Scripts\python.exe -m timbre_encoder.cache_alignment \
  --csv timbre_encoder/manifests/dev-clean.filtered.csv --renorm-ok --drop-legacy-cols

.\.venv\Scripts\python.exe -m timbre_encoder.cache_alignment \
  --csv timbre_encoder/manifests/train-clean-100.filtered.csv --renorm-ok --drop-legacy-cols
```


---

## Kokoro 2-step embed (LibriTTS × frozen Kokoro)

Style residual on `af_heart` + forced alignment. See [`kokoro_embed_train/README.md`](kokoro_embed_train/README.md).

```bash
.\.venv\Scripts\python.exe -m timbre_encoder.kokoro_embed_train.train_stage1 \
  --manifest timbre_encoder/manifests/dev-clean.filtered.csv --steps 50
```

---

## Застаріле (не використовувати)

```bash
# Старий dual loop — замінений kokoro_embed_train
python -m timbre_encoder.train_dual_encoders --manifest .../train-clean-100.filtered.csv
```

---

## Infer

```bash
python -m timbre_encoder.infer_dual \
  --wav ref.wav --text "Hello" \
  --ckpt timbre_encoder/runs/supervised_kokoro/dual_encoder_final.pt \
  --mode timbre_only --baseline af_heart
```
