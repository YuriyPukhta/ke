# TODO: Покращення просодії при клонуванні голосу

Нотатка для репозиторію. Описує проблему, поточний стан експериментів і конкретний спосіб виправлення.

Теоретичний контекст (AdaIN, просодія, короткий mel vs довгий текст): [`architecture_and_prosody_insights.md`](architecture_and_prosody_insights.md).

---

## Проблема

Скрипти оптимізації (`cloned_voice_*`, `demo/optimize_ref_s.py`) зараз:

1. Оптимізують **весь** `ref_s` `[1, 256]` — і тембр `[:128]`, і просодію `[128:]`.
2. Використовують **один** вектор для всіх довжин тексту (micro + macro в одному `ref_s`).
3. При збереженні роблять `expand(510, 1, 256)` — **ідентична** просодія на всіх слотах.

Наслідки на довгих текстах:

- шепіт, згасання гучності після перших слів;
- нестабільний темп / «поспіх»;
- LSTM у `ProsodyPredictor` поводиться гірше, ніж з офіційними voice pack.

Офіційні пакети Kokoro (`voices/af_heart.pt`) мають **різну просодію** `[128:]` для кожного слота `len(phonemes)-1`, а тембр `[:128]` майже стабільний.

---

## Ціль

- **Тембр** — один оптимізований вектор `[:128]` (ідентичність голосу).
- **Просодія** — не оптимізувати; брати зі стабільного baseline по довжині (`af_heart` або інший жіночий/чоловічий голос під ціль).

---

## Спосіб (рекомендований)

### Крок 1 — Timbre-only оптимізація

Під час тренування заморозити просодійну половину:

```python
baseline_voice = pipeline.load_voice("af_heart").to(device)  # або af_bella / am_michael

ref_s = start_voice[0].clone().detach()
ref_s.requires_grad_(True)

# Після кожного optimizer.step() — відновлювати просодію з baseline:
with torch.no_grad():
    ref_s[:, 128:] = baseline_voice[0, :, 128:]  # для joint training з одним ref_s
```

Або оптимізувати лише `timbre = ref_s[:, :128].clone().requires_grad_(True)` і перед `forward` збирати:

```python
full_ref_s = torch.cat([timbre, baseline_voice[0, :, 128:]], dim=-1)
```

### Крок 2 — Збір voice pack `[510, 1, 256]`

Після оптимізації **не** робити `expand` усього вектора. Зібрати пакет так:

```python
def assemble_voice_pack(timbre_optimized: torch.Tensor, baseline_voice: torch.Tensor) -> torch.Tensor:
    """
    timbre_optimized: [1, 128] або [1, 1, 128]
    baseline_voice:    [510, 1, 256] — офіційний pack (af_heart тощо)
    """
    if timbre_optimized.dim() == 3:
        timbre = timbre_optimized[0, 0, :128]
    elif timbre_optimized.dim() == 2:
        timbre = timbre_optimized[0, :128]
    else:
        timbre = timbre_optimized[:128]

    pack = torch.zeros((510, 1, 256))
    for i in range(510):
        pack[i, :, :128] = timbre
        pack[i, :, 128:] = baseline_voice[i, :, 128:]
    return pack
```

### Крок 3 — `target_dur` лишається як зараз

Для кожного тренувального кліпу:

```python
idx = len(phonemes) - 1
target_ref_s = baseline_voice[idx]
_, target_dur = model.forward_with_tokens_grad(input_ids, target_ref_s)
```

Це вже є в `optimize_scale.py` — **не змінювати**.

### Крок 4 — Інференс

```python
pipeline(text, voice=assembled_pack, speed=1)
```

Pipeline сам вибере `pack[len(phonemes)-1]` — просодія буде від baseline для цієї довжини, тембр — ваш клон.

---

## Файли для змін

| Файл | Що зробити |
|------|------------|
| `cloned_voice_scale/optimize_scale.py` | Timbre-only + `assemble_voice_pack` |
| `cloned_voice_curriculum/optimize_curriculum.py` | Те саме |
| `cloned_voice_scale_miku/optimize_scale_miku.py` | Те саме; baseline `af_heart` |
| `cloned_voice_demo/optimize_ref_s.py` | Те саме |
| `demo/optimize_ref_s.py` | Те саме |
| `demo/test_optimized_voice.py` | Завантажувати `[510,1,256]` або збирати pack, не `expand` |
| `timbre_encoder/voice_pack.py` | Спільна функція `assemble_voice_pack` |

---

## Чеклист

- [ ] Додати `assemble_voice_pack()` (спільний модуль або utils у `demo/`)
- [ ] Оновити Phase 1/2: оптимізувати лише `ref_s[:, :128]`
- [ ] Після `optimizer.step()` не давати `[128:]` змінюватися (clamp до baseline)
- [ ] Зберігати `optimized_ref_s.pt` як `[1, 128]` **або** повний `[510, 1, 256]`
- [ ] Замінити `expand(510, 1, 256)` на `assemble_voice_pack` у Phase 3 / test scripts
- [ ] A/B тест: старий `expand` vs новий pack на реченні 100–400 фонем
- [ ] Задокументувати вибір baseline (`af_heart` для female target, `am_michael` для male)
- [ ] (Опційно) WavLM loss замість mel L1 для тембру — див. [kokoro.embed](https://github.com/kdrkdrkdr/kokoro.embed)

---

## Критерії успіху

1. Довгий тестовий текст (200+ фонем) без шепоту / згасання в кінці.
2. Тембр близький до target (RVC / FreeVC reference) на коротких словах.
3. Темп і паузи природніші, ніж з `expand(510, 1, 256)`.

---

## Контекст (чому це працює)

| Частина `ref_s` | Хто використовує | Що контролює |
|-----------------|------------------|--------------|
| `[:128]` | `Decoder` | Тембр / ідентичність |
| `[128:]` | `ProsodyPredictor` | Тривалість, F0, енергія |

510 слотів у офіційних pack — головне для **`[128:]`** під довжину фонем. Тембр між слотами майже однаковий; при клонуванні достатньо **одного** тембру + **510 просодій** з baseline.

---

## Посилання в репо

- `documentation/voice_cloning_guide.md` — §3 «Timbre-only Optimization»
- `kokoro/pipeline.py` — `pack[len(ps)-1]`
- `demo/kokoro_revers/model.py` — `forward_with_tokens_grad`

---

*Створено: 2026-03-13. Статус: заплановано, не імплементовано в скриптах.*
