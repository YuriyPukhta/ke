# Рецепт: відновити под з network volume

Диск контейнера (container disk) **неперсистентний**. Усе важливе має жити на volume, змонтованому як `/workspace`. Цей файл — чекліст після stop/terminate + новий под з тим самим volume.

## Що на volume (зберігається)

| Шлях | Зміст |
|------|--------|
| `/workspace/kokoro/` | код (`runpod_training/`, `third_party/…`) |
| `/workspace/kokoro_assets/` | WAV LibriTTS_R, StyleTTS2 utils, HF/assets (якщо так залити) |
| `/workspace/kokoro-run/` | **run artifacts** — checkpoints, metrics, TB, state |
| `/workspace/runpod_training_ship.tar` | опційний архів (можна видалити) |

Layout run:

```text
/workspace/kokoro-run/
  checkpoints/{stage1,stage2}/   # latest, best_*, epoch_*, first_stage.pth, final.pth
  logs/tensorboard/{stage1,stage2}/
  logs/console/                  # stdout тренування + tensorboard process log
  logs/wandb/                    # локальний W&B cache (якщо WANDB_API_KEY є)
  metrics/{stage1,stage2}_*.jsonl, summary.json
  samples/
  exports/{kmodel,voicepacks}/
  indexes/                       # копії/аудит (payload indexes — окремо)
  weights/kokoro_base.pth
  configs/runtime.yml
  state/latest*.json, preflight.json
  cache/                         # HF / torch hub (щоб не губити на диску пода)
```

## Що НЕ на volume (втрачається з подом)

- Пакети, встановлені лише в container FS (`pip install` без фіксації в image).
- Усе під `/tmp`, `/root` (крім того, що ви явно спрямували в `/workspace`).
- Live GPU RAM / незаписаний крок тренування (останній завершений checkpoint на volume — точка відновлення).

Метрики/чеки/логи тренування після фіксу в `launch_runpod.sh` пишуться на volume. W&B у хмарі — додаткова копія, не єдина.

---

## 1. Новий под

1. Create Pod → **attach той самий Network Volume** (mount `/workspace`).
2. GPU: A40 48GB (або сумісна).
3. Image: той самий PyTorch template (або з `openssh-server` + TCP 22, якщо треба SCP).
4. Secrets: `WANDB_API_KEY` (опційно).
5. Start → Connect (web terminal або SSH).

SSH proxy:

```bash
ssh USER@ssh.runpod.io -i ~/.ssh/id_ed25519_runpod
```

SSH over exposed TCP (для `scp`/`rsync`) — з Connect tab: `root@IP -p PORT -i ~/.ssh/id_ed25519_runpod`.

---

## 2. Швидка перевірка volume

```bash
df -h /workspace
ls -la /workspace
ls -la /workspace/kokoro/runpod_training
ls -la /workspace/kokoro-run/checkpoints/stage1 /workspace/kokoro-run/checkpoints/stage2
ls -la /workspace/kokoro-run/metrics /workspace/kokoro-run/logs
cat /workspace/kokoro-run/state/latest.json 2>/dev/null || true
cat /workspace/kokoro-run/state/latest_stage1.json 2>/dev/null || true
cat /workspace/kokoro-run/state/latest_stage2.json 2>/dev/null || true
```

Або скрипт:

```bash
bash /workspace/kokoro/runpod_training/scripts/restore_pod.sh
```

Очікувано:

- `runpod_training/payload/kokoro_targets_v1` існує (~24k NPZ).
- Аудіо: або `/workspace/kokoro_assets/data/LibriTTS_R`, або `/workspace/data/LibriTTS_R`.
- StyleTTS2: `/workspace/kokoro/third_party/StyleTTS2` **або** utils уже в `kokoro_assets` + StyleTTS2 на path.

---

## 3. Env на новому поді

```bash
export REPO=/workspace/kokoro
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export RUN_ROOT=/workspace/kokoro-run

# Підставте реальний шлях до WAV на вашому volume:
export DATA_ROOT=/workspace/kokoro_assets/data/LibriTTS_R
# альтернатива: export DATA_ROOT=/workspace/data/LibriTTS_R

# Кеші моделей тільки на volume (не на container disk)
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" \
  "$RUN_ROOT/logs/console"

# Опційно
export WANDB_API_KEY=...   # з RunPod Secret / env
```

Якщо `kokoro_base.pth` уже є:

```bash
export SKIP_CONVERT=1
```

Indexes у payload уже готові:

```bash
export SKIP_INDEX=1
```

---

## 4. Залежності (лише якщо новий image без пакетів)

Після reinstall image контейнер чистий; volume з кодом лишається.

```bash
cd /workspace/kokoro
pip install -r runpod_training/requirements.txt
# + StyleTTS2/StyleTTS2 deps за потреби (torch уже в template)
```

Не кладіть `venv` лише в `/root` — краще `/workspace/kokoro/.venv` якщо потрібен окремий env.

---

## 5. Preflight + resume

```bash
cd /workspace/kokoro
python -m runpod_training.scripts.preflight --config "$RUN_ROOT/configs/runtime.yml" \
  || bash runpod_training/scripts/launch_runpod.sh   # перепише runtime.yml якщо ще немає
```

Resume Stage 1:

```bash
RESUME=auto STAGE=stage1 SKIP_CONVERT=1 SKIP_INDEX=1 \
  bash runpod_training/scripts/launch_runpod.sh
```

Resume Stage 2 (після `first_stage.pth`):

```bash
RESUME=auto STAGE=stage2 SKIP_CONVERT=1 SKIP_INDEX=1 \
  bash runpod_training/scripts/launch_runpod.sh
```

Повний цикл з нуля артефактів (рідко потрібно):

```bash
STAGE=all bash runpod_training/scripts/launch_runpod.sh
```

Export:

```bash
STAGE=export SKIP_CONVERT=1 SKIP_INDEX=1 \
  bash runpod_training/scripts/launch_runpod.sh
```

---

## 6. Моніторинг після відновлення

```bash
tensorboard --logdir "$RUN_ROOT/logs/tensorboard" --host 0.0.0.0 --port 6006
# або дивіться JSONL:
tail -f "$RUN_ROOT/metrics/stage1_train.jsonl"
tail -f "$RUN_ROOT/logs/console/stage1.log"
```

---

## 7. Чекліст «немає неповоротної втрати»

Перед **Stop/Terminate** пода:

- [ ] Останній checkpoint на volume (`checkpoints/.../latest.pth` або `epoch_*.pth`)
- [ ] `state/latest*.json` вказує на існуючий файл
- [ ] `metrics/*_train.jsonl` росте / свіжий mtime
- [ ] TB events під `logs/tensorboard/`
- [ ] Console log під `logs/console/` (після оновленого launcher)
- [ ] Немає важливих файлів лише в `/tmp` або `/root`

Після нового пода:

- [ ] Той самий volume ID
- [ ] `DATA_ROOT` вказує на реальні WAV
- [ ] `preflight` без `errors`
- [ ] `RESUME=auto` підхоплює правильну stage

---

## Типові шляхи на поточному volume (як було залито)

```text
/workspace/kokoro/runpod_training/          # код + payload NPZ
/workspace/kokoro_assets/data/LibriTTS_R/   # WAV
/workspace/kokoro-run/                      # тренувальні артефакти (створить launcher)
```

TCP SCP приклад (IP/port з Connect):

```bash
scp -P PORT -i ~/.ssh/id_ed25519_runpod LOCAL root@IP:/workspace/...
```
