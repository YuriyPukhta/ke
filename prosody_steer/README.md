# prosody_steer

Експерименти з **замороженим тембром** (`[:128]` / E_A) і пошуком лише просодії.

Ціль завжди **схожість на оригінальний голос** (mel / WavLM), не емоції.

## Структура

| Шлях | Що це |
|------|--------|
| [`optimize_prosody_pso.py`](optimize_prosody_pso.py) | PSO у сирому `prosody_delta[128]` |
| [`optimize_pca_pso.py`](optimize_pca_pso.py) | PCA + PSO: `--target full` (весь `[256]`) або `prosody` |
| [`runs/speaker_prosody_pso_6209/`](runs/speaker_prosody_pso_6209/) | Smoke: raw PSO |
| [`runs/speaker_pca_pso_6209/`](runs/speaker_pca_pso_6209/) | Smoke: PCA+PSO prosody-only |
| [`runs/speaker_pca_pso_full_6209/`](runs/speaker_pca_pso_full_6209/) | Smoke: PCA+PSO **full vector** |
| [`runs/kokoro_hack_samples/`](runs/kokoro_hack_samples/) | Семпли [kokoro_hack](https://github.com/eryawww/kokoro_hack) (emotion) — на слух слабкі |

Хелпери: `timbre_encoder/` (manifest, mel, packs, listen HTML).

---

## A) Raw PSO (`optimize_prosody_pso.py`)

Як kokoro_hack, але loss = mel до target WAV.

```text
timbre.pt (freeze) + baseline[128:] + delta[128]  ← PSO у R^128
```

```bash
.\.venv\Scripts\python.exe prosody_steer/optimize_prosody_pso.py ^
  --speaker 6209 ^
  --timbre-ckpt timbre_encoder/runs/speaker_ref_s_6209_bankmean_norm/timbre.pt ^
  --baseline mean --voice-bank us ^
  --bank-cache timbre_encoder/runs/voice_bank_us.pt ^
  --max-clips 8 --n-particles 16 --iters 30 --eval-clips 0
```

---

## B) PCA + PSO (`optimize_pca_pso.py`) — Fabul8-ідея без перебору ознак

**Fabul8**: variance top-dims → PCA → finite-diff / Adam.  
**Ми**: PCA на офіційних голосах → **PSO у R^k** (без enum dims / FD).

### Full vector (рекомендовано зараз)

```text
US voices → frame[256] @ slot 50 → PCA(k≈19)
     z ← PSO → decode → ref_s[256] → expand(510)
     forced align → mel vs LibriTTS
```

```bash
.\.venv\Scripts\python.exe prosody_steer/optimize_pca_pso.py ^
  --target full --speaker 6209 ^
  --pca-components 19 --max-clips 8 --n-particles 20 --iters 30 --eval-clips 0
```

### Prosody-only (freeze E_A)

```bash
.\.venv\Scripts\python.exe prosody_steer/optimize_pca_pso.py ^
  --target prosody --speaker 6209 ^
  --timbre-ckpt timbre_encoder/runs/speaker_ref_s_6209_bankmean_norm/timbre.pt ^
  --prosody-mode delta --pca-components 16 ^
  --max-clips 8 --n-particles 20 --iters 30 --eval-clips 0
```

Опційно: `--lambda-slm 0.5` (WavLM cos+L1).

---

## Порівняння підходів

| | Fabul8 | kokoro_hack | raw PSO (A) | PCA+PSO (B) |
|--|--------|-------------|-------------|-------------|
| Ціль | WavLM similarity | emotion centroid | mel voice match | mel voice match |
| Простір | PCA / top-var dims | R^128 style | R^128 delta | R^k PCA (k≪128) |
| Пошук | FD + Adam | PSO | PSO | PSO |
| Тембр | крутить усе [256] | freeze pack[:128] | freeze E_A | freeze E_A |

## Залежності

`torch`, `librosa`, `scikit-learn` (PCA), kokoro / `demo/kokoro_revers`.  
WavLM — опційно через `--lambda-slm`.
