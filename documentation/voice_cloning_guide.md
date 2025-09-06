# Kokoro & RVC v2 Local Voice Cloning & Optimization Guide

This guide details the complete knowledge, setup, and key success factors for cloning voices in Kokoro using a local Retrieval-based Voice Conversion (RVC) v2 CPU setup.

---

## 1. Local RVC v2 CPU Installation & Configuration

To set up RVC v2 on a Windows CPU-only environment:

### A. Python Environment Setup
1. Create a clean virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install CPU-only PyTorch, Torchvision, and Torchaudio from PyTorch's official repository:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

### B. Resolving Windows Compatibility & Fairseq CUDA Errors
1. **Faiss & PyWorld**: Install precompiled wheels or binaries:
   ```bash
   pip install faiss-cpu pyworld
   ```
2. **Fairseq Installation**: Fairseq frequently fails to build on Windows due to C++ compilation errors. Use a precompiled wheel built for Python 3.13:
   * Download and install the compatible `.whl` from trusted community repositories.
3. **Patching Fairseq CUDA Assertions**:
   By default, fairseq asserts CUDA availability when modules are loaded, which crashes CPU execution. Patch the following files inside `.venv\Lib\site-packages\fairseq`:
   * **`optim/adam.py`**:
     * Replace calls targeting CUDA-only initialization with standard PyTorch float/double tensor allocations.
   * **`modules/layer_norm.py`**:
     * Replace custom CUDA Fused Layer Norm fallback imports with standard native `torch.nn.LayerNorm`.

### C. FFMPEG Dependency Fallback
RVC natively uses PyDub or raw audio loading which relies on a system-installed `ffmpeg.exe` binary.
* **The Fix**: Patch RVC's `infer/lib/audio.py` file to add a `librosa` resampling fallback:
  ```python
  import librosa
  # Fallback decode if ffmpeg is missing
  try:
      # standard RVC load
  except Exception:
      y, sr_native = librosa.load(file_path, sr=sr)
  ```

### D. Model Weights Placement
Place the baseline model weights in their corresponding directories inside the RVC folder:
* **Hubert Content Encoder**: `assets/hubert/hubert_base.pt`
* **Pitch Estimator**: `assets/rmvpe/rmvpe.pt`
* **Target Voice Checkpoint**: `assets/weights/<model_name>.pth` (and corresponding `.index` files inside `logs/<model_name>/`)

---

## 2. 24kHz Target Dataset Generation Pipeline

To create target audios for Kokoro optimization, we run a pipeline that first generates TTS audio from Kokoro and then converts it in-memory via RVC.

### A. Voice Mismatch & Pitch Shift (Critical Success Factor)
* **Robotic Artifacts**: Using a male source voice (`am_michael`) and pitch-shifting it (`+12 semitones`) to match a female target voice inside RVC introduces severe metallic phase hums, metallic distortion, and format shifts.
* **The Fix**: Synthesize the source text directly in a female voice (e.g. `af_heart`) and run RVC with **0 pitch shift**. This generates extremely clean, natural target files.

### B. Filename Collisions
* On Windows (which is case-insensitive), text-based audio names like `micro_chunk_if.wav` and `micro_chunk_IF.wav` collide.
* **The Fix**: Prepend a unique row index to all filenames during dataset generation:
  ```python
  filename = f"micro_chunk_{idx}_{safe_text}.wav"
  ```

### C. G2P Phonemizer Spelling Overrides
* Certain spelling mismatches in British vs. US English (e.g. `"fulfil"` vs. `"fulfill"`) cause `misaki` English phonemizers to crash with `unsupported operand type(s) for +: 'NoneType' and 'str'`.
* **The Fix**: Use regex word-boundary substitution before passing text to the pipeline:
  ```python
  text = re.sub(r"\bfulfil\b", "fulfill", raw_text)
  ```
  *(Do not use simple `.replace("fulfil", "fulfill")` as it will corrupt words like `"fulfilled"` into `"fulfillled"` which contains three L's and crashes the model).*

---

## 3. Style Vector Structure & Timbre-only Optimization

### A. Style Vector Anatomy
A style vector `ref_s` in Kokoro has a shape of `(1, 256)` and represents two components:
1. **Dimensions 0 to 127 (Timbre / Vocal Identity)**: Sent to the Decoder to shape the speaker's tone.
2. **Dimensions 128 to 255 (Prosody / Rhythm / Pitch)**: Sent to the Prosody Predictor to predict durations, $F_0$, and energy.

### B. The Whispering / Volume Decay Issue
Optimizing the entire style vector under fixed training alignments forces the optimizer to distort the prosody vector to match the target's pitch. At test time, this distorted prosody vector causes the internal LSTM text encoder to decay over long steps, resulting in whispering and complete volume loss after the first words.

* **The Fix (Timbre-Only Optimization)**:
  1. Freeze the prosody dimensions (`ref_s[:, 128:]`) to the baseline voice (`af_heart`) during training.
  2. Optimize **only the timbre dimensions (`ref_s[:, :128]`)** on clean micro-words for 15-50 epochs.
  3. Reassemble the final voice file of shape `(510, 1, 256)` dynamically:
     ```python
     opt_tensor = torch.zeros((510, 1, 256))
     for i in range(510):
         opt_tensor[i, :, :128] = timbre_optimized   # Trained Miku Timbre
         opt_tensor[i, :, 128:] = af_heart_base[i, :, 128:]  # Length-Appropriate Prosody
     ```
     This keeps Miku's voice character while using `af_heart`'s stable, length-appropriate volume and pitch contours.
