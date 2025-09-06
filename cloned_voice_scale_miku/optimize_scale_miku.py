import os
import sys
import torch
import torchaudio
import soundfile as sf
import numpy as np
import librosa
import csv
import torch.nn.functional as F

# Add Kokoro paths to path
KOKORO_DIR = r"d:\projec\kokoro"
sys.path.append(KOKORO_DIR)
from kokoro.pipeline import KPipeline

# Add kokoro_revers to path for KModel
sys.path.insert(0, os.path.join(KOKORO_DIR, "demo"))
from kokoro_revers.model import KModel

def get_mel_spectrogram(waveform, sample_rate=24000):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    ).to(waveform.device)
    mel = mel_transform(waveform)
    return torch.log(torch.clamp(mel, min=1e-5))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    chunks_dir = os.path.join(cur_dir, "chunks")
    result_dir = os.path.join(cur_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. Initialize Pipeline & Sandboxed KModel
    pipeline = KPipeline(lang_code='a')
    model = KModel(repo_id='hexgrad/Kokoro-82M').to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
        
    # --- PHASE 1: MICRO-CHUNK (SHORT SOUNDS) OPTIMIZATION ---
    print("\n--- Phase 1: Jointly Optimizing on Scaled Micro-Chunks (Short Sounds) ---")
    
    csv_micro_path = os.path.join(cur_dir, "dataset_micro.csv")
    if not os.path.exists(csv_micro_path):
        print(f"Error: {csv_micro_path} not found.")
        return
        
    micro_records = []
    with open(csv_micro_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            micro_records.append(row)
            
    print(f"Loading {len(micro_records)} micro target files and precomputing Mel-spectrograms...")
    micro_dataset = []
    target_voice_af = pipeline.load_voice('af_heart').to(device)
    
    for idx, rec in enumerate(micro_records, 1):
        word = rec["text"]
        audio_file = rec["audio_file"]
        audio_path = os.path.join(chunks_dir, audio_file)
        
        # Load wave
        waveform, sr = sf.read(audio_path)
        target_waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
        target_mel = get_mel_spectrogram(target_waveform)
        
        # Get phonemes
        phonemes_tuple = pipeline.g2p(word)
        phonemes = ""
        for _, ps, _ in pipeline.en_tokenize(phonemes_tuple[1]):
            phonemes += ps
            
        # Get input_ids
        input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
        
        # Precompute target duration
        ps_length = len(phonemes)
        voice_idx = ps_length - 1 if ps_length > 0 else 0
        target_ref_s = target_voice_af[voice_idx]
        with torch.no_grad():
            _, target_dur = model.forward_with_tokens_grad(input_ids, target_ref_s)
            
        micro_dataset.append({
            'word': word,
            'input_ids': input_ids,
            'target_mel': target_mel,
            'target_dur': target_dur
        })
        
    # Initialize ref_s starting from af_heart (female baseline) to match Hatsune Miku
    start_voice = pipeline.load_voice('af_heart').to(device)
    ref_s = start_voice[0].clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([ref_s], lr=0.05)
    epochs_p1 = 8
    
    for epoch in range(epochs_p1):
        optimizer.zero_grad()
        epoch_loss = 0.0
        
        for item in micro_dataset:
            audio_pred, pred_dur = model.forward_with_tokens_grad(item['input_ids'], ref_s, target_dur=item['target_dur'])
            audio_pred = audio_pred.unsqueeze(0)
            pred_mel = get_mel_spectrogram(audio_pred)
            
            diff = item['target_mel'].size(-1) - pred_mel.size(-1)
            if diff > 0:
                pred_mel = F.pad(pred_mel, (0, diff))
            elif diff < 0:
                pred_mel = pred_mel[:, :, :item['target_mel'].size(-1)]
                
            loss = F.l1_loss(pred_mel, item['target_mel'])
            loss.backward()
            epoch_loss += loss.item()
            
        optimizer.step()
        print(f"Phase 1 Epoch {epoch:02d} | Avg Loss: {epoch_loss / len(micro_dataset):.4f}", flush=True)

    print("Phase 1 complete! Baseline ref_s aligned.")

    # --- PHASE 2: MACRO-CHUNK (SENTENCES) FINE-TUNING ---
    print("\n--- Phase 2: Jointly Fine-Tuning on 80 Sentences (~10 mins audio) ---")
    
    csv_macro_path = os.path.join(cur_dir, "dataset_macro.csv")
    if not os.path.exists(csv_macro_path):
        print(f"Error: {csv_macro_path} not found.")
        return
        
    macro_records = []
    with open(csv_macro_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            macro_records.append(row)
            
    print(f"Loading {len(macro_records)} macro sentence targets and precomputing Mel-spectrograms...")
    macro_dataset = []
    
    for idx, rec in enumerate(macro_records, 1):
        sentence = rec["text"]
        audio_file = rec["audio_file"]
        audio_path = os.path.join(chunks_dir, audio_file)
        
        # Load wave
        waveform, sr = sf.read(audio_path)
        target_waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
        target_mel = get_mel_spectrogram(target_waveform)
        
        # Get phonemes
        phonemes = ""
        for _, ps, _ in pipeline.en_tokenize(pipeline.g2p(sentence)[1]):
            phonemes += ps
        if len(phonemes) > 510:
            phonemes = phonemes[:510]
            
        # Get input_ids
        input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
        
        # Precompute target duration
        ps_length = len(phonemes)
        voice_idx = ps_length - 1 if ps_length > 0 else 0
        target_ref_s = target_voice_af[voice_idx]
        with torch.no_grad():
            _, target_dur = model.forward_with_tokens_grad(input_ids, target_ref_s)
            
        macro_dataset.append({
            'sentence': sentence,
            'input_ids': input_ids,
            'target_mel': target_mel,
            'target_dur': target_dur
        })
        
    # Re-initialize optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.Adam([ref_s], lr=0.01)
    epochs_p2 = 8
    
    for epoch in range(epochs_p2):
        optimizer.zero_grad()
        epoch_loss = 0.0
        
        for item in macro_dataset:
            audio_pred, pred_dur = model.forward_with_tokens_grad(item['input_ids'], ref_s, target_dur=item['target_dur'])
            audio_pred = audio_pred.unsqueeze(0)
            pred_mel = get_mel_spectrogram(audio_pred)
            
            diff = item['target_mel'].size(-1) - pred_mel.size(-1)
            if diff > 0:
                pred_mel = F.pad(pred_mel, (0, diff))
            elif diff < 0:
                pred_mel = pred_mel[:, :, :item['target_mel'].size(-1)]
                
            loss = F.l1_loss(pred_mel, item['target_mel'])
            loss.backward()
            epoch_loss += loss.item()
            
        optimizer.step()
        print(f"Phase 2 Epoch {epoch:02d} | Avg Loss: {epoch_loss / len(macro_dataset):.4f}", flush=True)

    print("Phase 2 complete! Final voice parameters trained.")
    
    # Save optimized token
    output_pt = os.path.join(cur_dir, "optimized_ref_s.pt")
    torch.save(ref_s.detach().cpu(), output_pt)
    print(f"Saved optimized_ref_s.pt to {output_pt}")

    # --- PHASE 3: GENERATE COMPARISON AUDIOS ---
    print("\n--- Phase 3: Generating Comparison Audios ---")
    test_text = "Hello there! This is a test of the scaled curriculum voice optimization process. I am speaking using the newly trained voice parameters."
    
    # 1. Source Original Voice (af_heart)
    audio_data = [result.audio for result in pipeline(test_text, voice='af_heart', speed=1) if result.audio is not None]
    if audio_data:
        full_audio = np.concatenate(audio_data)
        sf.write(os.path.join(result_dir, "test_source_original.wav"), full_audio, 24000)
        
    # 2. Target Timbre Reference (macro_chunk_01.wav)
    tgt_wav = os.path.join(chunks_dir, "macro_chunk_01.wav")
    if os.path.exists(tgt_wav):
        y, _ = librosa.load(tgt_wav, sr=24000)
        sf.write(os.path.join(result_dir, "test_target_reference.wav"), y, 24000)
        
    # 3. Scale-Optimized Voice
    opt_tensor = ref_s.detach()
    if opt_tensor.dim() == 2:
        opt_tensor = opt_tensor.unsqueeze(0).expand(510, 1, 256)
    audio_data = [result.audio for result in pipeline(test_text, voice=opt_tensor, speed=1) if result.audio is not None]
    if audio_data:
        full_audio = np.concatenate(audio_data)
        sf.write(os.path.join(result_dir, "test_optimized.wav"), full_audio, 24000)
        
    print(f"All files saved in: {result_dir}")

if __name__ == "__main__":
    main()
