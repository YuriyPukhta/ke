import os
import sys
import csv
import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F

# Add kokoro_revers to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo"))
from kokoro_revers.model import KModel

# Add parent to get KPipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kokoro.pipeline import KPipeline

def get_mel_spectrogram(waveform, sample_rate=24000):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    ).to(waveform.device)
    mel = mel_transform(waveform)
    # Log mel
    return torch.log(torch.clamp(mel, min=1e-5))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Initialize Pipeline & Sandboxed Model
    pipeline = KPipeline(lang_code='a', model=False)
    
    # We use the isolated KModel
    model = KModel(repo_id='hexgrad/Kokoro-82M').to(device).eval()
    
    # Ensure all original model weights don't require gradients
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Load dataset and target
    csv_file = os.path.join(os.path.dirname(__file__), "dataset.csv")
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        
    text = row["text"]
    audio_file = row["audio_file"]
    audio_path = os.path.join(os.path.dirname(__file__), "chunks", audio_file)
    print(f"Loading target audio from {audio_path}")
    
    waveform, sr = sf.read(audio_path)
    target_waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Resample if needed
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000).to(device)
        target_waveform = resampler(target_waveform)
        
    target_mel = get_mel_spectrogram(target_waveform)
    
    # 3. Get Phonemes & Input IDs
    print(f"Processing text: {text[:50]}...")
    phonemes = ""
    # Getting phonemes for the first chunk of text
    for _, ps, _ in pipeline.en_tokenize(pipeline.g2p(text)[1]):
        phonemes += ps
    
    if len(phonemes) > 510:
        phonemes = phonemes[:510]
        
    input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
    
    # 4. Initialize ref_s
    # We load a DIFFERENT voice (e.g., af_bella) to start the optimization
    # The target was generated with af_heart. We want to see if af_bella can optimize into af_heart.
    print("Loading starting voice 'am_michael'...")
    start_voice = pipeline.load_voice('am_michael').to(device)
    
    ps_length = len(phonemes)
    idx = ps_length - 1 if ps_length > 0 else 0
    ref_s_tensor = start_voice[idx] # [1, 256]
    
    ref_s = ref_s_tensor.clone().detach().requires_grad_(True)
    
    # Grab target_dur from af_heart (the voice used to generate the target audio)
    print("Extracting target duration from 'af_heart'...")
    target_voice = pipeline.load_voice('af_heart').to(device)
    target_ref_s = target_voice[idx]
    with torch.no_grad():
        _, target_dur = model.forward_with_tokens_grad(input_ids, target_ref_s)
        
    # 5. Optimizer
    learning_rate = 0.05
    optimizer = torch.optim.Adam([ref_s], lr=learning_rate)
    
    epochs = 30
    print(f"Starting optimization for {epochs} epochs...", flush=True)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass using our modified function that allows gradients
        audio_pred, pred_dur = model.forward_with_tokens_grad(input_ids, ref_s, target_dur=target_dur)
        
        # Audio is returned as 1D, make it 2D [channel, time]
        audio_pred = audio_pred.unsqueeze(0)
        
        # Convert to Mel
        pred_mel = get_mel_spectrogram(audio_pred)
        
        # Handle Length Mismatch by padding or truncating to target_mel length
        diff = target_mel.size(-1) - pred_mel.size(-1)
        if diff > 0:
            pred_mel = F.pad(pred_mel, (0, diff))
        elif diff < 0:
            pred_mel = pred_mel[:, :, :target_mel.size(-1)]
            
        # L1 Loss on Mel Spectrogram
        loss = F.l1_loss(pred_mel, target_mel)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}", flush=True)
            
    print(f"Final Loss: {loss.item():.4f}", flush=True)
    
    # Save the optimized ref_s
    output_pt = os.path.join(os.path.dirname(__file__), "optimized_ref_s.pt")
    torch.save(ref_s.detach().cpu(), output_pt)
    print(f"Saved optimized_ref_s.pt to {output_pt}")

if __name__ == "__main__":
    main()
