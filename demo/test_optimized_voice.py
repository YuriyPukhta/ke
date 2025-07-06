import os
import sys
import torch
import soundfile as sf
import numpy as np

# Add parent to get KPipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kokoro.pipeline import KPipeline

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Initialize Pipeline
    pipeline = KPipeline(lang_code='a')
    
    # 2. Text to generate
    text = "Hello there! I am testing the new optimized voice token. This is a completely fresh sentence, to see how the style has changed."
    
    # 3. Generate Target (af_heart)
    print("Generating target voice (af_heart)...")
    audio_data = []
    for _, _, audio in pipeline(text, voice='af_heart', speed=1):
        if audio is not None:
            audio_data.append(audio)
    if audio_data:
        full_audio = np.concatenate(audio_data)
        sf.write(os.path.join(os.path.dirname(__file__), "result", "test_target_heart.wav"), full_audio, 24000)
    
    # 4. Generate Original (am_michael)
    print("Generating original voice (am_michael)...")
    audio_data = []
    for _, _, audio in pipeline(text, voice='am_michael', speed=1):
        if audio is not None:
            audio_data.append(audio)
    if audio_data:
        full_audio = np.concatenate(audio_data)
        sf.write(os.path.join(os.path.dirname(__file__), "result", "test_original_michael.wav"), full_audio, 24000)
    
    # 5. Load optimized token and pad it to match voice pack shape
    print("Loading optimized token...")
    opt_tensor = torch.load(os.path.join(os.path.dirname(__file__), "optimized_ref_s.pt")).to(device)
    
    # opt_tensor is [1, 256]. A standard pack is [510, 1, 256] or similar.
    # The pipeline does: pack = self.load_voice(voice).to(self.device); ref_s = pack[len(ps)-1]
    # To be fully compatible, we can just expand our optimized tensor to [510, 1, 256].
    if opt_tensor.dim() == 2:
        # [1, 256] -> [510, 1, 256]
        opt_tensor = opt_tensor.unsqueeze(0).expand(510, 1, 256)
        
    print("Generating optimized voice...")
    audio_data = []
    for _, _, audio in pipeline(text, voice=opt_tensor, speed=1):
        if audio is not None:
            audio_data.append(audio)
    if audio_data:
        full_audio = np.concatenate(audio_data)
        sf.write(os.path.join(os.path.dirname(__file__), "result", "test_optimized.wav"), full_audio, 24000)
    
    print("Done! Check demo folder for the 3 wav files.")

if __name__ == "__main__":
    main()
