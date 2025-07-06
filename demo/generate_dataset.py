import os
import csv
import sys
import soundfile as sf

# Add the parent directory to sys.path so we can import kokoro
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kokoro import KPipeline

def main():
    input_file = r"d:\projec\kokoro\demo\gatsby5k.md"
    output_dir = r"d:\projec\kokoro\demo\chunks"
    csv_file = r"d:\projec\kokoro\demo\dataset.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    
    voice = "af_heart"  # ref_s
    model_name = "hexgrad/Kokoro-82M"
    
    print("Initializing KPipeline...")
    pipeline = KPipeline(lang_code='a')
    
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split text into short chunks
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    
    with open(csv_file, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["text", "ref_s", "audio_file", "model"])
        
        for i, chunk in enumerate(chunks, 1):
            audio_filename = f"chunk_{i:02d}.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            
            print(f"Generating audio for chunk {i}/{len(chunks)}...")
            
            # Generate audio using Kokoro
            generator = pipeline(chunk, voice=voice, speed=1, split_pattern=r'\n+')
            
            audio_pieces = []
            for result in generator:
                if result.audio is not None:
                    audio_pieces.append(result.audio)
            
            if audio_pieces:
                import torch
                # Concatenate audio pieces if the chunk was split further
                final_audio = torch.cat(audio_pieces, dim=0)
                sf.write(audio_path, final_audio.cpu().numpy(), 24000)
                
                # Write to CSV
                writer.writerow([chunk, voice, audio_filename, model_name])
                print(f"Saved {audio_filename}")
            else:
                print(f"Warning: No audio generated for chunk {i}")

if __name__ == "__main__":
    main()
