import os
import sys
import shutil
import torch
import torchaudio
import soundfile as sf
import numpy as np
import librosa
import csv
import torch.nn.functional as F

# Add FreeVC paths to import utils and models
FREEVC_DIR = r"d:\projec\freevc"
sys.path.append(FREEVC_DIR)

import utils
from models import SynthesizerTrn
from speaker_encoder.voice_encoder import SpeakerEncoder

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
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. Initialize Pipeline & Sandboxed KModel
    pipeline = KPipeline(lang_code='a')
    model = KModel(repo_id='hexgrad/Kokoro-82M').to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
        
    # --- PHASE 0: DATASET GENERATION ---
    print("\n--- Phase 0: Curriculum Dataset Generation ---")
    
    # Use first sentence from Gatsby
    sentence = "In my younger and more vulnerable years my father gave me some advice that I’ve been turning over in my mind ever since."
    words = sorted(list(set([w.strip('“”,.()”"“') for w in sentence.split() if w.strip('“”,.()”"“')])))
    
    # Load FreeVC models for micro-chunks voice conversion
    print("Loading FreeVC generator model...")
    hps = utils.get_hparams_from_file(os.path.join(FREEVC_DIR, "configs", "freevc.json"))
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    net_g.eval()
    utils.load_checkpoint(os.path.join(FREEVC_DIR, "checkpoints", "freevc.pth"), net_g, None, True)
    
    print("Loading WavLM model...")
    cmodel = utils.get_cmodel(0)
    cmodel.to(device)
    
    print("Loading Speaker Encoder...")
    smodel = SpeakerEncoder(os.path.join(FREEVC_DIR, 'speaker_encoder/ckpt/pretrained_bak_5805000.pt'))
    
    # Extract target speaker embedding once
    tgt_wav = os.path.join(FREEVC_DIR, "target.wav")
    print(f"Extracting target embedding from {tgt_wav}...")
    wav_tgt, _ = librosa.load(tgt_wav, sr=hps.data.sampling_rate)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
    g_tgt = smodel.embed_utterance(wav_tgt)
    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)
    
    # Filter words to 1-3 phonemes and generate/convert audios
    micro_chunks = []
    print("Generating micro-chunks (1-3 phonemes)...")
    for word in words:
        phonemes_tuple = pipeline.g2p(word)
        phonemes = ""
        for _, ps, _ in pipeline.en_tokenize(phonemes_tuple[1]):
            phonemes += ps
        
        # Only keep words with 1-3 phonemes
        if 1 <= len(phonemes) <= 3:
            audio_filename = f"micro_chunk_{word}.wav"
            audio_path = os.path.join(chunks_dir, audio_filename)
            
            # Generate Kokoro base audio for this word
            audio_data = [result.audio for result in pipeline(word, voice='af_heart', speed=1) if result.audio is not None]
            if audio_data:
                final_audio = torch.cat(audio_data, dim=0)
                # Resample 24k -> 16k for FreeVC content extraction
                wav_16k = librosa.resample(final_audio.cpu().numpy(), orig_sr=24000, target_sr=16000)
                
                # Perform FreeVC voice conversion
                with torch.no_grad():
                    wav_src_torch = torch.from_numpy(wav_16k).unsqueeze(0).to(device)
                    c = utils.get_content(cmodel, wav_src_torch)
                    audio_cloned = net_g.infer(c, g=g_tgt)
                    audio_cloned_np = audio_cloned[0][0].data.cpu().float().numpy()
                
                # Resample output from 16k back to 24k for Kokoro optimization
                audio_cloned_24k = librosa.resample(audio_cloned_np, orig_sr=16000, target_sr=24000)
                sf.write(audio_path, audio_cloned_24k, 24000)
                
                micro_chunks.append({
                    'word': word,
                    'phonemes': phonemes,
                    'audio_file': audio_filename,
                    'audio_path': audio_path
                })
                print(f"Generated micro-chunk: '{word}' -> phonemes: '{phonemes.encode('ascii', 'backslashreplace').decode('ascii')}' ({len(phonemes)} phonemes)")
                
    # Create dataset_micro.csv
    csv_micro_path = os.path.join(cur_dir, "dataset_micro.csv")
    with open(csv_micro_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "ref_s", "audio_file", "model"])
        for chunk in micro_chunks:
            writer.writerow([chunk['word'], "af_heart", chunk['audio_file'], "hexgrad/Kokoro-82M"])
            
    # Copy macro target chunk (resampled sentence) from cloned_voice_demo if it exists
    macro_src_chunk = os.path.join(KOKORO_DIR, "cloned_voice_demo", "chunks", "chunk_01.wav")
    macro_dest_chunk = os.path.join(chunks_dir, "macro_chunk.wav")
    if os.path.exists(macro_src_chunk):
        print(f"Copying macro-chunk target reference from {macro_src_chunk}...")
        shutil.copy(macro_src_chunk, macro_dest_chunk)
    else:
        # Fallback to copy target.wav directly
        print("Macro source chunk not found. Resampling target.wav to serve as macro-chunk...")
        y, _ = librosa.load(tgt_wav, sr=24000)
        sf.write(macro_dest_chunk, y, 24000)
        
    # Create dataset_macro.csv
    csv_macro_path = os.path.join(cur_dir, "dataset_macro.csv")
    with open(csv_macro_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "ref_s", "audio_file", "model"])
        writer.writerow([sentence, "af_heart", "macro_chunk.wav", "hexgrad/Kokoro-82M"])

    # Clean up FreeVC models to free memory
    del net_g, cmodel, smodel
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- PHASE 1: MICRO-CHUNK JOINT OPTIMIZATION ---
    print("\n--- Phase 1: Jointly Optimizing on Micro-Chunks (1-3 Phonemes) ---")
    
    # Load micro targets and precompute target mels and target durs
    micro_dataset = []
    target_voice_af = pipeline.load_voice('af_heart').to(device)
    
    for chunk in micro_chunks:
        # Load wave
        waveform, sr = sf.read(chunk['audio_path'])
        target_waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
        target_mel = get_mel_spectrogram(target_waveform)
        
        # Tokenize and get input_ids
        input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), chunk['phonemes'])))
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
        
        # Grab target_dur from af_heart voice
        ps_length = len(chunk['phonemes'])
        idx = ps_length - 1 if ps_length > 0 else 0
        target_ref_s = target_voice_af[idx]
        with torch.no_grad():
            _, target_dur = model.forward_with_tokens_grad(input_ids, target_ref_s)
            
        micro_dataset.append({
            'word': chunk['word'],
            'input_ids': input_ids,
            'target_mel': target_mel,
            'target_dur': target_dur,
            'idx': idx
        })
        
    # Initialize ref_s starting from am_michael
    start_voice = pipeline.load_voice('am_michael').to(device)
    ref_s_tensor = start_voice[0] # baseline embedding shape [1, 256]
    ref_s = ref_s_tensor.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([ref_s], lr=0.05)
    epochs_p1 = 30
    
    for epoch in range(epochs_p1):
        optimizer.zero_grad()
        epoch_loss = 0.0
        
        for item in micro_dataset:
            # Forward pass
            audio_pred, pred_dur = model.forward_with_tokens_grad(item['input_ids'], ref_s, target_dur=item['target_dur'])
            audio_pred = audio_pred.unsqueeze(0)
            
            # Convert to Mel
            pred_mel = get_mel_spectrogram(audio_pred)
            
            # Pad or truncate
            diff = item['target_mel'].size(-1) - pred_mel.size(-1)
            if diff > 0:
                pred_mel = F.pad(pred_mel, (0, diff))
            elif diff < 0:
                pred_mel = pred_mel[:, :, :item['target_mel'].size(-1)]
                
            loss = F.l1_loss(pred_mel, item['target_mel'])
            loss.backward()
            epoch_loss += loss.item()
            
        optimizer.step()
        print(f"Phase 1 Epoch {epoch:03d} | Avg Loss: {epoch_loss / len(micro_dataset):.4f}", flush=True)

    print("Phase 1 complete! Baseline ref_s aligned.")

    # --- PHASE 2: MACRO-CHUNK FINE-TUNING ---
    print("\n--- Phase 2: Fine-Tuning on Macro-Chunk (Full Sentence) ---")
    
    # Load macro target
    waveform, sr = sf.read(macro_dest_chunk)
    target_waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
    target_mel = get_mel_spectrogram(target_waveform)
    
    # Get Phonemes & Input IDs for full sentence
    phonemes = ""
    for _, ps, _ in pipeline.en_tokenize(pipeline.g2p(sentence)[1]):
        phonemes += ps
    if len(phonemes) > 510:
        phonemes = phonemes[:510]
        
    input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
    
    # Grab target duration using af_heart
    ps_length = len(phonemes)
    idx = ps_length - 1 if ps_length > 0 else 0
    target_ref_s = target_voice_af[idx]
    with torch.no_grad():
        _, target_dur = model.forward_with_tokens_grad(input_ids, target_ref_s)
        
    # Re-initialize optimizer with the aligned ref_s
    optimizer = torch.optim.Adam([ref_s], lr=0.01) # lower lr for fine-tuning
    epochs_p2 = 30
    
    for epoch in range(epochs_p2):
        optimizer.zero_grad()
        
        audio_pred, pred_dur = model.forward_with_tokens_grad(input_ids, ref_s, target_dur=target_dur)
        audio_pred = audio_pred.unsqueeze(0)
        pred_mel = get_mel_spectrogram(audio_pred)
        
        diff = target_mel.size(-1) - pred_mel.size(-1)
        if diff > 0:
            pred_mel = F.pad(pred_mel, (0, diff))
        elif diff < 0:
            pred_mel = pred_mel[:, :, :target_mel.size(-1)]
            
        loss = F.l1_loss(pred_mel, target_mel)
        loss.backward()
        optimizer.step()
        print(f"Phase 2 Epoch {epoch:03d} | Loss: {loss.item():.4f}", flush=True)

    print("Phase 2 complete! Final voice parameters trained.")
    
    # Save optimized token
    output_pt = os.path.join(cur_dir, "optimized_ref_s.pt")
    torch.save(ref_s.detach().cpu(), output_pt)
    print(f"Saved optimized_ref_s.pt to {output_pt}")

    # --- PHASE 3: GENERATE COMPARISON AUDIOS ---
    print("\n--- Phase 3: Generating Comparison Audios ---")
    test_text = "Hello there! This is a test of the curriculum voice optimization process. I am speaking using the newly trained voice parameters."
    
    # 1. Source Original Voice
    audio_data = [result.audio for result in pipeline(test_text, voice='am_michael', speed=1) if result.audio is not None]
    if audio_data:
        full_audio = np.concatenate(audio_data)
        sf.write(os.path.join(result_dir, "test_source_original.wav"), full_audio, 24000)
        
    # 2. Target Timbre Reference
    if os.path.exists(tgt_wav):
        y, _ = librosa.load(tgt_wav, sr=24000)
        sf.write(os.path.join(result_dir, "test_target_reference.wav"), y, 24000)
        
    # 3. Curriculum-Optimized Voice
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
