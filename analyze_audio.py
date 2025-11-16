"""
Analyze audio quality and embedding stability
"""

import wave
import numpy as np
from scipy import signal as scipy_signal
from speaker_diarization_robust import ResemblyzerEmbeddings


def load_and_analyze_wav(filename):
    """Load and analyze WAV file"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {filename}")
    print(f"{'='*70}")
    
    with wave.open(filename, 'rb') as wav:
        sr = wav.getframerate()
        n_channels = wav.getnchannels()
        n_frames = wav.getnframes()
        duration = n_frames / sr
        
        audio_bytes = wav.readframes(n_frames)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
        print(f"Duration: {duration:.1f}s")
        print(f"Sample rate: {sr} Hz")
        print(f"Channels: {n_channels}")
        
        # Resample to 16kHz
        if sr != 16000:
            num_samples = int(len(audio) * 16000 / sr)
            audio = scipy_signal.resample(audio, num_samples).astype(np.int16)
            sr = 16000
            print(f"Resampled to 16000 Hz")
            
        # Calculate RMS levels
        rms_full = np.sqrt(np.mean(audio.astype(np.float32)**2))
        print(f"\nAudio Level (RMS): {int(rms_full)}")
        
        # Calculate SNR estimate
        frame_size = 16000  # 1 second
        frames = [audio[i:i+frame_size] for i in range(0, len(audio)-frame_size, frame_size)]
        frame_rms = [np.sqrt(np.mean(f.astype(np.float32)**2)) for f in frames]
        
        if frame_rms:
            sorted_rms = sorted(frame_rms)
            noise_floor = np.mean(sorted_rms[:len(sorted_rms)//5])  # Bottom 20%
            signal_power = np.mean(sorted_rms)
            
            if noise_floor > 0:
                snr_db = 20 * np.log10(signal_power / noise_floor)
                print(f"Estimated SNR: {snr_db:.1f} dB")
                
                if snr_db > 20:
                    print("   Quality: EXCELLENT")
                elif snr_db > 15:
                    print("   Quality: GOOD")
                elif snr_db > 10:
                    print("   Quality: ACCEPTABLE")
                else:
                    print("   Quality: POOR")
                    
        # Test embedding extraction on first 5 seconds
        print(f"\nTesting embedding extraction...")
        embedder = ResemblyzerEmbeddings()
        
        chunk_5s = audio[:16000*5]
        emb1 = embedder.extract_embedding(chunk_5s, 16000)
        
        print(f"   Embedding shape: {emb1.shape}")
        print(f"   Embedding norm: {np.linalg.norm(emb1):.3f}")
        print(f"   Non-zero elements: {np.count_nonzero(emb1)}/{len(emb1)}")
        
        # Test consistency: extract from multiple 5-second chunks
        print(f"\nTesting embedding consistency (same speaker, different chunks):")
        embeddings = []
        
        for i in range(min(6, int(duration / 5))):
            start = i * 16000 * 5
            end = start + 16000 * 5
            
            if end > len(audio):
                break
                
            chunk = audio[start:end]
            emb = embedder.extract_embedding(chunk, 16000)
            embeddings.append(emb)
            
        # Calculate pairwise similarities
        if len(embeddings) >= 2:
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j])
                    similarities.append(sim)
                    
            avg_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)
            
            print(f"   Chunks analyzed: {len(embeddings)}")
            print(f"   Average self-similarity: {avg_sim:.3f}")
            print(f"   Std deviation: {std_sim:.3f}")
            print(f"   Range: {min_sim:.3f} - {max_sim:.3f}")
            
            if avg_sim > 0.85:
                print(f"   ✅ EXCELLENT consistency")
            elif avg_sim > 0.75:
                print(f"   ✅ GOOD consistency")
            elif avg_sim > 0.65:
                print(f"   ⚠️ MODERATE consistency")
            else:
                print(f"   ❌ POOR consistency - voice varies too much or quality issues")
                
        return {
            'filename': filename,
            'duration': duration,
            'rms': rms_full,
            'avg_self_similarity': avg_sim if len(embeddings) >= 2 else 0.0,
            'embeddings': embeddings
        }


if __name__ == "__main__":
    files = [
        "Kavin Interview77 (1).wav",
        "vid_orig_obf (1).wav", 
        "JiaJun_video_3 1.wav"
    ]
    
    print("="*70)
    print("  AUDIO QUALITY & EMBEDDING CONSISTENCY ANALYSIS")
    print("="*70)
    
    results = []
    
    for f in files:
        try:
            result = load_and_analyze_wav(f)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
    # Cross-file comparison
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print("CROSS-SPEAKER SIMILARITY (should be LOW):")
        print(f"{'='*70}")
        
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                file1 = results[i]['filename']
                file2 = results[j]['filename']
                
                embs1 = results[i]['embeddings']
                embs2 = results[j]['embeddings']
                
                if embs1 and embs2:
                    # Calculate cross-speaker similarity
                    cross_sims = []
                    for e1 in embs1[:3]:  # First 3 chunks each
                        for e2 in embs2[:3]:
                            sim = np.dot(e1, e2)
                            cross_sims.append(sim)
                            
                    avg_cross = np.mean(cross_sims)
                    
                    print(f"{file1.split()[0]} vs {file2.split()[0]}: {avg_cross:.3f}")
                    
                    if avg_cross < 0.60:
                        print(f"   ✅ GOOD separation")
                    elif avg_cross < 0.70:
                        print(f"   ⚠️ MODERATE separation")
                    else:
                        print(f"   ❌ POOR separation - too similar!")

