"""
Comprehensive Multi-Speaker Calibration
Collect data from BOTH male and female voices for production-level robustness

This ensures parameters work for all speakers, not just one gender.
"""

import numpy as np
import whisper
import time
from audio_capture import AudioCapture
from speaker_diarization_robust import ResemblyzerEmbeddings
import json
from datetime import datetime

print("="*100)
print("COMPREHENSIVE MULTI-SPEAKER CALIBRATION")
print("="*100)
print()
print("This will calibrate parameters for BOTH male and female voices")
print("for production-level robustness.")
print()

# Initialize models
print("Loading models...")
model = whisper.load_model("base")
embedder = ResemblyzerEmbeddings()

# Audio capture
audio = AudioCapture(sample_rate=16000, channels=1, device_index=5)
audio.start()

# Storage for all calibration data
all_data = {
    'male': {'rms': [], 'transcriptions': [], 'embeddings': []},
    'female': {'rms': [], 'transcriptions': [], 'embeddings': []}
}

def collect_speaker_data(speaker_type, speaker_name, num_chunks=15):
    """Collect calibration data from one speaker"""
    print(f"\n{'='*100}")
    print(f"SPEAKER: {speaker_name} ({speaker_type.upper()})")
    print(f"{'='*100}")
    print(f"Please speak normally for about {num_chunks * 3} seconds")
    print(f"Will collect {num_chunks} chunks")
    print("Press Enter when ready to start...")
    input()
    
    buffer = []
    buffer_duration = 3.0  # 3 seconds
    samples_needed = int(16000 * buffer_duration)
    chunk_count = 0
    
    print(f"\n[Recording {speaker_type} voice...]")
    print("Speak now...\n")
    
    start_time = time.time()
    last_print = time.time()
    
    while chunk_count < num_chunks:
        try:
            audio_data = audio.get_audio_chunk(timeout=0.1)
            if audio_data is not None:
                buffer.extend(audio_data)
        except:
            pass
        
        # Show progress
        if time.time() - last_print > 2.0:
            buffer_seconds = len(buffer) / 16000
            if len(buffer) > 0:
                recent_rms = np.sqrt(np.mean(np.array(buffer[-1600:], dtype=np.float32) ** 2))
                print(f"  Progress: {chunk_count}/{num_chunks} chunks | Buffer: {buffer_seconds:.1f}s | RMS: {recent_rms:.0f}")
            last_print = time.time()
        
        # Process when we have enough audio
        if len(buffer) >= samples_needed:
            audio_chunk = np.array(buffer[:samples_needed], dtype=np.int16)
            buffer = buffer[samples_needed:]
            
            chunk_count += 1
            
            # Calculate RMS
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            
            # Only process if RMS indicates speech
            if rms > 500:
                # Transcribe
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                result = whisper.transcribe(
                    model, 
                    audio_float, 
                    language='en', 
                    fp16=False, 
                    verbose=False,
                    beam_size=5,
                    temperature=0.0
                )
                transcribed_text = result['text'].strip()
                
                # Extract embedding
                embedding = embedder.extract_embedding(audio_chunk)
                
                # Store data
                all_data[speaker_type]['rms'].append(float(rms))
                all_data[speaker_type]['transcriptions'].append(transcribed_text)
                all_data[speaker_type]['embeddings'].append(embedding.tolist())
                
                print(f"  Chunk {chunk_count}: RMS={rms:.0f}, Text=\"{transcribed_text[:40]}...\"")
        
        time.sleep(0.01)
    
    elapsed = time.time() - start_time
    print(f"\n[Complete] Collected {chunk_count} chunks in {elapsed:.1f} seconds")
    print(f"  Average RMS: {np.mean(all_data[speaker_type]['rms']):.1f}")
    print(f"  RMS Range: {np.min(all_data[speaker_type]['rms']):.1f} - {np.max(all_data[speaker_type]['rms']):.1f}")

# === COLLECT DATA FROM BOTH SPEAKERS ===
print("\n" + "="*100)
print("STEP 1: MALE VOICE CALIBRATION")
print("="*100)
collect_speaker_data('male', 'Male Speaker', num_chunks=15)

print("\n" + "="*100)
print("STEP 2: FEMALE VOICE CALIBRATION")
print("="*100)
collect_speaker_data('female', 'Female Speaker', num_chunks=15)

audio.stop()

# === ANALYZE COMBINED DATA ===
print("\n\n" + "="*100)
print("ANALYZING COMBINED DATA")
print("="*100)

male_rms = np.array(all_data['male']['rms'])
female_rms = np.array(all_data['female']['rms'])
all_rms = np.concatenate([male_rms, female_rms])

print(f"\n[MALE VOICE STATISTICS]")
print(f"   Chunks: {len(male_rms)}")
print(f"   RMS Mean: {np.mean(male_rms):.1f}")
print(f"   RMS Median: {np.median(male_rms):.1f}")
print(f"   RMS Range: {np.min(male_rms):.1f} - {np.max(male_rms):.1f}")
print(f"   RMS Std: {np.std(male_rms):.1f}")

print(f"\n[FEMALE VOICE STATISTICS]")
print(f"   Chunks: {len(female_rms)}")
print(f"   RMS Mean: {np.mean(female_rms):.1f}")
print(f"   RMS Median: {np.median(female_rms):.1f}")
print(f"   RMS Range: {np.min(female_rms):.1f} - {np.max(female_rms):.1f}")
print(f"   RMS Std: {np.std(female_rms):.1f}")

print(f"\n[COMBINED STATISTICS (Both Genders)]")
print(f"   Total chunks: {len(all_rms)}")
print(f"   RMS Mean: {np.mean(all_rms):.1f}")
print(f"   RMS Median: {np.median(all_rms):.1f}")
print(f"   RMS Range: {np.min(all_rms):.1f} - {np.max(all_rms):.1f}")
print(f"   RMS Std: {np.std(all_rms):.1f}")

# Check for significant differences
male_median = np.median(male_rms)
female_median = np.median(female_rms)
difference_pct = abs(male_median - female_median) / max(male_median, female_median) * 100

print(f"\n[GENDER COMPARISON]")
print(f"   Male median RMS: {male_median:.1f}")
print(f"   Female median RMS: {female_median:.1f}")
print(f"   Difference: {difference_pct:.1f}%")
if difference_pct > 20:
    print(f"   ⚠️  Significant difference detected - may need gender-specific thresholds")
else:
    print(f"   ✓ Similar levels - single threshold should work")

# === RECOMMEND PARAMETERS ===
print(f"\n{'='*100}")
print("PRODUCTION-LEVEL PARAMETER RECOMMENDATIONS")
print(f"{'='*100}")

combined_median = np.median(all_rms)
combined_mean = np.mean(all_rms)

# Use the LOWER of the two medians to ensure we catch both
safe_median = min(male_median, female_median)

print(f"\n[RMS THRESHOLD RECOMMENDATIONS]")
print(f"   Option 1 (Conservative - catches both genders):")
print(f"      Threshold: {int(safe_median * 0.4)} (40% of lower median)")
print(f"      This ensures we catch quieter speakers")
print(f"\n   Option 2 (Balanced - based on combined data):")
print(f"      Threshold: {int(combined_median * 0.45)} (45% of combined median)")
print(f"      This is optimal for average case")
print(f"\n   Option 3 (Aggressive - filters more noise):")
print(f"      Threshold: {int(combined_median * 0.5)} (50% of combined median)")
print(f"      This filters more but might miss quiet speech")

recommended = int(combined_median * 0.45)
print(f"\n   [RECOMMENDED]: {recommended} (balanced approach)")

print(f"\n[VAD THRESHOLD]")
print(f"   Recommended: {int(recommended * 0.6)} (60% of RMS threshold)")

print(f"\n[TRANSCRIPTION ANALYSIS]")
male_trans = all_data['male']['transcriptions']
female_trans = all_data['female']['transcriptions']

print(f"   Male average length: {np.mean([len(t) for t in male_trans]):.1f} chars")
print(f"   Female average length: {np.mean([len(t) for t in female_trans]):.1f} chars")
print(f"   Male average words: {np.mean([len(t.split()) for t in male_trans]):.1f} words")
print(f"   Female average words: {np.mean([len(t.split()) for t in female_trans]):.1f} words")

# Check transcription quality
male_errors = sum(1 for t in male_trans if len(t) < 10 or any(ord(c) > 127 for c in t))
female_errors = sum(1 for t in female_trans if len(t) < 10 or any(ord(c) > 127 for c in t))

print(f"\n   Transcription issues:")
print(f"      Male: {male_errors}/{len(male_trans)} chunks with issues")
print(f"      Female: {female_errors}/{len(female_trans)} chunks with issues")

# === EMBEDDING ANALYSIS ===
print(f"\n[VOICE EMBEDDING ANALYSIS]")
male_emb = np.array(all_data['male']['embeddings'])
female_emb = np.array(all_data['female']['embeddings'])

male_emb_mean = np.mean(male_emb, axis=0)
female_emb_mean = np.mean(female_emb, axis=0)

# Calculate similarity between male and female embeddings
similarity = np.dot(male_emb_mean, female_emb_mean) / (np.linalg.norm(male_emb_mean) * np.linalg.norm(female_emb_mean))

print(f"   Male embedding norm: {np.linalg.norm(male_emb_mean):.4f}")
print(f"   Female embedding norm: {np.linalg.norm(female_emb_mean):.4f}")
print(f"   Gender similarity: {similarity:.4f}")
print(f"   (Lower = more different, Higher = more similar)")

if similarity > 0.7:
    print(f"   ⚠️  High similarity - may need gender-specific voice models")
else:
    print(f"   ✓ Good separation - single model should work")

# === SAVE DATA ===
calibration_data = {
    'timestamp': datetime.now().isoformat(),
    'male': {
        'rms_stats': {
            'mean': float(np.mean(male_rms)),
            'median': float(np.median(male_rms)),
            'min': float(np.min(male_rms)),
            'max': float(np.max(male_rms)),
            'std': float(np.std(male_rms))
        },
        'sample_count': len(male_rms)
    },
    'female': {
        'rms_stats': {
            'mean': float(np.mean(female_rms)),
            'median': float(np.median(female_rms)),
            'min': float(np.min(female_rms)),
            'max': float(np.max(female_rms)),
            'std': float(np.std(female_rms))
        },
        'sample_count': len(female_rms)
    },
    'combined': {
        'rms_stats': {
            'mean': float(np.mean(all_rms)),
            'median': float(np.median(all_rms)),
            'min': float(np.min(all_rms)),
            'max': float(np.max(all_rms)),
            'std': float(np.std(all_rms))
        },
        'total_samples': len(all_rms)
    },
    'recommendations': {
        'rms_threshold': recommended,
        'vad_threshold': int(recommended * 0.6),
        'gender_difference_pct': float(difference_pct),
        'embedding_similarity': float(similarity)
    }
}

with open('calibration_results.json', 'w') as f:
    json.dump(calibration_data, f, indent=2)

print(f"\n{'='*100}")
print("CALIBRATION COMPLETE")
print(f"{'='*100}")
print(f"\nData saved to: calibration_results.json")
print(f"\nRecommended parameters:")
print(f"   RMS Threshold: {recommended}")
print(f"   VAD Threshold: {int(recommended * 0.6)}")
print(f"\nThese parameters are optimized for BOTH male and female voices!")

