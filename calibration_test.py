"""
Calibration Test Script
Listen to user speaking and display ALL parameters for tuning

This will help determine optimal thresholds for:
- RMS (speech detection)
- Voice similarity thresholds
- Rejection parameters
- Transcription accuracy
"""

import numpy as np
import whisper
import time
from audio_capture import AudioCapture
from speaker_diarization_robust import ResemblyzerEmbeddings
from simple_robust_verification import SimpleRobustVerifier
import pyaudio

print("="*100)
print("CALIBRATION TEST - Speak normally and we'll capture all parameters")
print("="*100)
print()

# Initialize models
print("Loading models...")
model = whisper.load_model("base")
embedder = ResemblyzerEmbeddings()
verifier = SimpleRobustVerifier(base_threshold=0.60)

# Audio capture
audio = AudioCapture(sample_rate=16000, channels=1, device_index=5)
audio.start()

print("\n" + "="*100)
print("READY - Start speaking normally...")
print("Press Ctrl+C to stop")
print("="*100 + "\n")

try:
    chunk_count = 0
    buffer = []
    buffer_duration = 2.0  # 2 seconds of audio (faster processing)
    samples_needed = int(16000 * buffer_duration)
    max_chunks = 20  # Stop after 20 chunks (about 40 seconds)
    
    print("Collecting audio... (speak now)")
    print(f"Will process {max_chunks} chunks and then stop automatically")
    print("Or press Ctrl+C to stop early\n")
    
    last_print_time = time.time()
    
    while chunk_count < max_chunks:
        # Get audio chunk
        try:
            audio_data = audio.get_audio_chunk(timeout=0.1)
            if audio_data is not None:
                buffer.extend(audio_data)
        except:
            pass
        
        # Show progress every 2 seconds
        current_time = time.time()
        if current_time - last_print_time > 2.0:
            buffer_seconds = len(buffer) / 16000
            if len(buffer) > 0:
                recent_rms = np.sqrt(np.mean(np.array(buffer[-1600:], dtype=np.float32) ** 2))
                print(f"[Progress] Chunks: {chunk_count}/{max_chunks} | Buffer: {buffer_seconds:.1f}s | RMS: {recent_rms:.0f}")
            last_print_time = current_time
        
        # Check if we have enough audio
        if len(buffer) >= samples_needed:
            # Process chunk
            audio_chunk = np.array(buffer[:samples_needed], dtype=np.int16)
            buffer = buffer[samples_needed:]  # Keep remainder
            
            chunk_count += 1
            
            # === CALCULATE RMS ===
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            
            # === TRANSCRIBE ===
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            result = whisper.transcribe(model, audio_float, language='en', fp16=False, verbose=False)
            transcribed_text = result['text'].strip()
            
            # === EXTRACT VOICE EMBEDDING ===
            embedding = embedder.extract_embedding(audio_chunk)
            
            # === CALCULATE SIMILARITY (if we have enrolled speakers) ===
            # For now, just show the embedding stats
            
            # === DISPLAY ALL METRICS ===
            print(f"\n{'='*100}")
            print(f"CHUNK #{chunk_count} - {time.strftime('%H:%M:%S')}")
            print(f"{'='*100}")
            
            print(f"\n[AUDIO METRICS]")
            print(f"   RMS Level: {rms:.1f}")
            print(f"   Audio Length: {len(audio_chunk)/16000:.2f} seconds")
            print(f"   Max Amplitude: {np.max(np.abs(audio_chunk))}")
            print(f"   Mean Amplitude: {np.mean(np.abs(audio_chunk)):.1f}")
            print(f"   Std Deviation: {np.std(audio_chunk):.1f}")
            
            # RMS categories
            if rms < 500:
                rms_category = "VERY LOW (likely silence/noise)"
            elif rms < 1000:
                rms_category = "LOW (quiet speech)"
            elif rms < 2000:
                rms_category = "NORMAL (typical speech)"
            elif rms < 4000:
                rms_category = "LOUD (clear speech)"
            else:
                rms_category = "VERY LOUD (shouting)"
            print(f"   RMS Category: {rms_category}")
            
            print(f"\n[TRANSCRIPTION]")
            print(f"   Text: \"{transcribed_text}\"")
            print(f"   Length: {len(transcribed_text)} characters")
            print(f"   Words: {len(transcribed_text.split())}")
            if transcribed_text:
                print(f"   Confidence: {result.get('language_prob', 'N/A')}")
            
            print(f"\n[VOICE EMBEDDING]")
            print(f"   Embedding Shape: {embedding.shape}")
            print(f"   Embedding Norm: {np.linalg.norm(embedding):.4f}")
            print(f"   Embedding Mean: {np.mean(embedding):.6f}")
            print(f"   Embedding Std: {np.std(embedding):.6f}")
            print(f"   Embedding Min: {np.min(embedding):.6f}")
            print(f"   Embedding Max: {np.max(embedding):.6f}")
            
            # Show first few embedding values
            print(f"   First 5 values: {embedding[:5]}")
            
            print(f"\n[RECOMMENDED THRESHOLDS]")
            print(f"   RMS Threshold: {rms * 0.3:.0f} - {rms * 0.5:.0f} (30-50% of current RMS)")
            print(f"   Min RMS for speech: {rms * 0.4:.0f}")
            print(f"   Max RMS for speech: {rms * 2.0:.0f}")
            
            print(f"\n{'='*100}\n")
        
        time.sleep(0.01)  # Small delay
    
    print(f"\n\n{'='*100}")
    print(f"Processed {chunk_count} chunks. Stopping automatically.")
    print(f"{'='*100}")
        
except KeyboardInterrupt:
    print("\n\n" + "="*100)
    print("CALIBRATION TEST COMPLETE")
    print("="*100)
    print(f"\nTotal chunks processed: {chunk_count}")
    print("\nReview the metrics above to determine optimal parameters:")
    print("  - RMS threshold for speech detection")
    print("  - Voice similarity thresholds")
    print("  - Transcription accuracy")
    print("="*100)
    
finally:
    audio.stop()

