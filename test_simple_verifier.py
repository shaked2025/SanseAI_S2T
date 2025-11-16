"""
Test simple verifier with real WAV files
Prove it works BEFORE integrating into main system
"""

import wave
import numpy as np
from scipy import signal as scipy_signal
from speaker_diarization_robust import ResemblyzerEmbeddings
from speaker_enrollment import SpeakerEnrollment
from simple_robust_verification import SimpleRobustVerifier


def load_wav_chunks(filename, num_chunks=6):
    """Load WAV and extract chunks"""
    with wave.open(filename, 'rb') as wav:
        sr = wav.getframerate()
        n_channels = wav.getnchannels()
        
        audio_bytes = wav.readframes(wav.getnframes())
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
        if sr != 16000:
            num_samples = int(len(audio) * 16000 / sr)
            audio = scipy_signal.resample(audio, num_samples).astype(np.int16)
            
    # Extract chunks
    chunks = []
    for i in range(num_chunks):
        start = i * 16000 * 5
        end = start + 16000 * 5
        if end <= len(audio):
            chunks.append(audio[start:end])
            
    return chunks


print("="*80)
print("  TESTING SIMPLE VERIFIER WITH REAL WAV FILES")
print("="*80)
print()

# Load files
print("Loading WAV files...")
speaker1_chunks = load_wav_chunks("Kavin Interview77 (1).wav", 12)  # 12 chunks total
speaker2_chunks = load_wav_chunks("vid_orig_obf (1).wav", 12)
unknown_chunks = load_wav_chunks("JiaJun_video_3 1.wav", 6)

print(f"   Speaker 1: {len(speaker1_chunks)} chunks")
print(f"   Speaker 2: {len(speaker2_chunks)} chunks")
print(f"   Unknown: {len(unknown_chunks)} chunks")

# Initialize
embedder = ResemblyzerEmbeddings()
enrollment = SpeakerEnrollment(embedder)
verifier = SimpleRobustVerifier(base_threshold=0.65)

# Enroll with first 6 chunks
print("\n" + "="*80)
print("ENROLLMENT PHASE")
print("="*80)

for idx, (name, chunks) in enumerate([("Kavin", speaker1_chunks[:6]), ("VidOrig", speaker2_chunks[:6])]):
    print(f"\nEnrolling {name}...")
    speaker_key = f"speaker_{idx}"
    enrollment.start_enrollment(speaker_key, name, "Speaker")
    
    for i, chunk in enumerate(chunks):
        success, quality, msg = enrollment.add_enrollment_sample(speaker_key, chunk, 16000)
        print(f"   Sample {i+1}: quality={quality:.1%}")
        
    success, quality, msg = enrollment.complete_enrollment(speaker_key)
    print(f"   ✅ {msg}")

enrolled_dict = enrollment.get_enrolled_speakers()

# Test with DIFFERENT chunks from same speakers (chunks 6-12)
print("\n" + "="*80)
print("TESTING PHASE")
print("="*80)

results = {
    'enrolled_correct': 0,
    'enrolled_total': 0,
    'unknown_rejected': 0,
    'unknown_total': 0
}

print("\n--- Testing Speaker 1 (should be ACCEPTED) ---")
for i, chunk in enumerate(speaker1_chunks[6:9]):  # Chunks 7, 8, 9 (NOT used in enrollment)
    test_emb = embedder.extract_embedding(chunk, 16000)
    
    accept, best_key, best_name, similarity, reason = verifier.verify_speaker(
        test_emb, enrolled_dict, audio_quality=0.8
    )
    
    results['enrolled_total'] += 1
    
    if accept:
        results['enrolled_correct'] += 1
        print(f"   Chunk {i+1}: ✅ ACCEPTED as {best_name} (sim: {similarity:.3f}) - {reason}")
    else:
        print(f"   Chunk {i+1}: ❌ REJECTED (sim: {similarity:.3f}) - {reason}")

print("\n--- Testing Speaker 2 (should be ACCEPTED) ---")
for i, chunk in enumerate(speaker2_chunks[6:9]):
    test_emb = embedder.extract_embedding(chunk, 16000)
    
    accept, best_key, best_name, similarity, reason = verifier.verify_speaker(
        test_emb, enrolled_dict, audio_quality=0.8
    )
    
    results['enrolled_total'] += 1
    
    if accept:
        results['enrolled_correct'] += 1
        print(f"   Chunk {i+1}: ✅ ACCEPTED as {best_name} (sim: {similarity:.3f}) - {reason}")
    else:
        print(f"   Chunk {i+1}: ❌ REJECTED (sim: {similarity:.3f}) - {reason}")

print("\n--- Testing Unknown Speaker (should be REJECTED) ---")
for i, chunk in enumerate(unknown_chunks[:3]):
    test_emb = embedder.extract_embedding(chunk, 16000)
    
    accept, best_key, best_name, similarity, reason = verifier.verify_speaker(
        test_emb, enrolled_dict, audio_quality=0.8
    )
    
    results['unknown_total'] += 1
    
    if not accept:
        results['unknown_rejected'] += 1
        print(f"   Chunk {i+1}: ✅ REJECTED (sim: {similarity:.3f}) - {reason}")
    else:
        print(f"   Chunk {i+1}: ❌ FALSE ACCEPT as {best_name} (sim: {similarity:.3f})")

# Calculate metrics
print("\n" + "="*80)
print("RESULTS")
print("="*80)

tar = results['enrolled_correct'] / results['enrolled_total'] if results['enrolled_total'] > 0 else 0
trr = results['unknown_rejected'] / results['unknown_total'] if results['unknown_total'] > 0 else 0
accuracy = (results['enrolled_correct'] + results['unknown_rejected']) / (results['enrolled_total'] + results['unknown_total'])

print(f"True Accept Rate (enrolled): {tar:.1%} ({results['enrolled_correct']}/{results['enrolled_total']})")
print(f"True Reject Rate (unknown): {trr:.1%} ({results['unknown_rejected']}/{results['unknown_total']})")
print(f"Overall Accuracy: {accuracy:.1%}")

stats = verifier.get_statistics()
print(f"\nTotal verifications: {stats['total_checks']}")
print(f"Accepted: {stats['accepted']}")
print(f"Rejected: {stats['rejected']}")
print(f"Acceptance rate: {stats['accepted']/stats['total_checks']*100:.1f}%")

print("\n" + "="*80)

if tar >= 0.90 and trr >= 0.90:
    print("✅✅✅ SUCCESS! System works with SIMPLE approach!")
    print("This verifier should be used in production.")
else:
    print(f"⚠️ Needs more tuning. TAR={tar:.1%}, TRR={trr:.1%}")

