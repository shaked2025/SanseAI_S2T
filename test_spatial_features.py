"""
Test spatial location features with real WAV files
Verify that location features help distinguish passersby from enrolled speakers
"""

import wave
import numpy as np
from scipy import signal as scipy_signal
from speaker_diarization_robust import ResemblyzerEmbeddings
from speaker_enrollment import SpeakerEnrollment
from simple_robust_verification import SimpleRobustVerifier
from spatial_location_features import LocationAwareVerifier, SpatialLocationFeatures


def load_wav_chunks(filename, num_chunks=12):
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
print("  TESTING SPATIAL LOCATION FEATURES")
print("="*80)
print()

# Load files
speaker1_chunks = load_wav_chunks("Kavin Interview77 (1).wav", 12)
speaker2_chunks = load_wav_chunks("vid_orig_obf (1).wav", 12)
unknown_chunks = load_wav_chunks("JiaJun_video_3 1.wav", 6)

print(f"Loaded {len(speaker1_chunks)} + {len(speaker2_chunks)} + {len(unknown_chunks)} chunks")

# Initialize
embedder = ResemblyzerEmbeddings()
enrollment = SpeakerEnrollment(embedder)
base_verifier = SimpleRobustVerifier(base_threshold=0.64)
location_verifier = LocationAwareVerifier(base_verifier, spatial_weight=0.15)

# Enroll with first 6 chunks
print("\n" + "="*80)
print("ENROLLMENT WITH SPATIAL FINGERPRINTS")
print("="*80)

for idx, (name, chunks) in enumerate([("Kavin", speaker1_chunks[:6]), ("VidOrig", speaker2_chunks[:6])]):
    print(f"\nEnrolling {name}...")
    speaker_key = f"speaker_{idx}"
    enrollment.start_enrollment(speaker_key, name, "Speaker")
    
    for chunk in chunks:
        enrollment.add_enrollment_sample(speaker_key, chunk, 16000)
        
    enrollment.complete_enrollment(speaker_key)
    
    # Create spatial fingerprint
    location_verifier.enroll_spatial_profile(speaker_key, chunks)

enrolled_dict = enrollment.get_enrolled_speakers()

# Test with and without spatial features
print("\n" + "="*80)
print("COMPARISON: Voice-Only vs Voice+Spatial")
print("="*80)

test_cases = [
    ("Speaker 1 (enrolled)", speaker1_chunks[6:9], True),
    ("Speaker 2 (enrolled)", speaker2_chunks[6:9], True),
    ("Unknown", unknown_chunks[:3], False)
]

voice_only_correct = 0
spatial_correct = 0
total = 0

for test_name, test_chunks, is_enrolled in test_cases:
    print(f"\n--- {test_name} ({'should ACCEPT' if is_enrolled else 'should REJECT'}) ---")
    
    for i, chunk in enumerate(test_chunks):
        test_emb = embedder.extract_embedding(chunk, 16000)
        
        # Voice-only verification
        voice_accept, _, _, voice_sim, voice_reason = base_verifier.verify_speaker(
            test_emb, enrolled_dict, 0.8
        )
        
        # Voice + Spatial verification
        spatial_accept, spk_key, spk_name, combined_score, spatial_reason = location_verifier.verify_with_location(
            test_emb, chunk, enrolled_dict
        )
        
        # Check correctness
        voice_correct = (voice_accept == is_enrolled)
        spatial_correct_flag = (spatial_accept == is_enrolled)
        
        if voice_correct:
            voice_only_correct += 1
        if spatial_correct_flag:
            spatial_correct += 1
        total += 1
        
        # Display
        voice_result = "✅" if voice_correct else "❌"
        spatial_result = "✅" if spatial_correct_flag else "❌"
        
        print(f"   Chunk {i+1}:")
        print(f"      Voice-only: {voice_result} (sim: {voice_sim:.3f}) - {voice_reason}")
        print(f"      Voice+Spatial: {spatial_result} (combined: {combined_score:.3f}) - {spatial_reason}")

# Summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

voice_only_accuracy = voice_only_correct / total
spatial_accuracy = spatial_correct / total

print(f"Voice-Only Accuracy: {voice_only_accuracy:.1%} ({voice_only_correct}/{total})")
print(f"Voice+Spatial Accuracy: {spatial_accuracy:.1%} ({spatial_correct}/{total})")

improvement = spatial_accuracy - voice_only_accuracy
print(f"\nImprovement from spatial features: {improvement:+.1%}")

if spatial_accuracy >= 0.95:
    print("\n✅✅✅ EXCELLENT! Spatial features boost accuracy!")
    print("Location-aware verification should be used in production.")
elif spatial_accuracy > voice_only_accuracy:
    print(f"\n✅ IMPROVED! Spatial features help (+{improvement:.1%})")
else:
    print("\n⚠️ No improvement. Spatial features may not help for these recordings.")
    print("   (Possibly: recordings already from similar positions)")

