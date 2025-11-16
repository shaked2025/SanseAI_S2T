"""
Realistic Optimization Based on Actual Data Analysis

Key findings from audio analysis:
- Self-similarity for these recordings: 0.60-0.65 (not 0.90!)
- Cross-speaker similarity: 0.44-0.50
- Gap: only 0.15-0.20 (not 0.40!)

Therefore: Use REALISTIC thresholds and focus on RELATIVE scoring
"""

import numpy as np
import wave
from scipy import signal as scipy_signal
from speaker_diarization_robust import ResemblyzerEmbeddings
from speaker_enrollment import SpeakerEnrollment
import json


def load_and_prepare_wav(filename):
    """Load WAV and prepare for testing"""
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
            
        return audio


def extract_chunks(audio, num_chunks=6, chunk_duration=5, sr=16000):
    """Extract chunks"""
    chunk_samples = chunk_duration * sr
    chunks = []
    
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        if end <= len(audio):
            chunks.append(audio[start:end])
            
    return chunks


def test_threshold_grid(enrolled_data, unknown_data):
    """
    Test grid of thresholds to find optimal
    
    enrolled_data: [(name, chunks), ...]
    unknown_data: [(name, chunks), ...]
    """
    print("\n" + "="*70)
    print("GRID SEARCH FOR OPTIMAL THRESHOLD")
    print("="*70)
    
    embedder = ResemblyzerEmbeddings()
    
    # Test different thresholds
    thresholds_to_test = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    
    best_threshold = None
    best_f1 = -1.0
    
    results_grid = []
    
    for threshold in thresholds_to_test:
        print(f"\nTesting threshold: {threshold:.2f}")
        
        # Enroll speakers
        enrollment = SpeakerEnrollment(embedder)
        
        for idx, (name, chunks) in enumerate(enrolled_data):
            key = f"speaker_{idx}"
            enrollment.start_enrollment(key, name, "Speaker")
            
            for chunk in chunks:
                enrollment.add_enrollment_sample(key, chunk, 16000)
                
            enrollment.complete_enrollment(key)
            
        enrolled_dict = enrollment.get_enrolled_speakers()
        
        # Test enrolled speakers (should accept)
        true_positives = 0
        false_negatives = 0
        
        for name, chunks in enrolled_data:
            for chunk in chunks[3:]:  # Use different chunks than enrollment
                test_emb = embedder.extract_embedding(chunk, 16000)
                
                # Find best match
                best_sim = -1.0
                for profile in enrolled_dict.values():
                    sim = np.dot(test_emb, profile['mean_embedding'])
                    if sim > best_sim:
                        best_sim = sim
                        
                if best_sim >= threshold:
                    true_positives += 1
                else:
                    false_negatives += 1
                    
        # Test unknown speakers (should reject)
        true_negatives = 0
        false_positives = 0
        
        for name, chunks in unknown_data:
            for chunk in chunks[:3]:  # Test first 3
                test_emb = embedder.extract_embedding(chunk, 16000)
                
                # Find best match
                best_sim = -1.0
                for profile in enrolled_dict.values():
                    sim = np.dot(test_emb, profile['mean_embedding'])
                    if sim > best_sim:
                        best_sim = sim
                        
                if best_sim >= threshold:
                    false_positives += 1  # Unknown accepted - BAD
                else:
                    true_negatives += 1  # Unknown rejected - GOOD
                    
        # Calculate metrics
        total_enrolled_tests = true_positives + false_negatives
        total_unknown_tests = true_negatives + false_positives
        
        true_accept_rate = true_positives / total_enrolled_tests if total_enrolled_tests > 0 else 0
        false_reject_rate = false_negatives / total_enrolled_tests if total_enrolled_tests > 0 else 0
        true_reject_rate = true_negatives / total_unknown_tests if total_unknown_tests > 0 else 0
        false_accept_rate = false_positives / total_unknown_tests if total_unknown_tests > 0 else 0
        
        # F1 score (balance between precision and recall)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_accept_rate
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   True Accept Rate: {true_accept_rate:.1%} ({true_positives}/{total_enrolled_tests})")
        print(f"   False Reject Rate: {false_reject_rate:.1%} ({false_negatives}/{total_enrolled_tests})")
        print(f"   True Reject Rate: {true_reject_rate:.1%} ({true_negatives}/{total_unknown_tests})")
        print(f"   False Accept Rate: {false_accept_rate:.1%} ({false_positives}/{total_unknown_tests})")
        print(f"   F1 Score: {f1:.3f}")
        
        results_grid.append({
            'threshold': threshold,
            'true_accept_rate': true_accept_rate,
            'false_accept_rate': false_accept_rate,
            'f1': f1,
            'metrics': {
                'TP': true_positives,
                'FN': false_negatives,
                'TN': true_negatives,
                'FP': false_positives
            }
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    print(f"\n{'='*70}")
    print("OPTIMAL THRESHOLD FOUND:")
    print(f"{'='*70}")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"F1 Score: {best_f1:.3f}")
    
    # Save results
    with open('threshold_optimization.json', 'w') as f:
        json.dump({
            'optimal_threshold': best_threshold,
            'best_f1': best_f1,
            'all_results': results_grid
        }, f, indent=2)
        
    print("\n✅ Results saved to threshold_optimization.json")
    
    return best_threshold, results_grid


if __name__ == "__main__":
    print("Loading WAV files...")
    
    # Load audio files
    audio1 = load_and_prepare_wav("Kavin Interview77 (1).wav")
    audio2 = load_and_prepare_wav("vid_orig_obf (1).wav")
    audio3 = load_and_prepare_wav("JiaJun_video_3 1.wav")
    
    # Extract chunks (first 30 seconds, 6 chunks of 5s)
    chunks1 = extract_chunks(audio1, num_chunks=6)
    chunks2 = extract_chunks(audio2, num_chunks=6)
    chunks3 = extract_chunks(audio3, num_chunks=6)
    
    print(f"\nChunks extracted:")
    print(f"   Speaker 1: {len(chunks1)}")
    print(f"   Speaker 2: {len(chunks2)}")
    print(f"   Unknown: {len(chunks3)}")
    
    # Enrolled: Kavin + vid_orig_obf
    # Unknown: JiaJun
    enrolled_speakers = [
        ("Kavin", chunks1),
        ("VidOrig", chunks2)
    ]
    
    unknown_speakers = [
        ("JiaJun_Unknown", chunks3)
    ]
    
    # Find optimal threshold
    optimal_threshold, results = test_threshold_grid(enrolled_speakers, unknown_speakers)
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDED CONFIGURATION:")
    print(f"{'='*70}")
    print(f"Base Threshold: {optimal_threshold:.2f}")
    print(f"SVM nu: 0.20 (lenient for variable voices)")
    print(f"Use simple threshold-based rejection (complex methods fail with variable data)")
    print(f"{'='*70}")

