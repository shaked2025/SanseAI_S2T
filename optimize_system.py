"""
Systematic Hyperparameter Optimization
Uses provided WAV files to find optimal settings
"""

import numpy as np
import wave
import json
from speaker_diarization_robust import ResemblyzerEmbeddings
from speaker_enrollment import SpeakerEnrollment
from unknown_speaker_rejection import AdvancedSpeakerRejection, calculate_audio_quality, MultiMetricVerifier


def load_wav(filename):
    """Load WAV file"""
    print(f"Loading {filename}...")
    with wave.open(filename, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        n_frames = wav.getnframes()
        duration = n_frames / sample_rate
        
        # Read audio
        audio_bytes = wav.readframes(n_frames)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert stereo to mono if needed
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
        print(f"   Duration: {duration:.1f}s, Sample rate: {sample_rate} Hz")
        
        return audio, sample_rate, duration


def resample_audio(audio, orig_sr, target_sr=16000):
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
        return audio
        
    from scipy import signal
    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_samples)
    return resampled.astype(np.int16)


def extract_30s_chunks(audio, sample_rate, num_chunks=6, chunk_duration=5):
    """Extract 6 chunks of 5 seconds each from first 30 seconds"""
    chunk_samples = int(chunk_duration * sample_rate)
    chunks = []
    
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        
        if end <= len(audio):
            chunk = audio[start:end]
            chunks.append(chunk)
        else:
            # Pad if needed
            chunk = np.pad(audio[start:], (0, end - len(audio)))
            chunks.append(chunk)
            
    return chunks


def test_configuration(config, enrolled_speakers, test_samples):
    """
    Test a configuration and return metrics
    
    Args:
        config: Dict with hyperparameters
        enrolled_speakers: List of (name, chunks) for enrolled speakers
        test_samples: List of (name, chunks, is_enrolled) for testing
        
    Returns:
        Dict with performance metrics
    """
    print(f"\n{'='*70}")
    print(f"Testing Configuration:")
    print(f"   SVM nu: {config['nu']}")
    print(f"   Base threshold: {config['base_threshold']}")
    print(f"   Quality thresholds: {config['quality_thresholds']}")
    print(f"{'='*70}")
    
    # Initialize system
    embedder = ResemblyzerEmbeddings()
    enrollment = SpeakerEnrollment(embedder)
    
    # Enroll speakers
    print("\n📝 Enrolling speakers...")
    for speaker_idx, (name, chunks) in enumerate(enrolled_speakers):
        speaker_key = f"speaker_{speaker_idx}"
        enrollment.start_enrollment(speaker_key, name, "Speaker")
        
        for chunk_idx, chunk in enumerate(chunks):
            success, quality, msg = enrollment.add_enrollment_sample(speaker_key, chunk, 16000)
            print(f"   {name} sample {chunk_idx+1}: {msg}")
            
        success, quality, msg = enrollment.complete_enrollment(speaker_key)
        print(f"   {name}: {msg}")
        
    # Create rejector with config
    rejector = AdvancedSpeakerRejection(nu=config['nu'])
    rejector.base_threshold = config['base_threshold']
    rejector.fit_enrolled_speakers(enrollment)
    
    # Test samples
    print(f"\n🧪 Testing samples...")
    
    results = {
        'enrolled_correct': 0,
        'enrolled_total': 0,
        'unknown_rejected': 0,
        'unknown_total': 0,
        'false_accepts': 0,
        'false_rejects': 0,
        'details': []
    }
    
    for test_name, test_chunks, is_enrolled in test_samples:
        print(f"\n   Testing: {test_name} ({'ENROLLED' if is_enrolled else 'UNKNOWN'})")
        
        for chunk_idx, chunk in enumerate(test_chunks[:3]):  # Test first 3 chunks
            # Extract embedding
            test_emb = embedder.extract_embedding(chunk, 16000)
            
            # Calculate quality
            quality = calculate_audio_quality(chunk, 16000)
            
            # Get best match from basic verifier
            enrolled_dict = enrollment.get_enrolled_speakers()
            
            if not enrolled_dict:
                continue
                
            # Find best match
            best_key = None
            best_sim = -1.0
            
            for spk_key, profile in enrolled_dict.items():
                sim = np.dot(test_emb, profile['mean_embedding'])
                if sim > best_sim:
                    best_sim = sim
                    best_key = spk_key
                    
            if best_key is None:
                continue
                
            profile = enrolled_dict[best_key]
            
            # Test rejection
            accept, score, reason, details = rejector.verify_and_reject(
                test_emb,
                best_key,
                best_sim,
                profile,
                quality
            )
            
            # Record result
            if is_enrolled:
                results['enrolled_total'] += 1
                if accept:
                    results['enrolled_correct'] += 1
                    print(f"      Sample {chunk_idx+1}: ✅ ACCEPTED (score: {score:.3f})")
                else:
                    results['false_rejects'] += 1
                    print(f"      Sample {chunk_idx+1}: ❌ FALSE REJECT (score: {score:.3f}) - {reason}")
            else:
                results['unknown_total'] += 1
                if not accept:
                    results['unknown_rejected'] += 1
                    print(f"      Sample {chunk_idx+1}: ✅ REJECTED (score: {score:.3f}) - {reason}")
                else:
                    results['false_accepts'] += 1
                    print(f"      Sample {chunk_idx+1}: ❌ FALSE ACCEPT (score: {score:.3f})")
                    
            results['details'].append({
                'test_name': test_name,
                'is_enrolled': is_enrolled,
                'accepted': accept,
                'score': score,
                'reason': reason,
                'quality': quality
            })
            
    # Calculate metrics
    if results['enrolled_total'] > 0:
        results['true_accept_rate'] = results['enrolled_correct'] / results['enrolled_total']
        results['false_reject_rate'] = results['false_rejects'] / results['enrolled_total']
    else:
        results['true_accept_rate'] = 0.0
        results['false_reject_rate'] = 0.0
        
    if results['unknown_total'] > 0:
        results['true_reject_rate'] = results['unknown_rejected'] / results['unknown_total']
        results['false_accept_rate'] = results['false_accepts'] / results['unknown_total']
    else:
        results['true_reject_rate'] = 0.0
        results['false_accept_rate'] = 0.0
        
    # Overall accuracy
    total_correct = results['enrolled_correct'] + results['unknown_rejected']
    total_samples = results['enrolled_total'] + results['unknown_total']
    results['accuracy'] = total_correct / total_samples if total_samples > 0 else 0.0
    
    return results


def main():
    print("="*70)
    print("  HYPERPARAMETER OPTIMIZATION FOR SPEAKER REJECTION")
    print("="*70)
    print()
    
    # Load WAV files
    wav_files = [
        "Kavin Interview77 (1).wav",
        "vid_orig_obf (1).wav",
        "JiaJun_video_3 1.wav"
    ]
    
    print("Loading WAV files...")
    audio_files = []
    
    for filename in wav_files:
        try:
            audio, sr, duration = load_wav(filename)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                print(f"   Resampling from {sr} to 16000 Hz...")
                audio = resample_audio(audio, sr, 16000)
                sr = 16000
                
            audio_files.append((filename, audio, sr))
        except Exception as e:
            print(f"   Error loading {filename}: {e}")
            
    if len(audio_files) < 3:
        print("❌ Need 3 WAV files")
        return
        
    print(f"\n✅ Loaded {len(audio_files)} files")
    
    # Extract enrollment chunks (first 30 seconds, 6 chunks of 5s each)
    print("\nExtracting enrollment chunks...")
    
    speaker1_chunks = extract_30s_chunks(audio_files[0][1], 16000, num_chunks=6)
    speaker2_chunks = extract_30s_chunks(audio_files[1][1], 16000, num_chunks=6)
    unknown_chunks = extract_30s_chunks(audio_files[2][1], 16000, num_chunks=6)
    
    print(f"   Speaker 1 ({wav_files[0]}): {len(speaker1_chunks)} chunks")
    print(f"   Speaker 2 ({wav_files[1]}): {len(speaker2_chunks)} chunks")
    print(f"   Unknown ({wav_files[2]}): {len(unknown_chunks)} chunks")
    
    # Setup test data
    enrolled_speakers = [
        ("Speaker_1", speaker1_chunks),
        ("Speaker_2", speaker2_chunks)
    ]
    
    test_samples = [
        ("Speaker_1_test", speaker1_chunks, True),   # Should be accepted
        ("Speaker_2_test", speaker2_chunks, True),   # Should be accepted
        ("Unknown_impostor", unknown_chunks, False)  # Should be REJECTED
    ]
    
    # Test different configurations
    configurations = [
        # Conservative (low false accepts, may have false rejects)
        {
            'name': 'Conservative',
            'nu': 0.05,
            'base_threshold': 0.85,
            'quality_thresholds': {0.9: 0.80, 0.7: 0.85, 0.5: 0.88}
        },
        # Balanced
        {
            'name': 'Balanced',
            'nu': 0.10,
            'base_threshold': 0.80,
            'quality_thresholds': {0.9: 0.75, 0.7: 0.80, 0.5: 0.85}
        },
        # Lenient (high accepts, may have more false accepts)
        {
            'name': 'Lenient',
            'nu': 0.15,
            'base_threshold': 0.75,
            'quality_thresholds': {0.9: 0.70, 0.7: 0.75, 0.5: 0.80}
        },
    ]
    
    best_config = None
    best_score = -1.0
    all_results = []
    
    for config in configurations:
        results = test_configuration(config, enrolled_speakers, test_samples)
        
        print(f"\n{'='*70}")
        print(f"RESULTS FOR: {config['name']}")
        print(f"{'='*70}")
        print(f"   True Accept Rate (enrolled): {results['true_accept_rate']:.1%}")
        print(f"   False Reject Rate (enrolled): {results['false_reject_rate']:.1%}")
        print(f"   True Reject Rate (unknown): {results['true_reject_rate']:.1%}")
        print(f"   False Accept Rate (unknown): {results['false_accept_rate']:.1%}")
        print(f"   Overall Accuracy: {results['accuracy']:.1%}")
        
        # Score: prioritize low false accepts while maintaining good true accepts
        score = (
            0.40 * results['true_accept_rate'] +  # Want high
            0.40 * results['true_reject_rate'] +   # Want high
            0.10 * (1 - results['false_accept_rate']) +  # Want low
            0.10 * (1 - results['false_reject_rate'])    # Want low
        )
        
        print(f"   Combined Score: {score:.3f}")
        
        all_results.append({
            'config': config,
            'results': results,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_config = config
            
    # Display best configuration
    print(f"\n{'='*70}")
    print("BEST CONFIGURATION FOUND:")
    print(f"{'='*70}")
    print(f"Config: {best_config['name']}")
    print(f"Score: {best_score:.3f}")
    print(f"\nParameters:")
    print(f"   nu (SVM): {best_config['nu']}")
    print(f"   base_threshold: {best_config['base_threshold']}")
    print(f"\nSave this configuration? (y/n): ", end='')
    
    response = input().strip().lower()
    if response == 'y':
        with open('optimal_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)
        print("✅ Saved to optimal_config.json")
        
    # Save detailed results
    with open('optimization_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON
        results_serializable = []
        for r in all_results:
            r_copy = {
                'config': r['config'],
                'score': float(r['score']),
                'results': {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in r['results'].items()
                    if k != 'details'
                }
            }
            results_serializable.append(r_copy)
            
        json.dump(results_serializable, f, indent=2)
        
    print("✅ Detailed results saved to optimization_results.json")


if __name__ == "__main__":
    main()

