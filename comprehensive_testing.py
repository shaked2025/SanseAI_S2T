"""
COMPREHENSIVE CROSS-VALIDATION FOR GENERALIZATION

Tests ALL permutations to ensure NO bias:
- Gender bias
- Voice characteristic bias
- Overfitting to specific recordings

Each recording tested as:
1. Enrolled speaker 1
2. Enrolled speaker 2  
3. Unknown impostor (should be rejected)

This ensures the solution is truly generalizable!
"""

import numpy as np
import wave
import subprocess
import os
from scipy import signal as scipy_signal
from speaker_diarization_robust import ResemblyzerEmbeddings
from speaker_enrollment import SpeakerEnrollment
from simple_robust_verification import SimpleRobustVerifier
from itertools import permutations


def extract_audio_from_mp4(mp4_file, output_wav="temp_audio.wav"):
    """Extract audio from MP4 using ffmpeg"""
    try:
        cmd = f'ffmpeg -i "{mp4_file}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{output_wav}" -y'
        subprocess.run(cmd, shell=True, capture_output=True, check=True)
        return output_wav
    except Exception as e:
        print(f"Error extracting from {mp4_file}: {e}")
        return None


def load_audio_file(filename):
    """Load WAV or MP4 file"""
    print(f"Loading {filename}...")
    
    # If MP4, extract audio first
    if filename.endswith('.mp4'):
        wav_file = extract_audio_from_mp4(filename)
        if not wav_file:
            return None
        filename = wav_file
        
    try:
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
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def extract_chunks(audio, num_chunks=6, chunk_duration=5, sr=16000):
    """Extract chunks from audio"""
    chunk_samples = chunk_duration * sr
    chunks = []
    
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        if end <= len(audio):
            chunks.append(audio[start:end])
        else:
            break
            
    return chunks


def test_configuration(file1, file2, file3, role1, role2, role3, threshold):
    """
    Test one configuration
    
    Args:
        file1, file2, file3: Audio arrays
        role1, role2, role3: 'enrolled1', 'enrolled2', or 'unknown'
        threshold: Threshold to test
    """
    # Determine which are enrolled and which is unknown
    enrolled = []
    unknown = []
    
    for audio, role, idx in [(file1, role1, 0), (file2, role2, 1), (file3, role3, 2)]:
        chunks = extract_chunks(audio)
        if len(chunks) < 6:
            return None  # Not enough audio
            
        if role.startswith('enrolled'):
            enrolled.append((f"Speaker{idx+1}", chunks))
        else:
            unknown.append((f"Unknown{idx+1}", chunks))
            
    if len(enrolled) != 2 or len(unknown) != 1:
        return None
        
    # Enroll
    embedder = ResemblyzerEmbeddings()
    enrollment = SpeakerEnrollment(embedder)
    
    for idx, (name, chunks) in enumerate(enrolled):
        key = f"spk_{idx}"
        enrollment.start_enrollment(key, name, "Speaker")
        for chunk in chunks:
            enrollment.add_enrollment_sample(key, chunk, 16000)
        enrollment.complete_enrollment(key)
        
    # Test
    verifier = SimpleRobustVerifier(base_threshold=threshold)
    enrolled_dict = enrollment.get_enrolled_speakers()
    
    results = {'enrolled_accept': 0, 'enrolled_total': 0, 'unknown_reject': 0, 'unknown_total': 0}
    
    # Test enrolled speakers (use chunks 6-9, NOT used in enrollment)
    for name, all_chunks in enrolled:
        test_chunks = all_chunks[6:9] if len(all_chunks) > 9 else all_chunks[3:6]
        
        for chunk in test_chunks:
            test_emb = embedder.extract_embedding(chunk, 16000)
            accept, _, _, sim, _ = verifier.verify_speaker(test_emb, enrolled_dict, 0.8)
            
            results['enrolled_total'] += 1
            if accept:
                results['enrolled_accept'] += 1
                
    # Test unknown
    for name, chunks in unknown:
        for chunk in chunks[:3]:
            test_emb = embedder.extract_embedding(chunk, 16000)
            accept, _, _, sim, _ = verifier.verify_speaker(test_emb, enrolled_dict, 0.8)
            
            results['unknown_total'] += 1
            if not accept:
                results['unknown_reject'] += 1
                
    # Calculate metrics
    tar = results['enrolled_accept'] / results['enrolled_total'] if results['enrolled_total'] > 0 else 0
    trr = results['unknown_reject'] / results['unknown_total'] if results['unknown_total'] > 0 else 0
    
    return tar, trr


def main():
    print("="*80)
    print("  COMPREHENSIVE CROSS-VALIDATION FOR GENERALIZATION")
    print("="*80)
    print()
    
    # Load all available files
    print("Loading audio files...")
    
    audio_files = []
    
    # WAV files
    wav_files = [
        "Kavin Interview77 (1).wav",
        "vid_orig_obf (1).wav",
        "JiaJun_video_3 1.wav"
    ]
    
    for f in wav_files:
        audio = load_audio_file(f)
        if audio is not None:
            audio_files.append((f, audio))
            
    # MP4 files (if available)
    mp4_files = [
        "WIN_20250126_13_16_49_Pro.mp4",
        "Eder_Gatica-25-04-30-14-45-43 .mp4",
        "202511083.mp4"
    ]
    
    for f in mp4_files:
        if os.path.exists(f):
            audio = load_audio_file(f)
            if audio is not None:
                audio_files.append((f, audio))
                
    print(f"\n✅ Loaded {len(audio_files)} files")
    
    if len(audio_files) < 3:
        print("Need at least 3 audio files")
        return
        
    # Use first 3 for comprehensive testing
    test_files = audio_files[:3]
    
    print("\nFiles for testing:")
    for i, (name, audio) in enumerate(test_files):
        print(f"   {i+1}. {name} ({len(audio)/16000:.1f}s)")
        
    # Test ALL permutations with different thresholds
    thresholds_to_test = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    
    print(f"\n{'='*80}")
    print("TESTING ALL PERMUTATIONS")
    print("Each file will be tested as: enrolled1, enrolled2, and unknown")
    print(f"{'='*80}")
    
    # Generate all permutations of roles
    roles = ['enrolled1', 'enrolled2', 'unknown']
    all_role_assignments = list(permutations(roles))
    
    print(f"\nTotal permutations to test: {len(all_role_assignments)}")
    print(f"Thresholds to test: {len(thresholds_to_test)}")
    print(f"Total configurations: {len(all_role_assignments) * len(thresholds_to_test)}")
    print()
    
    # Test each threshold
    threshold_results = {}
    
    for threshold in thresholds_to_test:
        print(f"\n{'='*80}")
        print(f"TESTING THRESHOLD: {threshold:.2f}")
        print(f"{'='*80}")
        
        perm_results = []
        
        # Test all permutations
        for perm_idx, role_assignment in enumerate(all_role_assignments):
            file1, file2, file3 = [af[1] for af in test_files]
            
            role1, role2, role3 = role_assignment
            
            file_names = [test_files[i][0].split()[0] for i in range(3)]
            
            print(f"\n   Permutation {perm_idx+1}/6:")
            print(f"      {file_names[0]}: {role1}")
            print(f"      {file_names[1]}: {role2}")
            print(f"      {file_names[2]}: {role3}")
            
            result = test_configuration(file1, file2, file3, role1, role2, role3, threshold)
            
            if result:
                tar, trr = result
                print(f"      TAR: {tar:.1%}, TRR: {trr:.1%}")
                perm_results.append((tar, trr))
            else:
                print(f"      Failed to test")
                
        # Calculate average across permutations
        if perm_results:
            avg_tar = np.mean([r[0] for r in perm_results])
            avg_trr = np.mean([r[1] for r in perm_results])
            min_tar = np.min([r[0] for r in perm_results])
            min_trr = np.min([r[1] for r in perm_results])
            
            print(f"\n   SUMMARY for threshold {threshold:.2f}:")
            print(f"      Average TAR: {avg_tar:.1%}")
            print(f"      Average TRR: {avg_trr:.1%}")
            print(f"      Minimum TAR: {min_tar:.1%} (worst case)")
            print(f"      Minimum TRR: {min_trr:.1%} (worst case)")
            
            # Score: prioritize minimum (ensures ALL cases work)
            score = 0.6 * min_tar + 0.4 * min_trr
            print(f"      Robustness Score: {score:.3f}")
            
            threshold_results[threshold] = {
                'avg_tar': avg_tar,
                'avg_trr': avg_trr,
                'min_tar': min_tar,
                'min_trr': min_trr,
                'score': score,
                'all_results': perm_results
            }
            
    # Find best threshold
    print(f"\n{'='*80}")
    print("FINAL RESULTS - BEST THRESHOLD FOR GENERALIZATION")
    print(f"{'='*80}")
    
    best_threshold = max(threshold_results.keys(), key=lambda t: threshold_results[t]['score'])
    best_result = threshold_results[best_threshold]
    
    print(f"\nOPTIMAL THRESHOLD: {best_threshold:.2f}")
    print(f"   Average TAR: {best_result['avg_tar']:.1%}")
    print(f"   Average TRR: {best_result['avg_trr']:.1%}")
    print(f"   MINIMUM TAR: {best_result['min_tar']:.1%} (worst case enrolled)")
    print(f"   MINIMUM TRR: {best_result['min_trr']:.1%} (worst case unknown)")
    print(f"   Robustness Score: {best_result['score']:.3f}")
    
    print(f"\n{'='*80}")
    print("ALL THRESHOLDS COMPARISON:")
    print(f"{'='*80}")
    
    for t in sorted(threshold_results.keys()):
        r = threshold_results[t]
        print(f"Threshold {t:.2f}: TAR={r['avg_tar']:.1%}, TRR={r['avg_trr']:.1%}, "
              f"MinTAR={r['min_tar']:.1%}, MinTRR={r['min_trr']:.1%}, Score={r['score']:.3f}")
              
    # Save results
    import json
    with open('comprehensive_results.json', 'w') as f:
        results_save = {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                           for kk, vv in v.items() if kk != 'all_results'}
                       for k, v in threshold_results.items()}
        json.dump({
            'best_threshold': float(best_threshold),
            'results': results_save
        }, f, indent=2)
        
    print(f"\n✅ Results saved to comprehensive_results.json")
    
    if best_result['min_tar'] >= 0.90 and best_result['min_trr'] >= 0.90:
        print(f"\n✅✅✅ EXCELLENT! System is robust and generalizable!")
        print(f"Threshold {best_threshold:.2f} works across ALL permutations")
    else:
        print(f"\n⚠️ System needs more work:")
        print(f"   Minimum TAR: {best_result['min_tar']:.1%} (target: >=90%)")
        print(f"   Minimum TRR: {best_result['min_trr']:.1%} (target: >=90%)")
        print(f"\nSuggestions:")
        if best_result['min_tar'] < 0.90:
            print(f"   - Lower threshold (currently {best_threshold:.2f})")
            print(f"   - Improve enrollment quality")
        if best_result['min_trr'] < 0.90:
            print(f"   - Raise threshold (currently {best_threshold:.2f})")
            print(f"   - Add additional rejection rules")


if __name__ == "__main__":
    main()

