"""
Compare two stress analysis systems:
1. Current system (enhanced_acoustic_features.py) - Rule-based with 60+ features
2. Audio_lib system - LSTM-based with embeddings and audio features

Runs both on the same audio file and generates side-by-side comparison graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import wave
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import current system
from enhanced_acoustic_features import ComprehensiveAcousticAnalyzer

# Import audio_lib system
try:
    from audio_lib.predictor import RealTimeStressPredictor
    from audio_lib.audio_model import AudioStressModel
    AUDIO_LIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: audio_lib not available: {e}")
    AUDIO_LIB_AVAILABLE = False


def load_audio_file(file_path):
    """Load audio file and return numpy array"""
    print(f"Loading audio file: {file_path}")
    
    if file_path.endswith('.wav'):
        with wave.open(file_path, 'rb') as wav:
            sr = wav.getframerate()
            n_channels = wav.getnchannels()
            duration = wav.getnframes() / sr
            
            audio_bytes = wav.readframes(wav.getnframes())
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
                
            if sr != 16000:
                from scipy import signal
                num_samples = int(len(audio) * 16000 / sr)
                audio = signal.resample(audio, num_samples).astype(np.int16)
                sr = 16000
                
        return audio, sr, duration
    else:
        # Try librosa for other formats
        import librosa
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        audio_int16 = (audio * 32768).astype(np.int16)
        duration = len(audio) / sr
        return audio_int16, sr, duration


def analyze_with_current_system(audio_int16, sample_rate, chunk_duration=2.5):
    """
    Analyze using current rule-based system (enhanced_acoustic_features)
    """
    print("\n=== Analyzing with Current System (Rule-Based) ===")
    
    analyzer = ComprehensiveAcousticAnalyzer(sample_rate=sample_rate)
    
    chunk_samples = int(chunk_duration * sample_rate)
    results = []
    timestamps = []
    
    num_chunks = len(audio_int16) // chunk_samples
    
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = audio_int16[start:end]
        
        # Convert to float32
        chunk_float = chunk.astype(np.float32) / 32768.0
        
        # Extract features
        try:
            features = analyzer.extract_all_features(chunk)
            stress_result = analyzer.assess_stress_from_acoustics(features)
            
            stress_prob = stress_result['acoustic_stress_probability']
            timestamp = i * chunk_duration
            
            results.append(stress_prob)
            timestamps.append(timestamp)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_chunks} chunks...")
                
        except Exception as e:
            print(f"  Error processing chunk {i}: {e}")
            results.append(np.nan)
            timestamps.append(i * chunk_duration)
    
    print(f"Current system: Processed {len(results)} chunks")
    return np.array(timestamps), np.array(results)


def analyze_with_audio_lib(audio_int16, sample_rate, chunk_duration=2.0):
    """
    Analyze using audio_lib LSTM-based system
    Uses AudioStressModel.run() which is the main interface
    """
    print("\n=== Analyzing with Audio_Lib System (LSTM-Based) ===")
    
    if not AUDIO_LIB_AVAILABLE:
        print("ERROR: audio_lib not available")
        return None, None
    
    try:
        # Use AudioStressModel - the main interface
        model = AudioStressModel()
        print("  Loading models...")
        model.load_models()
        
        # Check if in demo mode
        model_info = model.get_model_info()
        if model_info.get('demo_mode', False):
            print("  WARNING: Running in demo mode (model files not found)")
            print("  Demo mode will generate predictions based on audio energy")
        else:
            print("  Models loaded successfully")
        
    except Exception as e:
        print(f"  ERROR: Could not load audio_lib system: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # Process audio in chunks using real-time method (accepts int16 directly)
    chunk_samples = int(chunk_duration * sample_rate)
    results = []
    timestamps = []
    
    num_chunks = len(audio_int16) // chunk_samples
    
    # Process in batches to use real-time method more efficiently
    batch_size = 10  # Process 10 chunks at a time
    chunk_idx = 0
    
    for batch_start in range(0, num_chunks, batch_size):
        batch_end = min(batch_start + batch_size, num_chunks)
        batch_chunks = []
        batch_timestamps = []
        
        for i in range(batch_start, batch_end):
            start = i * chunk_samples
            end = start + chunk_samples
            chunk_int16 = audio_int16[start:end].copy()  # Make a copy to ensure contiguous
            batch_chunks.append(chunk_int16)
            batch_timestamps.append(i * chunk_duration)
        
        try:
            # Try real-time method first (accepts int16 directly)
            # This processes chunks with proper buffering and smoothing
            batch_audio = np.concatenate(batch_chunks)
            batch_results = model.run_real_time(batch_audio)
            
            # Extract results from batch
            for j, result in enumerate(batch_results):
                if 'error' in result:
                    print(f"  Error in batch chunk {batch_start + j}: {result['error']}")
                    results.append(np.nan)
                    timestamps.append(batch_timestamps[j])
                    continue
                
                # Extract stress probability
                # Priority: smoothed_probability > raw_probability > confidence
                stress_prob = result.get('smoothed_probability', np.nan)
                if np.isnan(stress_prob):
                    stress_prob = result.get('raw_probability', np.nan)
                if np.isnan(stress_prob):
                    stress_prob = result.get('confidence', np.nan)
                
                # If still NaN/None, check if filter passed
                if (np.isnan(stress_prob) or stress_prob is None) and result.get('filter_passed', False):
                    stress_prob = 0.0
                elif np.isnan(stress_prob) or stress_prob is None:
                    stress_prob = np.nan
                
                # Ensure it's a float
                if stress_prob is not None:
                    stress_prob = float(stress_prob)
                
                results.append(stress_prob)
                timestamps.append(batch_timestamps[j])
                
        except Exception as e:
            print(f"  Error processing batch {batch_start}-{batch_end}: {e}")
            # Fallback: process individually with run() method
            for j, chunk_int16 in enumerate(batch_chunks):
                try:
                    # Convert to float32
                    chunk_float = chunk_int16.astype(np.float32) / 32768.0
                    chunk_float = np.ascontiguousarray(chunk_float)
                    
                    # Use run() method
                    result = model.run(chunk_float, normalize=False, return_details=False)
                    
                    # Extract stress probability
                    stress_prob = result.get('raw_probability', np.nan)
                    if np.isnan(stress_prob) or stress_prob is None:
                        stress_prob = result.get('confidence', np.nan)
                    if np.isnan(stress_prob) or stress_prob is None:
                        stress_prob = 0.0 if result.get('filter_passed', False) else np.nan
                    
                    if stress_prob is not None:
                        stress_prob = float(stress_prob)
                    
                    results.append(stress_prob)
                    timestamps.append(batch_timestamps[j])
                except Exception as e2:
                    print(f"  Error processing individual chunk {batch_start + j}: {e2}")
                    results.append(np.nan)
                    timestamps.append(batch_timestamps[j])
        
        if (batch_end) % (batch_size * 10) == 0 or batch_end == num_chunks:
            print(f"  Processed {batch_end}/{num_chunks} chunks...")
    
    print(f"Audio_lib system: Processed {len(results)} chunks")
    
    # Print some debug info
    valid_results = [r for r in results if not np.isnan(r)]
    if len(valid_results) > 0:
        print(f"  Valid predictions: {len(valid_results)}/{len(results)}")
        print(f"  Sample predictions: {valid_results[:5]}")
    else:
        print(f"  WARNING: No valid predictions! All NaN.")
        print(f"  First result dict: {result if 'result' in locals() else 'N/A'}")
    
    return np.array(timestamps), np.array(results)


def create_comparison_graph(times1, probs1, times2, probs2, audio_file, output_file):
    """
    Create side-by-side comparison graph
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Graph 1: Current System (Rule-Based)
    ax1.plot(times1, probs1, 'b-', linewidth=2, label='Stress Probability')
    ax1.fill_between(times1, 0, probs1, alpha=0.3, color='blue')
    ax1.axhline(y=0.35, color='green', linestyle='--', linewidth=1, label='LOW threshold')
    ax1.axhline(y=0.60, color='orange', linestyle='--', linewidth=1, label='MODERATE threshold')
    ax1.axhline(y=0.60, color='red', linestyle='--', linewidth=1, label='HIGH threshold')
    ax1.set_ylabel('Stress Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Current System (Rule-Based: 60+ Acoustic Features)', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Graph 2: Audio_Lib System (LSTM-Based)
    if times2 is not None and probs2 is not None:
        ax2.plot(times2, probs2, 'r-', linewidth=2, label='Stress Probability')
        ax2.fill_between(times2, 0, probs2, alpha=0.3, color='red')
        ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Decision threshold (0.5)')
        ax2.set_ylabel('Stress Probability', fontsize=12, fontweight='bold')
        ax2.set_title('Audio_Lib System (LSTM-Based: Embeddings + Audio Features)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1.0])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'Audio_Lib System Not Available', 
                ha='center', va='center', fontsize=16, color='red',
                transform=ax2.transAxes)
        ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Stress Analysis Comparison: {os.path.basename(audio_file)}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison graph saved to: {output_file}")
    plt.close()


def main():
    # Get audio file from command line or use default
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Try to find a test file
        test_files = [
            "Kavin Interview77 (1).wav",
            "Kavin Interview77 (1).mp4",
            "test_audio.wav"
        ]
        audio_file = None
        for f in test_files:
            if os.path.exists(f):
                audio_file = f
                break
        
        if not audio_file:
            print("ERROR: No audio file specified and no test file found")
            print("Usage: python compare_stress_analysis.py <audio_file>")
            return
    
    if not os.path.exists(audio_file):
        print(f"ERROR: Audio file not found: {audio_file}")
        return
    
    print("="*80)
    print("STRESS ANALYSIS COMPARISON")
    print("="*80)
    print(f"Audio file: {audio_file}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load audio
    audio_int16, sample_rate, duration = load_audio_file(audio_file)
    print(f"Audio loaded: {len(audio_int16)} samples, {sample_rate} Hz, {duration:.2f} seconds")
    
    # Analyze with current system
    times1, probs1 = analyze_with_current_system(audio_int16, sample_rate, chunk_duration=2.5)
    
    # Analyze with audio_lib system
    times2, probs2 = analyze_with_audio_lib(audio_int16, sample_rate, chunk_duration=2.0)
    
    # Create comparison graph
    output_file = f"stress_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    create_comparison_graph(times1, probs1, times2, probs2, audio_file, output_file)
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    print("\nCurrent System (Rule-Based):")
    valid1 = probs1[~np.isnan(probs1)]
    if len(valid1) > 0:
        print(f"  Mean stress: {np.mean(valid1):.3f}")
        print(f"  Std stress: {np.std(valid1):.3f}")
        print(f"  Min stress: {np.min(valid1):.3f}")
        print(f"  Max stress: {np.max(valid1):.3f}")
        print(f"  Valid predictions: {len(valid1)}/{len(probs1)}")
    
    if times2 is not None and probs2 is not None:
        print("\nAudio_Lib System (LSTM-Based):")
        valid2 = probs2[~np.isnan(probs2)]
        if len(valid2) > 0:
            print(f"  Mean stress: {np.mean(valid2):.3f}")
            print(f"  Std stress: {np.std(valid2):.3f}")
            print(f"  Min stress: {np.min(valid2):.3f}")
            print(f"  Max stress: {np.max(valid2):.3f}")
            print(f"  Valid predictions: {len(valid2)}/{len(probs2)}")
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print(f"Graph saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()

