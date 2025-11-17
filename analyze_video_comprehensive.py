"""
Comprehensive Analysis of Video File with Visual Outputs

Analyzes complete video/audio file and generates:
1. Stress timeline graph (acoustic + linguistic over time)
2. Topic segmentation visualization
3. Per-topic stress comparison
4. Acoustic feature heatmap
5. Conversation dynamics chart
6. Detailed text report

Designed for: 202511083.mp4 (or any long audio/video)
"""

import numpy as np
import wave
import subprocess
import os
from datetime import datetime, timedelta
import json

# Analysis components
from speaker_diarization_robust import ResemblyzerEmbeddings
from enhanced_acoustic_features import ComprehensiveAcousticAnalyzer
from linguistic_stress_analysis import LinguisticStressAnalyzer
from topic_modeling_analysis import TopicSegmentationSystem, TemporalStressAnalyzer
import whisper

# For visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def extract_audio_from_video(video_file, output_wav="temp_analysis_audio.wav"):
    """Extract audio from video"""
    print(f"Extracting audio from {video_file}...")
    cmd = f'ffmpeg -i "{video_file}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{output_wav}" -y -loglevel warning'
    
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=120)
        print(f"  [OK] Audio extracted to {output_wav}")
        return output_wav
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def load_audio(wav_file):
    """Load WAV file"""
    with wave.open(wav_file, 'rb') as wav:
        sr = wav.getframerate()
        n_frames = wav.getnframes()
        duration = n_frames / sr
        
        audio_bytes = wav.readframes(n_frames)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    return audio, sr, duration


def analyze_file_comprehensive(audio_file_or_video):
    """
    Perform comprehensive analysis on audio/video file
    
    Returns all analysis results + generates visualizations
    """
    print("="*90)
    print(" "*20 + "COMPREHENSIVE FILE ANALYSIS")
    print("="*90)
    print()
    
    # Extract audio if video
    if audio_file_or_video.endswith('.mp4'):
        audio_file = extract_audio_from_video(audio_file_or_video)
        if not audio_file:
            return None
    else:
        audio_file = audio_file_or_video
        
    # Load audio
    print(f"\nLoading audio...")
    audio_data, sample_rate, duration = load_audio(audio_file)
    
    # Initialize analyzers
    print("\nInitializing analysis components...")
    whisper_model = whisper.load_model("base")
    acoustic_analyzer = ComprehensiveAcousticAnalyzer()
    linguistic_analyzer = LinguisticStressAnalyzer()
    topic_system = TopicSegmentationSystem(similarity_threshold=0.65)
    temporal_analyzer = TemporalStressAnalyzer()
    
    # Process in chunks
    print("\nProcessing audio in chunks...")
    chunk_duration = 10  # 10-second chunks
    chunk_samples = chunk_duration * 16000
    
    results = {
        'file': audio_file_or_video,
        'duration_seconds': duration,
        'chunks_analyzed': 0,
        'timeline': [],
        'topics': {},
        'acoustic_features_timeline': [],
        'linguistic_features_timeline': []
    }
    
    num_chunks = int(duration / chunk_duration)
    print(f"  Total chunks to process: {num_chunks}")
    
    for chunk_idx in range(num_chunks):
        start_sample = chunk_idx * chunk_samples
        end_sample = min(start_sample + chunk_samples, len(audio_data))
        
        chunk = audio_data[start_sample:end_sample]
        
        if len(chunk) < 16000:  # Skip if less than 1 second
            continue
            
        # Check if speech present
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        
        if rms < 300:  # No speech
            continue
            
        print(f"\r  Processing chunk {chunk_idx+1}/{num_chunks} ({chunk_idx*chunk_duration}s)...", end='', flush=True)
        
        timestamp = timedelta(seconds=chunk_idx * chunk_duration)
        
        # === ACOUSTIC ANALYSIS ===
        try:
            acoustic_features = acoustic_analyzer.extract_all_features(chunk)
            acoustic_stress = acoustic_analyzer.assess_stress_from_acoustics(acoustic_features)
        except Exception as e:
            print(f"\n  Warning: Acoustic analysis failed for chunk {chunk_idx}: {e}")
            acoustic_features = {}
            acoustic_stress = {'acoustic_stress_probability': 0.0, 'acoustic_stress_category': 'UNKNOWN'}
            
        # === TRANSCRIPTION ===
        try:
            audio_float = chunk.astype(np.float32) / 32768.0
            transcription = whisper_model.transcribe(audio_float, language='en', fp16=False, verbose=False)
            text = transcription['text'].strip()
        except Exception as e:
            print(f"\n  Warning: Transcription failed for chunk {chunk_idx}: {e}")
            text = ""
            
        if not text:
            continue
            
        # === LINGUISTIC ANALYSIS ===
        try:
            linguistic_features = linguistic_analyzer.analyze_text(text)
        except Exception as e:
            print(f"\n  Warning: Linguistic analysis failed: {e}")
            linguistic_features = {'linguistic_stress_probability': 0.0}
            
        # === TOPIC ASSIGNMENT ===
        try:
            topic_assignment = topic_system.add_utterance(
                datetime.now() + timestamp,
                "speaker_0",
                "Speaker",
                text,
                acoustic_stress,
                linguistic_features
            )
        except Exception as e:
            print(f"\n  Warning: Topic assignment failed: {e}")
            topic_assignment = {'topic_id': 0, 'topic_label': 'General'}
            
        # === TEMPORAL TRACKING ===
        temporal_analyzer.add_measurement(
            datetime.now() + timestamp,
            acoustic_stress.get('acoustic_stress_probability', 0),
            linguistic_features.get('linguistic_stress_probability', 0)
        )
        
        # Store results
        result_entry = {
            'chunk_index': chunk_idx,
            'timestamp_seconds': chunk_idx * chunk_duration,
            'text': text,
            'acoustic_stress': acoustic_stress.get('acoustic_stress_probability', 0),
            'linguistic_stress': linguistic_features.get('linguistic_stress_probability', 0),
            'combined_stress': 0.6 * acoustic_stress.get('acoustic_stress_probability', 0) + 
                             0.4 * linguistic_features.get('linguistic_stress_probability', 0),
            'topic_id': topic_assignment['topic_id'],
            'topic_label': topic_assignment['topic_label'],
            'rms': int(rms),
            'key_acoustic': {
                'f0_mean': acoustic_features.get('f0_mean', 0),
                'jitter': acoustic_features.get('jitter_percent', 0),
                'shimmer': acoustic_features.get('shimmer_percent', 0)
            },
            'key_linguistic': {
                'uncertainty_ratio': linguistic_features.get('uncertainty_ratio', 0),
                'hedge_ratio': linguistic_features.get('hedge_ratio', 0),
                'word_count': linguistic_features.get('word_count', 0)
            }
        }
        
        results['timeline'].append(result_entry)
        results['chunks_analyzed'] += 1
        
    print(f"\n\n[OK] Analysis complete!")
    print(f"  Chunks analyzed: {results['chunks_analyzed']}")
    
    # Get topic summaries
    results['topics'] = topic_system.get_all_topics_summary()
    
    # Get change points
    change_points = temporal_analyzer.detect_change_points(min_change=0.15)
    results['change_points'] = change_points
    
    # Get trend
    results['overall_trend'] = temporal_analyzer.calculate_stress_trend()
    
    return results


def generate_visualizations(results, output_prefix="analysis"):
    """
    Generate comprehensive visual outputs
    """
    print("\n" + "="*90)
    print("GENERATING VISUALIZATIONS")
    print("="*90)
    
    if not results['timeline']:
        print("  No data to visualize")
        return
        
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract data
    times = [entry['timestamp_seconds']/60 for entry in results['timeline']]  # Convert to minutes
    acoustic_stress = [entry['acoustic_stress'] for entry in results['timeline']]
    linguistic_stress = [entry['linguistic_stress'] for entry in results['timeline']]
    combined_stress = [entry['combined_stress'] for entry in results['timeline']]
    topics = [entry['topic_id'] for entry in results['timeline']]
    
    # === PLOT 1: Stress Timeline ===
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, acoustic_stress, 'b-', label='Acoustic Stress', linewidth=2, alpha=0.7)
    ax1.plot(times, linguistic_stress, 'r-', label='Linguistic Stress', linewidth=2, alpha=0.7)
    ax1.plot(times, combined_stress, 'g-', label='Combined Stress', linewidth=3)
    
    # Add change points
    if results.get('change_points'):
        for cp in results['change_points']:
            ax1.axvline(cp['time_elapsed_minutes'], color='orange', linestyle='--', alpha=0.5)
            ax1.text(cp['time_elapsed_minutes'], 0.9, 'Change', rotation=90, va='bottom')
            
    ax1.axhline(0.6, color='red', linestyle=':', alpha=0.3, label='HIGH Stress')
    ax1.axhline(0.35, color='orange', linestyle=':', alpha=0.3, label='MODERATE Stress')
    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Stress Probability', fontsize=12)
    ax1.set_title('Stress Timeline Over Session', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # === PLOT 2: Topic Segmentation ===
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Color-code by topic
    unique_topics = list(set(topics))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_topics)))
    topic_colors = {topic_id: colors[i] for i, topic_id in enumerate(unique_topics)}
    
    for i, topic_id in enumerate(topics):
        ax2.scatter(times[i], 0.5, c=[topic_colors[topic_id]], s=100, alpha=0.6)
        
    # Add topic labels
    topic_labels = {}
    for entry in results['timeline']:
        if entry['topic_id'] not in topic_labels:
            topic_labels[entry['topic_id']] = entry['topic_label']
            
    # Legend
    patches = [mpatches.Patch(color=topic_colors[tid], label=f"{tid}: {label}") 
              for tid, label in topic_labels.items()]
    ax2.legend(handles=patches, loc='upper right', fontsize=9)
    
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_title('Topic Segmentation', fontsize=14, fontweight='bold')
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')
    
    # === PLOT 3: Per-Topic Stress ===
    ax3 = fig.add_subplot(gs[1, 1])
    
    if results.get('topics'):
        topic_names = [t['topic_label'] for t in results['topics']]
        topic_stress_acoustic = [t.get('acoustic_stress_mean', 0) for t in results['topics']]
        topic_stress_linguistic = [t.get('linguistic_stress_mean', 0) for t in results['topics']]
        
        x = np.arange(len(topic_names))
        width = 0.35
        
        ax3.bar(x - width/2, topic_stress_acoustic, width, label='Acoustic', color='blue', alpha=0.7)
        ax3.bar(x + width/2, topic_stress_linguistic, width, label='Linguistic', color='red', alpha=0.7)
        
        ax3.set_xlabel('Topic', fontsize=12)
        ax3.set_ylabel('Average Stress', fontsize=12)
        ax3.set_title('Stress by Topic', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(topic_names, rotation=45, ha='right')
        ax3.legend()
        ax3.axhline(0.6, color='red', linestyle=':', alpha=0.3)
        ax3.axhline(0.35, color='orange', linestyle=':', alpha=0.3)
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, axis='y')
        
    # === PLOT 4: Acoustic Features Heatmap ===
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Extract key acoustic features
    f0_values = [entry['key_acoustic']['f0_mean'] for entry in results['timeline']]
    jitter_values = [entry['key_acoustic']['jitter'] for entry in results['timeline']]
    shimmer_values = [entry['key_acoustic']['shimmer'] for entry in results['timeline']]
    
    # Normalize to 0-1 for visualization
    f0_norm = np.array(f0_values) / (max(f0_values) + 1e-10) if f0_values else np.zeros(1)
    jitter_norm = np.array(jitter_values) / (max(jitter_values) + 1e-10) if jitter_values else np.zeros(1)
    shimmer_norm = np.array(shimmer_values) / (max(shimmer_values) + 1e-10) if shimmer_values else np.zeros(1)
    
    # Stack features
    feature_matrix = np.vstack([f0_norm, jitter_norm, shimmer_norm])
    
    im = ax4.imshow(feature_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['F0 (Pitch)', 'Jitter', 'Shimmer'])
    ax4.set_xlabel('Time Chunks', fontsize=12)
    ax4.set_title('Acoustic Features Over Time', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Normalized Intensity')
    
    # === PLOT 5: Linguistic Features ===
    ax5 = fig.add_subplot(gs[2, 1])
    
    uncertainty = [entry['key_linguistic']['uncertainty_ratio'] for entry in results['timeline']]
    hedges = [entry['key_linguistic']['hedge_ratio'] for entry in results['timeline']]
    words = [entry['key_linguistic']['word_count'] for entry in results['timeline']]
    
    ax5_twin = ax5.twinx()
    
    ax5.plot(times, uncertainty, 'purple', label='Uncertainty', linewidth=2)
    ax5.plot(times, hedges, 'brown', label='Hedging', linewidth=2)
    ax5_twin.plot(times, words, 'gray', label='Word Count', linewidth=1, alpha=0.5)
    
    ax5.set_xlabel('Time (minutes)', fontsize=12)
    ax5.set_ylabel('Linguistic Ratios', fontsize=12)
    ax5_twin.set_ylabel('Word Count', fontsize=12, color='gray')
    ax5.set_title('Linguistic Features Over Time', fontsize=14, fontweight='bold')
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle(f'Comprehensive Analysis: {os.path.basename(results["file"])}',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_file = f"{output_prefix}_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n\n  [OK] Visualization saved: {output_file}")
    
    # Show
    plt.show(block=False)
    
    return results


def generate_text_report(results, output_file="analysis_report.txt"):
    """Generate detailed text report"""
    print(f"\nGenerating text report...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write(" "*25 + "COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("="*90 + "\n\n")
        
        f.write(f"File: {results['file']}\n")
        f.write(f"Duration: {results['duration_seconds']:.1f}s ({results['duration_seconds']/60:.1f} minutes)\n")
        f.write(f"Chunks Analyzed: {results['chunks_analyzed']}\n")
        f.write(f"Overall Trend: {results.get('overall_trend', 0):.4f} (stress per minute)\n")
        f.write("\n")
        
        # === TOPIC SUMMARY ===
        f.write("="*90 + "\n")
        f.write("TOPIC ANALYSIS\n")
        f.write("="*90 + "\n\n")
        
        if results.get('topics'):
            for topic in results['topics']:
                f.write(f"Topic: {topic['topic_label']} (ID: {topic['topic_id']})\n")
                f.write(f"  Utterances: {topic['utterance_count']}\n")
                f.write(f"  Duration: {topic['total_duration_seconds']/60:.1f} minutes\n")
                f.write(f"  First mention: {topic['first_mention']}\n")
                f.write(f"  Last mention: {topic['last_mention']}\n")
                f.write(f"  Acoustic Stress: {topic.get('acoustic_stress_mean', 0):.2f} "
                       f"({topic.get('topic_stress_category', 'N/A')})\n")
                f.write(f"  Linguistic Stress: {topic.get('linguistic_stress_mean', 0):.2f}\n")
                f.write(f"  Stress Trend: {topic.get('stress_progression', 'N/A')}\n")
                f.write("\n")
        else:
            f.write("  No topics detected\n\n")
            
        # === STRESS CHANGE POINTS ===
        f.write("="*90 + "\n")
        f.write("STRESS CHANGE POINTS\n")
        f.write("="*90 + "\n\n")
        
        if results.get('change_points'):
            for cp in results['change_points']:
                f.write(f"Time: {cp['time_elapsed_minutes']:.1f} minutes\n")
                f.write(f"  Type: {cp['change_type']}\n")
                f.write(f"  Magnitude: {cp['change_magnitude']:+.2f}\n")
                f.write(f"  Before: {cp['stress_before']:.2f}\n")
                f.write(f"  After: {cp['stress_after']:.2f}\n")
                f.write("\n")
        else:
            f.write("  No significant change points detected\n\n")
            
        # === DETAILED TIMELINE ===
        f.write("="*90 + "\n")
        f.write("DETAILED TIMELINE\n")
        f.write("="*90 + "\n\n")
        
        for entry in results['timeline']:
            minutes = entry['timestamp_seconds'] // 60
            seconds = entry['timestamp_seconds'] % 60
            
            f.write(f"[{int(minutes):02d}:{int(seconds):02d}] Topic: {entry['topic_label']}\n")
            f.write(f"  Text: {entry['text']}\n")
            f.write(f"  Acoustic Stress: {entry['acoustic_stress']:.2f}\n")
            f.write(f"  Linguistic Stress: {entry['linguistic_stress']:.2f}\n")
            f.write(f"  Combined: {entry['combined_stress']:.2f}\n")
            
            # Stress category
            if entry['combined_stress'] >= 0.60:
                f.write(f"  ⚠️ HIGH STRESS\n")
            elif entry['combined_stress'] >= 0.35:
                f.write(f"  ⚠️ MODERATE STRESS\n")
                
            f.write("\n")
            
    print(f"  [OK] Text report saved: {output_file}")
    
    return output_file


def save_json_results(results, output_file="analysis_results.json"):
    """Save results as JSON"""
    print(f"\nSaving JSON results...")
    
    # Make serializable
    serializable = {
        'file': results['file'],
        'duration_seconds': results['duration_seconds'],
        'chunks_analyzed': results['chunks_analyzed'],
        'overall_trend': float(results.get('overall_trend', 0)),
        'timeline': results['timeline'],
        'topics': results.get('topics', []),
        'change_points': results.get('change_points', [])
    }
    
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
        
    print(f"  [OK] JSON saved: {output_file}")
    
    return output_file


if __name__ == "__main__":
    # Analyze the video
    video_file = "Brad Pitt_ Between Two Ferns with Zach Galifianakis.mp4"
    
    if not os.path.exists(video_file):
        print(f"Error: {video_file} not found")
        exit(1)
        
    # Run comprehensive analysis
    results = analyze_file_comprehensive(video_file)
    
    if results:
        # Generate visualizations
        generate_visualizations(results, output_prefix="brad_pitt_analysis")
        
        # Generate text report
        generate_text_report(results, output_file="brad_pitt_report.txt")
        
        # Save JSON
        save_json_results(results, output_file="brad_pitt_results.json")
        
        print("\n" + "="*90)
        print("ANALYSIS COMPLETE!")
        print("="*90)
        print(f"\nGenerated files:")
        print(f"  1. brad_pitt_analysis_visualization.png (graphs)")
        print(f"  2. brad_pitt_report.txt (detailed text report)")
        print(f"  3. brad_pitt_results.json (machine-readable)")
        print("\n  Open the PNG file to see visual analysis!")
        print("="*90)
        
        # Keep plot window open
        plt.show()

