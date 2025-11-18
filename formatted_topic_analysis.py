"""
Generate Beautifully Formatted Topic Analysis Report

For each topic:
- Topic name and statistics
- Top 10 most impactful utterances
- Timestamp for each
- Stress indicators
- Sorted by impact (stress + length + centrality)
"""

import wave
import subprocess
import numpy as np
from datetime import datetime, timedelta
import whisper

from proper_semantic_topics import ResearchGradeTopicModeling
from liwc_based_analysis import ComprehensiveLIWC
from enhanced_acoustic_features import ComprehensiveAcousticAnalyzer


def extract_audio(audio_file):
    """Extract/load audio"""
    print(f"Loading {audio_file}...")
    
    if audio_file.endswith('.wav'):
        wav_file = audio_file
    else:
        wav_file = "temp_analysis.wav"
        cmd = f'ffmpeg -i "{audio_file}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{wav_file}" -y -loglevel warning'
        subprocess.run(cmd, shell=True, check=True)
    
    with wave.open(wav_file, 'rb') as wav:
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
            
    return audio, duration


def calculate_impact_score(utterance, topic_centroid=None):
    """
    Calculate "impact" score for ranking utterances
    
    Factors:
    - Stress level (high stress = important)
    - Length (longer = more content)
    - Semantic centrality (if close to topic centroid = representative)
    """
    impact = 0.0
    
    # Stress contribution (40%)
    acoustic_stress = utterance.get('acoustic_stress_prob', 0)
    linguistic_stress = utterance.get('linguistic_stress_prob', 0)
    combined_stress = 0.6 * acoustic_stress + 0.4 * linguistic_stress
    
    impact += 0.40 * combined_stress
    
    # Length contribution (30%) - longer utterances have more content
    word_count = len(utterance['text'].split())
    length_score = min(1.0, word_count / 50)  # Normalize: 50 words = max
    
    impact += 0.30 * length_score
    
    # Semantic centrality (30%) - how representative of topic
    if topic_centroid is not None and 'embedding' in utterance:
        centrality = np.dot(utterance['embedding'], topic_centroid)
        impact += 0.30 * centrality
    else:
        impact += 0.15  # Neutral if no centrality
        
    return impact


def analyze_and_format(video_file):
    """
    Complete analysis with beautiful formatting
    """
    print("="*120)
    print(" "*40 + "FORMATTED TOPIC ANALYSIS REPORT")
    print("="*120)
    print()
    
    # Extract audio
    audio, duration = extract_audio(video_file)
    
    # Initialize
    print("Loading models...")
    whisper_model = whisper.load_model("base")
    topic_modeler = ResearchGradeTopicModeling(min_topic_size=2)
    liwc_analyzer = ComprehensiveLIWC()
    acoustic_analyzer = ComprehensiveAcousticAnalyzer()
    
    print("[OK] Models loaded\n")
    
    # Process chunks
    print(f"Processing {int(duration/10)} chunks...\n")
    
    utterances = []
    chunk_duration = 10
    
    for chunk_idx in range(int(duration / chunk_duration)):
        start_sample = chunk_idx * 16000 * chunk_duration
        end_sample = min(start_sample + 16000 * chunk_duration, len(audio))
        
        chunk = audio[start_sample:end_sample]
        
        if len(chunk) < 16000:
            continue
            
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        
        if rms < 300:
            continue
            
        # Transcribe
        audio_float = chunk.astype(np.float32) / 32768.0
        result = whisper_model.transcribe(audio_float, language='en', fp16=False, verbose=False)
        
        text = result['text'].strip()
        
        if not text:
            continue
            
        # Analyses
        try:
            acoustic_features = acoustic_analyzer.extract_all_features(chunk)
            acoustic_stress = acoustic_analyzer.assess_stress_from_acoustics(acoustic_features)
            acoustic_stress_prob = acoustic_stress.get('acoustic_stress_probability', 0)
        except:
            acoustic_stress_prob = 0.0
            
        try:
            liwc_results = liwc_analyzer.analyze_text_comprehensive(text)
            linguistic_stress_prob = liwc_results['composite_scores']['linguistic_stress']
        except:
            linguistic_stress_prob = 0.0
            
        # Calculate timestamp
        minutes = (chunk_idx * chunk_duration) // 60
        seconds = (chunk_idx * chunk_duration) % 60
        timestamp_str = f"{int(minutes):02d}:{int(seconds):02d}"
        
        utterance = {
            'index': len(utterances),
            'timestamp': datetime.now() + timedelta(seconds=chunk_idx * chunk_duration),
            'timestamp_str': timestamp_str,
            'text': text,
            'speaker_key': 'speaker_0',
            'speaker_role': 'Speaker',
            'acoustic_stress': {'acoustic_stress_probability': acoustic_stress_prob},
            'linguistic_stress': {'linguistic_stress_probability': linguistic_stress_prob},
            'acoustic_stress_prob': acoustic_stress_prob,
            'linguistic_stress_prob': linguistic_stress_prob,
            'combined_stress': 0.6 * acoustic_stress_prob + 0.4 * linguistic_stress_prob,
            'word_count': len(text.split())
        }
        
        utterances.append(utterance)
        
    print(f"[OK] Processed {len(utterances)} utterances\n")
    
    # Topic analysis
    print("Performing semantic topic analysis...\n")
    topic_analysis = topic_modeler.analyze_interrogation_topics(utterances)
    
    # Add embeddings to utterances for centrality calculation
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_embeddings = model.encode([u['text'] for u in utterances], show_progress_bar=False)
    
    for i, utt in enumerate(utterances):
        utt['embedding'] = all_embeddings[i]
        
    # === FORMAT OUTPUT ===
    print("\n" + "="*120)
    print(" "*45 + "TOPIC ANALYSIS REPORT")
    print("="*120)
    print(f"\nFile: {video_file}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Utterances: {len(utterances)}")
    print(f"Main Topics Identified: {len(topic_analysis['topics'])}")
    print()
    
    # Sort topics by importance (number of mentions)
    topics_sorted = sorted(topic_analysis['topics'], key=lambda t: t['mention_count'], reverse=True)
    
    for topic_rank, topic in enumerate(topics_sorted, 1):
        print("\n" + "="*120)
        print(f"TOPIC #{topic_rank}: {topic['label']}")
        print("="*120)
        
        print(f"  Total Mentions: {topic['mention_count']}")
        
        if 'total_span_minutes' in topic:
            print(f"  Time Span: {topic['total_span_minutes']:.1f} minutes")
            print(f"  First mentioned: {topic['first_mention'][11:19]}")  # HH:MM:SS
            print(f"  Last mentioned: {topic['last_mention'][11:19]}")
            
        if topic.get('is_revisited'):
            print(f"  🔄 TOPIC REVISITED: {topic['period_count']} discussion periods")
            gaps = topic.get('revisit_gaps_minutes', [])
            if gaps:
                print(f"  Gaps between discussions: {[f'{g:.1f}min' for g in gaps]}")
                
        # Get all utterances for this topic
        topic_utterances = [utterances[idx] for idx in topic['utterance_indices']]
        
        # Calculate topic centroid
        topic_embeddings = [u['embedding'] for u in topic_utterances]
        topic_centroid = np.mean(topic_embeddings, axis=0)
        
        # Calculate impact scores
        for utt in topic_utterances:
            utt['impact_score'] = calculate_impact_score(utt, topic_centroid)
            
        # Sort by impact
        topic_utterances_sorted = sorted(topic_utterances, key=lambda u: u['impact_score'], reverse=True)
        
        # Show top 10 (or all if less than 10)
        top_n = min(10, len(topic_utterances_sorted))
        
        print(f"\n  TOP {top_n} MOST IMPACTFUL UTTERANCES:")
        print("  " + "-"*116)
        
        for rank, utt in enumerate(topic_utterances_sorted[:top_n], 1):
            print(f"\n  #{rank} | Time: [{utt['timestamp_str']}] | Stress: {utt['combined_stress']:.2f} | Words: {utt['word_count']}")
            print(f"      {utt['text']}")
            
            # Stress breakdown
            if utt['combined_stress'] >= 0.60:
                stress_indicator = "⚠️ HIGH STRESS"
            elif utt['combined_stress'] >= 0.35:
                stress_indicator = "⚠️ MODERATE STRESS"
            else:
                stress_indicator = "✓ Low stress"
                
            print(f"      {stress_indicator} (Acoustic: {utt['acoustic_stress_prob']:.2f}, Linguistic: {utt['linguistic_stress_prob']:.2f})")
            
        print("\n  " + "-"*116)
        
        # Topic-level stress summary
        if topic['mention_count'] > 1:
            topic_stresses = [u['combined_stress'] for u in topic_utterances]
            avg_stress = np.mean(topic_stresses)
            max_stress = np.max(topic_stresses)
            
            print(f"\n  📊 TOPIC STRESS SUMMARY:")
            print(f"      Average stress: {avg_stress:.2f}")
            print(f"      Maximum stress: {max_stress:.2f}")
            
            if avg_stress >= 0.50:
                print(f"      ⚠️ HIGH STRESS on this topic overall!")
            elif avg_stress >= 0.35:
                print(f"      ⚠️ MODERATE stress on this topic")
            else:
                print(f"      ✓ Low stress on this topic")
                
    print("\n" + "="*120)
    print(" "*50 + "END OF REPORT")
    print("="*120)
    
    return topic_analysis, utterances


if __name__ == "__main__":
    # Try different video to verify
    video_file = "Kavin Interview77 (1).wav"
    
    analyze_and_format(video_file)

