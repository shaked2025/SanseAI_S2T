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

from llm_based_topic_analysis import LLMBasedTopicAnalysis
from liwc_based_analysis import ComprehensiveLIWC
from enhanced_acoustic_features import ComprehensiveAcousticAnalyzer


def extract_audio(audio_file):
    """Extract/load audio"""
    print(f"Loading {audio_file}...", flush=True)
    
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
    print("="*120, flush=True)
    print(" "*40 + "FORMATTED TOPIC ANALYSIS REPORT", flush=True)
    print("="*120, flush=True)
    print(flush=True)
    
    # Extract audio
    audio, duration = extract_audio(video_file)
    
    # Initialize
    print("Loading models...", flush=True)
    whisper_model = whisper.load_model("base")
    topic_modeler = LLMBasedTopicAnalysis(use_llm=True, llm_model='llama2')  # LLM-based approach
    liwc_analyzer = ComprehensiveLIWC()
    acoustic_analyzer = ComprehensiveAcousticAnalyzer()
    
    print("[OK] Models loaded\n", flush=True)
    
    # Process chunks
    print(f"Processing {int(duration/10)} chunks...\n", flush=True)
    
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
        
    print(f"[OK] Processed {len(utterances)} utterances\n", flush=True)
    
    # Topic analysis (LLM-based approach)
    print("Performing LLM-based topic analysis...\n", flush=True)
    topic_analysis = topic_modeler.analyze_conversation(utterances)
    
    # Add embeddings to utterances for centrality calculation (if not already present)
    if not any('embedding' in u for u in utterances):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        all_embeddings = model.encode([u['text'] for u in utterances], show_progress_bar=False)
        
        for i, utt in enumerate(utterances):
            utt['embedding'] = all_embeddings[i]
        
    # === FORMAT OUTPUT ===
    print("\n" + "="*120, flush=True)
    print(" "*45 + "CONVERSATION ANALYSIS REPORT", flush=True)
    print("="*120, flush=True)
    print(f"\nFile: {video_file}", flush=True)
    print(f"Duration: {duration/60:.1f} minutes", flush=True)
    print(f"Utterances: {len(utterances)}", flush=True)
    print(f"Topics Identified: {len(topic_analysis.get('topics', []))}", flush=True)
    print(f"Questions Asked: {len(topic_analysis.get('questions', []))}", flush=True)
    print(flush=True)
    
    # Show all questions
    questions = topic_analysis.get('questions', [])
    if questions:
        print("\n" + "="*120, flush=True)
        print(" "*50 + "ALL QUESTIONS ASKED", flush=True)
        print("="*120, flush=True)
        for idx, q in enumerate(questions, 1):
            print(f"\nQ{idx} [{q.get('timestamp', '')}] ({q.get('speaker', 'Unknown')}):", flush=True)
            print(f"  {q['text']}", flush=True)
        print(flush=True)
    
    # Sort topics by importance (number of mentions)
    topics_sorted = sorted(topic_analysis.get('topics', []), key=lambda t: t.get('mention_count', 0), reverse=True)
    
    for topic_rank, topic in enumerate(topics_sorted, 1):
        print("\n" + "="*120, flush=True)
        print(f"TOPIC #{topic_rank}: {topic['label']}", flush=True)
        print("="*120, flush=True)
        
        print(f"  Questions in this topic: {len(topic.get('questions', []))}", flush=True)
        print(f"  Total Mentions: {topic['mention_count']}", flush=True)
        
        if 'total_span_minutes' in topic:
            print(f"  Time Span: {topic['total_span_minutes']:.1f} minutes", flush=True)
            if 'first_mention' in topic:
                print(f"  First mentioned: {topic['first_mention'][11:19] if len(topic['first_mention']) > 19 else topic['first_mention']}", flush=True)
            if 'last_mention' in topic:
                print(f"  Last mentioned: {topic['last_mention'][11:19] if len(topic['last_mention']) > 19 else topic['last_mention']}", flush=True)
            
        if topic.get('is_revisited'):
            print(f"  [REVISITED] TOPIC REVISITED: {topic['period_count']} discussion periods", flush=True)
            gaps = topic.get('revisit_gaps_minutes', [])
            if gaps:
                print(f"  Gaps between discussions: {[f'{g:.1f}min' for g in gaps]}", flush=True)
        
        # Show questions for this topic
        if topic.get('questions'):
            print(f"\n  QUESTIONS IN THIS TOPIC:", flush=True)
            print("  " + "-"*116, flush=True)
            for q_idx, q_text in enumerate(topic['questions'], 1):
                print(f"  Q{q_idx}: {q_text}", flush=True)
            print("  " + "-"*116, flush=True)
        
        # Show natural language summary
        if topic.get('summary'):
            print(f"\n  NATURAL LANGUAGE SUMMARY (What Happened in This Topic):", flush=True)
            print("  " + "-"*116, flush=True)
            # Print summary as natural text (not line by line if it's a paragraph)
            summary_text = topic['summary'].strip()
            # If it's a multi-line summary, print each line
            if '\n' in summary_text:
                summary_lines = summary_text.split('\n')
                for line in summary_lines:
                    if line.strip():
                        print(f"  {line}", flush=True)
            else:
                # Single paragraph - wrap it nicely
                print(f"  {summary_text}", flush=True)
            print("  " + "-"*116, flush=True)
        
        # Show all utterances in this topic (full transcription)
        if topic.get('all_utterances'):
            print(f"\n  FULL TRANSCRIPTION FOR THIS TOPIC ({len(topic['all_utterances'])} utterances):", flush=True)
            print("  " + "-"*116, flush=True)
            for i, utt in enumerate(topic['all_utterances'][:15], 1):  # Show first 15
                timestamp = utt.get('timestamp_str', '')
                speaker = utt.get('speaker_role', 'Speaker')
                text = utt.get('text', '').strip()
                if text:
                    # Fix Unicode encoding
                    try:
                        text_safe = text.encode('ascii', 'replace').decode('ascii')
                    except:
                        text_safe = ''.join(char if ord(char) < 128 else '?' for char in text)
                    print(f"  [{timestamp}] {speaker}: {text_safe[:200]}{'...' if len(text_safe) > 200 else ''}", flush=True)
            if len(topic['all_utterances']) > 15:
                print(f"  ... and {len(topic['all_utterances']) - 15} more utterances", flush=True)
            print("  " + "-"*116, flush=True)
                
        # Show key answers for this topic
        if topic.get('answers'):
            print(f"\n  KEY ANSWERS IN THIS TOPIC:", flush=True)
            print("  " + "-"*116, flush=True)
            # Show top 5 most informative answers (longest ones)
            answers_sorted = sorted(topic['answers'], key=len, reverse=True)
            for a_idx, answer in enumerate(answers_sorted[:5], 1):
                if len(answer) > 20:  # Only meaningful answers
                    # Fix Unicode encoding
                    try:
                        answer_safe = answer.encode('ascii', 'replace').decode('ascii')
                    except:
                        answer_safe = ''.join(char if ord(char) < 128 else '?' for char in answer)
                    print(f"  Answer {a_idx}: {answer_safe[:200]}{'...' if len(answer_safe) > 200 else ''}", flush=True)
            print("  " + "-"*116, flush=True)
                
    print("\n" + "="*120, flush=True)
    print(" "*50 + "END OF REPORT", flush=True)
    print("="*120, flush=True)
    
    return topic_analysis, utterances


if __name__ == "__main__":
    import sys
    
    # Get video file from command line argument or use default
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        video_file = "Kavin Interview77 (1).wav"
    
    analyze_and_format(video_file)

