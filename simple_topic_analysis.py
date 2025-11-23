"""
Simple Topic Analysis - Just show all topics clearly
"""

import sys
import wave
import numpy as np
from datetime import datetime, timedelta
import whisper

# Import topic modeling
from proper_semantic_topics import ResearchGradeTopicModeling

def analyze_topics_simple(wav_file):
    """Simple topic analysis - just show all topics"""
    
    print("="*80)
    print("SIMPLE TOPIC ANALYSIS")
    print("="*80)
    print(f"\nLoading: {wav_file}")
    
    # Load audio
    with wave.open(wav_file, 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        audio = np.frombuffer(wf.readframes(frames), dtype=np.int16)
        duration = len(audio) / sample_rate
    
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Sample rate: {sample_rate} Hz")
    
    # Load models
    print("\nLoading models...")
    whisper_model = whisper.load_model("base")
    topic_modeler = ResearchGradeTopicModeling(min_topic_size=2)
    print("[OK] Models loaded\n")
    
    # Process in chunks
    print("Processing audio...")
    utterances = []
    chunk_duration = 10  # 10 seconds per chunk
    
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
        }
        
        utterances.append(utterance)
    
    print(f"[OK] Processed {len(utterances)} utterances\n")
    
    # Topic analysis
    print("Analyzing topics...\n")
    topic_analysis = topic_modeler.analyze_interrogation_topics(utterances)
    
    topics = topic_analysis.get('topics', [])
    
    print("="*80)
    print(f"FOUND {len(topics)} TOPICS")
    print("="*80)
    print()
    
    # Sort by mention count
    topics_sorted = sorted(topics, key=lambda t: t.get('mention_count', 0), reverse=True)
    
    # Show all topics
    for idx, topic in enumerate(topics_sorted, 1):
        label = topic.get('label', 'Unknown')
        mentions = topic.get('mention_count', 0)
        indices = topic.get('utterance_indices', [])
        
        print(f"TOPIC #{idx}: {label}")
        print(f"  Mentions: {mentions}")
        print(f"  Utterance indices: {indices[:10]}{'...' if len(indices) > 10 else ''}")
        
        # Show first few utterances
        if indices:
            print(f"  Sample utterances:")
            for i, utt_idx in enumerate(indices[:3]):
                if utt_idx < len(utterances):
                    utt = utterances[utt_idx]
                    text = utt.get('text', '')[:100]  # First 100 chars
                    # Safe encoding
                    try:
                        text_safe = text.encode('ascii', 'replace').decode('ascii')
                    except:
                        text_safe = ''.join(char if ord(char) < 128 else '?' for char in text)
                    print(f"    [{utt.get('timestamp_str', '?')}] {text_safe}")
        
        print()
    
    print("="*80)
    print("END")
    print("="*80)
    
    return topics, utterances


if __name__ == "__main__":
    wav_file = "Kavin Interview77 (1).wav"
    if len(sys.argv) > 1:
        wav_file = sys.argv[1]
    
    analyze_topics_simple(wav_file)

