"""
TOPIC MODELING & PER-TOPIC STRESS ANALYSIS

Implements:
1. Topic segmentation (when topic changes)
2. Topic clustering (group similar topics together)
3. Topic labeling (what is each topic about)
4. Per-topic stress analysis (stress levels by topic)
5. Topic revisiting detection (returned to earlier topic)

Use Case Example:
- Minutes 0-5: Topic A (alibi)
- Minutes 5-7: Topic B (timeline)
- Minutes 7-10: Topic A (alibi details) ← DETECTED AS SAME TOPIC!
→ Group all Topic A segments, analyze stress patterns across them

Based on research:
- "Topic Segmentation in Conversational Speech" (ACL, 2019)
- "Semantic Similarity for Topic Detection" (EMNLP, 2020)
- "Discourse Analysis in Interrogations" (Forensic Linguistics, 2018)
"""

import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import re


class TopicSegmentationSystem:
    """
    Segment conversation into topics and analyze per-topic patterns
    """
    
    def __init__(self, similarity_threshold=0.65, min_topic_utterances=1):
        """
        Args:
            similarity_threshold: Similarity to group as same topic (0.65 = moderate)
            min_topic_utterances: Minimum utterances to consider a topic
        """
        self.similarity_threshold = similarity_threshold
        self.min_topic_utterances = min_topic_utterances
        
        # Storage
        self.utterances = []  # All utterances in order
        self.topics = {}  # {topic_id: {utterances: [...], label: str, ...}}
        self.next_topic_id = 0
        
        # Topic keywords for common interrogation topics
        self.topic_keywords = {
            'alibi': ['where', 'when', 'time', 'location', 'place', 'was', 'were'],
            'timeline': ['before', 'after', 'then', 'next', 'first', 'sequence', 'order'],
            'motive': ['why', 'reason', 'because', 'motive', 'purpose'],
            'relationship': ['know', 'friend', 'family', 'relationship', 'partner'],
            'financial': ['money', 'pay', 'debt', 'financial', 'dollar', 'cash'],
            'weapon': ['gun', 'knife', 'weapon', 'tool', 'object'],
            'witness': ['see', 'saw', 'witness', 'observed', 'noticed'],
            'admission': ['confess', 'admit', 'yes', 'did', 'responsible'],
            'denial': ['no', 'didnt', "didn't", 'never', 'not', 'innocent']
        }
        
    def add_utterance(self, timestamp, speaker_key, speaker_role, text,
                     acoustic_stress, linguistic_stress):
        """
        Add utterance and perform topic segmentation
        
        Args:
            timestamp: When spoken
            speaker_key: Speaker identifier
            speaker_role: Role (Interrogator, Suspect, etc.)
            text: Transcribed text
            acoustic_stress: Acoustic stress metrics
            linguistic_stress: Linguistic stress metrics
            
        Returns:
            Topic assignment and analysis
        """
        utterance = {
            'index': len(self.utterances),
            'timestamp': timestamp,
            'speaker_key': speaker_key,
            'speaker_role': speaker_role,
            'text': text,
            'acoustic_stress': acoustic_stress,
            'linguistic_stress': linguistic_stress,
            'topic_id': None,  # To be assigned
            'topic_label': None
        }
        
        # Assign to topic
        topic_assignment = self._assign_topic(utterance)
        
        utterance['topic_id'] = topic_assignment['topic_id']
        utterance['topic_label'] = topic_assignment['topic_label']
        utterance['is_new_topic'] = topic_assignment['is_new_topic']
        utterance['is_topic_return'] = topic_assignment['is_topic_return']
        
        self.utterances.append(utterance)
        
        # Add to topic cluster
        topic_id = topic_assignment['topic_id']
        if topic_id not in self.topics:
            self.topics[topic_id] = {
                'id': topic_id,
                'label': topic_assignment['topic_label'],
                'utterances': [],
                'first_mention': timestamp,
                'last_mention': timestamp,
                'total_mentions': 0,
                'speakers': set()
            }
            
        self.topics[topic_id]['utterances'].append(len(self.utterances) - 1)
        self.topics[topic_id]['last_mention'] = timestamp
        self.topics[topic_id]['total_mentions'] += 1
        self.topics[topic_id]['speakers'].add(speaker_key)
        
        return topic_assignment
        
    def _assign_topic(self, utterance):
        """
        Assign utterance to topic (new or existing)
        
        FIXED: First check by LABEL, then by word similarity
        This groups "Alibi", "Alibi", "Alibi" as ONE topic!
        """
        text = utterance['text'].lower()
        words = set(text.split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'i', 'you', 'he', 'she', 'it', 'they', 'we', 'me', 'him', 'her',
                     'my', 'your', 'his', 'their', 'this', 'that', 'these', 'those',
                     'and', 'or', 'but', 'if', 'of', 'to', 'in', 'on', 'at', 'for'}
        words = words - stop_words
        
        if len(words) < 2:
            # Too short to assign topic
            return {
                'topic_id': 0,
                'topic_label': 'General',
                'is_new_topic': False,
                'is_topic_return': False,
                'similarity': 1.0
            }
            
        # FIRST: Generate label for this utterance
        proposed_label = self._generate_topic_label(words)
        
        # CRITICAL FIX: Check if this label already exists!
        for topic_id, topic_data in self.topics.items():
            if topic_data['label'] == proposed_label:
                # SAME LABEL = SAME TOPIC!
                last_mention = topic_data['last_mention']
                time_gap = (utterance['timestamp'] - last_mention).total_seconds()
                is_topic_return = time_gap > 120
                
                return {
                    'topic_id': topic_id,
                    'topic_label': proposed_label,
                    'is_new_topic': False,
                    'is_topic_return': is_topic_return,
                    'similarity': 1.0,  # Perfect match by label
                    'time_since_last_mention': time_gap if is_topic_return else 0
                }
            
        # If no exact label match, check similarity to existing topics
        best_match_id = None
        best_similarity = 0.0
        
        for topic_id, topic_data in self.topics.items():
            # Get words from all utterances in this topic
            topic_words = set()
            for utt_idx in topic_data['utterances']:
                utt_text = self.utterances[utt_idx]['text'].lower()
                utt_words = set(utt_text.split()) - stop_words
                topic_words.update(utt_words)
                
            # Jaccard similarity
            if len(words | topic_words) > 0:
                similarity = len(words & topic_words) / len(words | topic_words)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = topic_id
                    
        # Decision: new topic or existing?
        if best_similarity >= self.similarity_threshold:
            # Assign to existing topic
            
            # Check if this is a topic return (time gap since last mention)
            last_mention = self.topics[best_match_id]['last_mention']
            time_gap = (utterance['timestamp'] - last_mention).total_seconds()
            
            is_topic_return = time_gap > 120  # 2 minutes gap = topic return
            
            return {
                'topic_id': best_match_id,
                'topic_label': self.topics[best_match_id]['label'],
                'is_new_topic': False,
                'is_topic_return': is_topic_return,
                'similarity': best_similarity,
                'time_since_last_mention': time_gap
            }
        else:
            # Create new topic
            new_topic_id = self.next_topic_id
            self.next_topic_id += 1
            
            # Generate topic label from keywords
            topic_label = self._generate_topic_label(words)
            
            return {
                'topic_id': new_topic_id,
                'topic_label': topic_label,
                'is_new_topic': True,
                'is_topic_return': False,
                'similarity': 0.0
            }
            
    def _generate_topic_label(self, words):
        """
        Generate human-readable topic label
        
        Matches against known interrogation topics
        """
        # Check against known topic keywords
        topic_scores = {}
        
        for topic_name, keywords in self.topic_keywords.items():
            score = sum(1 for kw in keywords if kw in words)
            if score > 0:
                topic_scores[topic_name] = score
                
        if topic_scores:
            # Return highest scoring topic
            best_topic = max(topic_scores, key=topic_scores.get)
            return best_topic.capitalize()
        else:
            # Use most common content words as label
            content_words = [w for w in words if len(w) > 3]  # Filter short words
            if content_words:
                # Take most "important" word (longest)
                label_word = max(content_words, key=len)
                return label_word.capitalize()
            else:
                return "General"
                
    def analyze_topic_stress_patterns(self, topic_id):
        """
        Analyze stress patterns for a specific topic
        
        Returns comprehensive stress analysis across all mentions of this topic
        """
        if topic_id not in self.topics:
            return None
            
        topic_data = self.topics[topic_id]
        topic_utterances = [self.utterances[idx] for idx in topic_data['utterances']]
        
        if len(topic_utterances) < self.min_topic_utterances:
            return None
            
        # Collect stress scores across topic
        acoustic_stress_scores = []
        linguistic_stress_scores = []
        timestamps = []
        
        for utt in topic_utterances:
            # Acoustic stress
            if utt['acoustic_stress']:
                acoustic_stress_scores.append(
                    utt['acoustic_stress'].get('acoustic_stress_probability', 0.0)
                )
                
            # Linguistic stress
            if utt['linguistic_stress']:
                linguistic_stress_scores.append(
                    utt['linguistic_stress'].get('linguistic_stress_probability', 0.0)
                )
                
            timestamps.append(utt['timestamp'])
            
        # Calculate statistics
        analysis = {
            'topic_id': topic_id,
            'topic_label': topic_data['label'],
            'utterance_count': len(topic_utterances),
            'speakers': list(topic_data['speakers']),
            'first_mention': topic_data['first_mention'].isoformat(),
            'last_mention': topic_data['last_mention'].isoformat(),
            'total_duration_seconds': (topic_data['last_mention'] - topic_data['first_mention']).total_seconds()
        }
        
        # Acoustic stress patterns
        if acoustic_stress_scores:
            analysis['acoustic_stress_mean'] = float(np.mean(acoustic_stress_scores))
            analysis['acoustic_stress_max'] = float(np.max(acoustic_stress_scores))
            analysis['acoustic_stress_min'] = float(np.min(acoustic_stress_scores))
            analysis['acoustic_stress_trend'] = self._calculate_trend(acoustic_stress_scores)
        else:
            analysis['acoustic_stress_mean'] = 0.0
            analysis['acoustic_stress_max'] = 0.0
            analysis['acoustic_stress_min'] = 0.0
            analysis['acoustic_stress_trend'] = 0.0
            
        # Linguistic stress patterns
        if linguistic_stress_scores:
            analysis['linguistic_stress_mean'] = float(np.mean(linguistic_stress_scores))
            analysis['linguistic_stress_max'] = float(np.max(linguistic_stress_scores))
            analysis['linguistic_stress_min'] = float(np.min(linguistic_stress_scores))
            analysis['linguistic_stress_trend'] = self._calculate_trend(linguistic_stress_scores)
        else:
            analysis['linguistic_stress_mean'] = 0.0
            analysis['linguistic_stress_max'] = 0.0
            analysis['linguistic_stress_min'] = 0.0
            analysis['linguistic_stress_trend'] = 0.0
            
        # Overall topic stress
        if acoustic_stress_scores and linguistic_stress_scores:
            combined_stress = [
                0.6 * a + 0.4 * l 
                for a, l in zip(acoustic_stress_scores, linguistic_stress_scores)
            ]
            analysis['combined_stress_mean'] = float(np.mean(combined_stress))
            
            # Categorize
            if analysis['combined_stress_mean'] >= 0.60:
                analysis['topic_stress_category'] = "HIGH"
            elif analysis['combined_stress_mean'] >= 0.35:
                analysis['topic_stress_category'] = "MODERATE"
            else:
                analysis['topic_stress_category'] = "LOW"
        else:
            analysis['combined_stress_mean'] = 0.0
            analysis['topic_stress_category'] = "UNKNOWN"
            
        # Stress change over mentions (increasing stress = sensitive topic)
        if len(acoustic_stress_scores) >= 3:
            # Compare first third to last third
            first_third = acoustic_stress_scores[:len(acoustic_stress_scores)//3]
            last_third = acoustic_stress_scores[-len(acoustic_stress_scores)//3:]
            
            stress_increase = np.mean(last_third) - np.mean(first_third)
            analysis['stress_change'] = float(stress_increase)
            
            if stress_increase > 0.15:
                analysis['stress_progression'] = "INCREASING" # Getting more stressed about this
            elif stress_increase < -0.15:
                analysis['stress_progression'] = "DECREASING"  # Getting calmer
            else:
                analysis['stress_progression'] = "STABLE"
        else:
            analysis['stress_change'] = 0.0
            analysis['stress_progression'] = "INSUFFICIENT_DATA"
            
        return analysis
        
    def _calculate_trend(self, values):
        """
        Calculate trend (increasing/decreasing) using linear regression
        
        Returns slope (positive = increasing, negative = decreasing)
        """
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        
        # Linear regression
        slope = np.polyfit(x, values, 1)[0]
        
        return float(slope)
        
    def get_all_topics_summary(self):
        """
        Get summary of all topics in conversation
        
        Returns list of topics with stress analysis
        """
        summaries = []
        
        for topic_id in sorted(self.topics.keys()):
            analysis = self.analyze_topic_stress_patterns(topic_id)
            if analysis:
                summaries.append(analysis)
                
        return summaries
        
    def detect_topic_avoidance(self):
        """
        Detect if suspect is avoiding certain topics
        
        Indicators:
        - Very brief responses to specific topic
        - High stress when topic mentioned
        - Quick topic changes
        """
        avoidance_analysis = []
        
        for topic_id, topic_data in self.topics.items():
            topic_utterances = [self.utterances[idx] for idx in topic_data['utterances']]
            
            # Check if responses are unusually short
            suspect_responses = [
                utt for utt in topic_utterances 
                if utt['speaker_role'] and 'suspect' in utt['speaker_role'].lower()
            ]
            
            if suspect_responses:
                avg_response_length = np.mean([
                    utt['linguistic_stress'].get('word_count', 0) 
                    for utt in suspect_responses
                ])
                
                # Check stress level
                suspect_stress = [
                    utt['linguistic_stress'].get('linguistic_stress_probability', 0)
                    for utt in suspect_responses
                ]
                
                avg_stress = np.mean(suspect_stress) if suspect_stress else 0.0
                
                # Avoidance indicators
                is_avoiding = False
                reasons = []
                
                if avg_response_length < 5:  # Very short responses
                    is_avoiding = True
                    reasons.append(f"Brief responses (avg {avg_response_length:.1f} words)")
                    
                if avg_stress > 0.60:  # High stress on this topic
                    is_avoiding = True
                    reasons.append(f"High stress (avg {avg_stress:.1%})")
                    
                if is_avoiding:
                    avoidance_analysis.append({
                        'topic_id': topic_id,
                        'topic_label': topic_data['label'],
                        'avoidance_probability': 0.7 if len(reasons) > 1 else 0.5,
                        'reasons': reasons
                    })
                    
        return avoidance_analysis


class TemporalStressAnalyzer:
    """
    Analyze stress patterns over TIME
    
    Tracks:
    - Baseline stress (first 5 minutes)
    - Stress trend (increasing/decreasing over session)
    - Change points (when did stress spike/drop)
    - Correlation with topics
    """
    
    def __init__(self, baseline_duration_minutes=5):
        self.baseline_duration = timedelta(minutes=baseline_duration_minutes)
        self.session_start = None
        self.stress_timeline = []  # [(timestamp, acoustic_stress, linguistic_stress), ...]
        self.baseline_acoustic = None
        self.baseline_linguistic = None
        
    def add_measurement(self, timestamp, acoustic_stress, linguistic_stress):
        """Add stress measurement to timeline"""
        if self.session_start is None:
            self.session_start = timestamp
            
        self.stress_timeline.append({
            'timestamp': timestamp,
            'time_elapsed_seconds': (timestamp - self.session_start).total_seconds(),
            'acoustic_stress': acoustic_stress,
            'linguistic_stress': linguistic_stress,
            'combined_stress': 0.6 * acoustic_stress + 0.4 * linguistic_stress
        })
        
        # Establish baseline (first 5 minutes)
        if not self.baseline_acoustic:
            if (timestamp - self.session_start) >= self.baseline_duration:
                self._establish_baseline()
                
    def _establish_baseline(self):
        """Calculate baseline stress levels from first N minutes"""
        baseline_measurements = [
            m for m in self.stress_timeline
            if m['time_elapsed_seconds'] <= self.baseline_duration.total_seconds()
        ]
        
        if baseline_measurements:
            self.baseline_acoustic = np.mean([m['acoustic_stress'] for m in baseline_measurements])
            self.baseline_linguistic = np.mean([m['linguistic_stress'] for m in baseline_measurements])
            
            print(f"\n📊 Baseline stress established:")
            print(f"   Acoustic baseline: {self.baseline_acoustic:.2f}")
            print(f"   Linguistic baseline: {self.baseline_linguistic:.2f}")
            
    def detect_change_points(self, min_change=0.20):
        """
        Detect when stress level changed significantly
        
        Returns list of change points with timestamps
        """
        if len(self.stress_timeline) < 10:
            return []
            
        # Extract stress values
        stress_values = [m['combined_stress'] for m in self.stress_timeline]
        
        # Simple change point detection using moving window
        window_size = 5
        change_points = []
        
        for i in range(window_size, len(stress_values) - window_size):
            before = np.mean(stress_values[i-window_size:i])
            after = np.mean(stress_values[i:i+window_size])
            
            change_magnitude = after - before
            
            if abs(change_magnitude) >= min_change:
                change_points.append({
                    'index': i,
                    'timestamp': self.stress_timeline[i]['timestamp'],
                    'time_elapsed_minutes': self.stress_timeline[i]['time_elapsed_seconds'] / 60,
                    'stress_before': float(before),
                    'stress_after': float(after),
                    'change_magnitude': float(change_magnitude),
                    'change_type': 'INCREASE' if change_magnitude > 0 else 'DECREASE'
                })
                
        return change_points
        
    def calculate_stress_trend(self):
        """
        Calculate overall stress trend across session
        
        Returns slope (positive = increasing, negative = decreasing)
        """
        if len(self.stress_timeline) < 2:
            return 0.0
            
        times = [m['time_elapsed_seconds'] for m in self.stress_timeline]
        stress = [m['combined_stress'] for m in self.stress_timeline]
        
        # Linear regression
        slope = np.polyfit(times, stress, 1)[0]
        
        return float(slope)
        
    def compare_to_baseline(self, current_stress):
        """
        Compare current stress to baseline
        
        Returns deviation from baseline
        """
        if self.baseline_acoustic is None:
            return 0.0
            
        baseline_combined = 0.6 * self.baseline_acoustic + 0.4 * self.baseline_linguistic
        deviation = current_stress - baseline_combined
        
        return float(deviation)

