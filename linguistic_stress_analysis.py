"""
LINGUISTIC & SEMANTIC STRESS ANALYSIS

Analyzes WHAT was said (content), not just HOW it was said (acoustics)

Key indicators from interrogation research:
1. Sentiment analysis (emotional tone)
2. Cognitive load (complexity, hesitations, corrections)
3. Linguistic deception markers (pronouns, details, temporal markers)
4. Emotional keywords (fear, anger, anxiety words)
5. Certainty/uncertainty language
6. Response coherence (direct vs evasive answers)

Based on research:
- "Linguistic Indicators of Deception" (Applied Cognitive Psychology, 2003)
- "Automated Deception Detection in Text" (ACL, 2020)
- "Cognitive Load in Interrogation" (Legal and Criminological Psychology, 2018)
- "Sentiment Analysis for Stress Detection" (IEEE, 2021)
"""

import re
from collections import Counter
import numpy as np


class LinguisticStressAnalyzer:
    """
    Analyze text content for stress/deception/emotion indicators
    """
    
    def __init__(self):
        # Emotional keyword dictionaries
        self._load_emotion_lexicons()
        
        # Linguistic patterns
        self._compile_linguistic_patterns()
        
    def _load_emotion_lexicons(self):
        """
        Load emotion/stress word lists
        
        In production: Would load from comprehensive lexicons (NRC, LIWC)
        For now: Core words for each category
        """
        self.stress_words = {
            'anxiety': ['worried', 'anxious', 'nervous', 'scared', 'afraid', 'fearful', 
                       'terrified', 'panic', 'stress', 'tense', 'uneasy'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'irritated', 
                     'annoyed', 'frustrated', 'upset'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'down', 
                       'heartbroken', 'grief', 'sorrow'],
            'uncertainty': ['maybe', 'perhaps', 'possibly', 'might', 'could', 
                          'probably', 'unsure', 'dont know', "don't know", 
                          'not sure', 'uncertain', 'guess'],
            'certainty': ['definitely', 'certainly', 'absolutely', 'surely', 
                         'positive', 'sure', 'confident', 'know', 'certain'],
            'deception_markers': ['honestly', 'truthfully', 'to be honest', 
                                 'believe me', 'trust me', 'swear', 'promise']
        }
        
    def _compile_linguistic_patterns(self):
        """Compile regex patterns for linguistic analysis"""
        self.patterns = {
            # Pronoun patterns (deception: avoid "I", use "we"/"they")
            'first_person': re.compile(r'\b(I|me|my|mine|myself)\b', re.IGNORECASE),
            'third_person': re.compile(r'\b(he|she|they|them|him|her)\b', re.IGNORECASE),
            'second_person': re.compile(r'\b(you|your|yours)\b', re.IGNORECASE),
            
            # Temporal markers (deception: vague vs specific)
            'specific_time': re.compile(r'\b(\d{1,2}:\d{2}|\d{1,2}\s*(am|pm|oclock))\b', re.IGNORECASE),
            'vague_time': re.compile(r'\b(sometime|around|about|approximately|roughly)\b', re.IGNORECASE),
            
            # Hedging (uncertainty, equivocation)
            'hedges': re.compile(r'\b(kind of|sort of|like|basically|actually|literally)\b', re.IGNORECASE),
            
            # Self-corrections (cognitive load indicator)
            'corrections': re.compile(r'\b(I mean|actually|correction|sorry|wait)\b', re.IGNORECASE),
            
            # Negations (complex processing)
            'negations': re.compile(r'\b(no|not|never|nothing|nobody|nowhere|neither)\b', re.IGNORECASE),
            
            # Filled pauses (hesitation)
            'filled_pauses': re.compile(r'\b(uh|um|er|ah|hmm|like)\b', re.IGNORECASE)
        }
        
    def analyze_text(self, text, speaker_role=None):
        """
        Comprehensive linguistic analysis of utterance
        
        Args:
            text: Transcribed text
            speaker_role: "Interrogator" or "Suspect" (context-dependent analysis)
            
        Returns:
            Dictionary of linguistic features and stress indicators
        """
        features = {}
        
        # Basic statistics
        words = text.lower().split()
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0.0
        
        # === EMOTIONAL CONTENT ===
        emotion_scores = self._analyze_emotional_content(text)
        features.update(emotion_scores)
        
        # === PRONOUN USAGE ===
        pronoun_features = self._analyze_pronouns(text)
        features.update(pronoun_features)
        
        # === TEMPORAL MARKERS ===
        temporal_features = self._analyze_temporal_markers(text)
        features.update(temporal_features)
        
        # === CERTAINTY/UNCERTAINTY ===
        certainty_features = self._analyze_certainty(text)
        features.update(certainty_features)
        
        # === COGNITIVE LOAD INDICATORS ===
        cognitive_features = self._analyze_cognitive_load(text)
        features.update(cognitive_features)
        
        # === LINGUISTIC COMPLEXITY ===
        complexity_features = self._analyze_complexity(text)
        features.update(complexity_features)
        
        # === DECEPTION INDICATORS (if suspect) ===
        if speaker_role and "suspect" in speaker_role.lower():
            deception_features = self._analyze_deception_markers(text, features)
            features.update(deception_features)
            
        # === OVERALL LINGUISTIC STRESS SCORE ===
        linguistic_stress = self._calculate_linguistic_stress_score(features, speaker_role)
        features.update(linguistic_stress)
        
        return features
        
    def _analyze_emotional_content(self, text):
        """Count emotional words by category"""
        text_lower = text.lower()
        
        emotion_counts = {}
        for category, word_list in self.stress_words.items():
            count = sum(1 for word in word_list if word in text_lower)
            emotion_counts[f'{category}_word_count'] = count
            
        # Calculate emotional intensity
        total_emotion_words = sum(emotion_counts.values())
        word_count = len(text.split())
        
        emotion_counts['emotion_word_ratio'] = total_emotion_words / max(1, word_count)
        
        return emotion_counts
        
    def _analyze_pronouns(self, text):
        """
        Pronoun usage patterns
        
        Research shows:
        - Truth-tellers: Use "I" frequently (ownership)
        - Deceivers: Avoid "I", use "we"/"they" (distance)
        """
        features = {}
        
        words = text.split()
        word_count = max(1, len(words))
        
        # Count pronouns
        first_person = len(self.patterns['first_person'].findall(text))
        second_person = len(self.patterns['second_person'].findall(text))
        third_person = len(self.patterns['third_person'].findall(text))
        
        features['first_person_count'] = first_person
        features['second_person_count'] = second_person
        features['third_person_count'] = third_person
        
        # Ratios
        features['first_person_ratio'] = first_person / word_count
        features['third_person_ratio'] = third_person / word_count
        
        # Self-reference ratio (high = more ownership/truthful)
        features['self_reference_ratio'] = first_person / max(1, first_person + third_person)
        
        return features
        
    def _analyze_temporal_markers(self, text):
        """
        Temporal specificity
        
        Research shows:
        - Truth-tellers: Specific times ("at 3:15 PM")
        - Deceivers: Vague times ("sometime in the afternoon")
        """
        specific = len(self.patterns['specific_time'].findall(text))
        vague = len(self.patterns['vague_time'].findall(text))
        
        return {
            'specific_time_count': specific,
            'vague_time_count': vague,
            'temporal_specificity_ratio': specific / max(1, specific + vague)
        }
        
    def _analyze_certainty(self, text):
        """
        Certainty vs uncertainty language
        
        Stressed/deceptive: More hedging, uncertainty
        """
        text_lower = text.lower()
        
        certainty_count = sum(1 for word in self.stress_words['certainty'] if word in text_lower)
        uncertainty_count = sum(1 for word in self.stress_words['uncertainty'] if word in text_lower)
        
        word_count = max(1, len(text.split()))
        
        return {
            'certainty_word_count': certainty_count,
            'uncertainty_word_count': uncertainty_count,
            'certainty_ratio': certainty_count / word_count,
            'uncertainty_ratio': uncertainty_count / word_count,
            'certainty_balance': (certainty_count - uncertainty_count) / max(1, certainty_count + uncertainty_count)
        }
        
    def _analyze_cognitive_load(self, text):
        """
        Cognitive load indicators
        
        High cognitive load (deception/stress):
        - More hedges
        - More self-corrections
        - More filled pauses
        - Shorter sentences
        - Simpler words
        """
        hedges = len(self.patterns['hedges'].findall(text))
        corrections = len(self.patterns['corrections'].findall(text))
        filled_pauses = len(self.patterns['filled_pauses'].findall(text))
        negations = len(self.patterns['negations'].findall(text))
        
        word_count = max(1, len(text.split()))
        
        # Sentence complexity
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0.0
        
        return {
            'hedge_count': hedges,
            'correction_count': corrections,
            'filled_pause_count': filled_pauses,
            'negation_count': negations,
            'hedge_ratio': hedges / word_count,
            'correction_ratio': corrections / word_count,
            'filled_pause_ratio': filled_pauses / word_count,
            'negation_ratio': negations / word_count,
            'avg_sentence_length': float(avg_sentence_length),
            'sentence_count': len(sentences)
        }
        
    def _analyze_complexity(self, text):
        """
        Linguistic complexity
        
        Deception/stress → Simpler language (cognitive load)
        Truth-telling → Natural complexity
        """
        words = text.split()
        
        if not words:
            return {'lexical_diversity': 0.0, 'avg_word_complexity': 0.0}
            
        # Lexical diversity (unique words / total words)
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / len(words)
        
        # Average word complexity (syllable count proxy: vowel clusters)
        complexities = []
        for word in words:
            # Rough syllable estimate: count vowel groups
            vowel_groups = len(re.findall(r'[aeiouy]+', word.lower()))
            complexities.append(max(1, vowel_groups))
            
        avg_complexity = np.mean(complexities)
        
        return {
            'lexical_diversity': float(lexical_diversity),
            'avg_word_complexity': float(avg_complexity),
            'unique_word_count': unique_words
        }
        
    def _analyze_deception_markers(self, text, existing_features):
        """
        Specific deception indicators (for suspects)
        
        Based on Reality Monitoring and SCAN (Scientific Content Analysis)
        """
        text_lower = text.lower()
        
        # Deception cue words
        deception_words = self.stress_words['deception_markers']
        deception_count = sum(1 for word in deception_words if word in text_lower)
        
        # Lack of detail (deception = vague, fewer sensory details)
        sensory_words = ['saw', 'heard', 'felt', 'smelled', 'touched', 'looked', 'sounded']
        sensory_detail_count = sum(1 for word in sensory_words if word in text_lower)
        
        # Present vs past tense (deception = more present tense for fabricated story)
        # Simplified: count "is/are/am" vs "was/were"
        present_tense = len(re.findall(r'\b(is|are|am|being)\b', text_lower))
        past_tense = len(re.findall(r'\b(was|were|been|had)\b', text_lower))
        
        word_count = max(1, len(text.split()))
        
        return {
            'deception_cue_count': deception_count,
            'deception_cue_ratio': deception_count / word_count,
            'sensory_detail_count': sensory_detail_count,
            'sensory_detail_ratio': sensory_detail_count / word_count,
            'present_tense_count': present_tense,
            'past_tense_count': past_tense,
            'tense_balance': (past_tense - present_tense) / max(1, past_tense + present_tense)
        }
        
    def _calculate_linguistic_stress_score(self, features, speaker_role):
        """
        Calculate overall linguistic stress probability (0-1)
        
        Combines multiple linguistic indicators
        """
        stress_score = 0.0
        indicators = []
        
        # High uncertainty → stress
        if features.get('uncertainty_ratio', 0) > 0.15:
            stress_score += 0.20
            indicators.append('High uncertainty language')
        elif features.get('uncertainty_ratio', 0) > 0.08:
            stress_score += 0.10
            
        # Low certainty → stress/deception
        if features.get('certainty_ratio', 0) < 0.03:
            stress_score += 0.10
            indicators.append('Lack of certainty')
            
        # Excessive hedging → cognitive load
        if features.get('hedge_ratio', 0) > 0.15:
            stress_score += 0.15
            indicators.append('Excessive hedging')
        elif features.get('hedge_ratio', 0) > 0.08:
            stress_score += 0.08
            
        # Self-corrections → processing difficulty
        if features.get('correction_ratio', 0) > 0.08:
            stress_score += 0.12
            indicators.append('Frequent self-corrections')
            
        # Filled pauses → hesitation
        if features.get('filled_pause_ratio', 0) > 0.10:
            stress_score += 0.10
            indicators.append('Frequent hesitations')
            
        # Emotional language → stress
        if features.get('anxiety_word_count', 0) > 0:
            stress_score += 0.15
            indicators.append('Anxiety language')
        if features.get('anger_word_count', 0) > 0:
            stress_score += 0.12
            indicators.append('Anger language')
            
        # Low lexical diversity → cognitive load
        if features.get('lexical_diversity', 0) < 0.40:
            stress_score += 0.08
            indicators.append('Low lexical diversity')
            
        # For suspects: deception markers
        if speaker_role and "suspect" in speaker_role.lower():
            # Deception cue words → suspicious
            if features.get('deception_cue_ratio', 0) > 0.05:
                stress_score += 0.15
                indicators.append('Deception cue words present')
                
            # Low self-reference → distancing (deception)
            if features.get('self_reference_ratio', 0) < 0.30:
                stress_score += 0.10
                indicators.append('Low self-reference (distancing)')
                
            # Low sensory details → fabrication
            if features.get('sensory_detail_ratio', 0) < 0.02:
                stress_score += 0.08
                indicators.append('Lack of sensory details')
                
        # Clip to 0-1
        stress_score = min(1.0, stress_score)
        
        # Categories
        if stress_score >= 0.60:
            category = "HIGH"
        elif stress_score >= 0.35:
            category = "MODERATE"  
        else:
            category = "LOW"
            
        return {
            'linguistic_stress_probability': float(stress_score),
            'linguistic_stress_category': category,
            'linguistic_stress_indicators': indicators
        }


class ConversationDynamicsAnalyzer:
    """
    Analyze turn-taking, response patterns, conversation flow
    """
    
    def __init__(self):
        self.conversation_history = []
        
    def add_utterance(self, timestamp, speaker_key, speaker_role, text, 
                     acoustic_features, linguistic_features):
        """Add utterance to conversation history"""
        utterance = {
            'timestamp': timestamp,
            'speaker_key': speaker_key,
            'speaker_role': speaker_role,
            'text': text,
            'acoustic': acoustic_features,
            'linguistic': linguistic_features
        }
        
        self.conversation_history.append(utterance)
        
    def analyze_response_latency(self, current_idx):
        """
        Measure time between question and answer
        
        Long latency → processing time (deception, fabrication)
        """
        if current_idx < 1:
            return None
            
        current = self.conversation_history[current_idx]
        previous = self.conversation_history[current_idx - 1]
        
        # Check if this is a response to previous speaker
        if current['speaker_key'] != previous['speaker_key']:
            # Different speaker (turn change)
            latency = (current['timestamp'] - previous['timestamp']).total_seconds()
            
            # Classify latency
            if latency < 0.5:
                latency_category = "IMMEDIATE"  # Quick response
            elif latency < 1.5:
                latency_category = "NORMAL"  # Normal thinking time
            elif latency < 3.0:
                latency_category = "DELAYED"  # Some hesitation
            else:
                latency_category = "EXCESSIVE"  # Long pause (suspicious)
                
            return {
                'response_latency_seconds': latency,
                'latency_category': latency_category,
                'is_turn_change': True
            }
        else:
            return {'is_turn_change': False}
            
    def analyze_interruption_patterns(self):
        """
        Detect interruptions, overlaps, dominance patterns
        """
        if len(self.conversation_history) < 2:
            return {}
            
        # Count turn changes
        turn_changes = 0
        interruptions = 0  # Quick turn changes (<0.3s)
        
        for i in range(1, len(self.conversation_history)):
            if self.conversation_history[i]['speaker_key'] != self.conversation_history[i-1]['speaker_key']:
                turn_changes += 1
                
                # Check if rapid (interruption)
                latency = (self.conversation_history[i]['timestamp'] - 
                          self.conversation_history[i-1]['timestamp']).total_seconds()
                          
                if latency < 0.3:
                    interruptions += 1
                    
        # Speaking time per person
        speaker_times = {}
        for utterance in self.conversation_history:
            key = utterance['speaker_key']
            # Estimate duration from word count (rough: 2.5 words/second)
            duration = utterance['linguistic'].get('word_count', 0) / 2.5
            
            if key not in speaker_times:
                speaker_times[key] = 0.0
            speaker_times[key] += duration
            
        # Dominance (who speaks more)
        if speaker_times:
            max_time = max(speaker_times.values())
            min_time = min(speaker_times.values())
            dominance = max_time / (min_time + 1e-10) if min_time > 0 else 10.0
        else:
            dominance = 1.0
            
        return {
            'turn_changes': turn_changes,
            'interruptions': interruptions,
            'interruption_ratio': interruptions / max(1, turn_changes),
            'speaking_dominance': float(dominance),
            'speaker_times': speaker_times
        }
        
    def analyze_response_coherence(self, current_idx):
        """
        Measure how well response addresses previous question
        
        Evasive answers → lack of coherence
        """
        if current_idx < 1:
            return {'coherence_score': 1.0}
            
        current = self.conversation_history[current_idx]
        previous = self.conversation_history[current_idx - 1]
        
        # Simple heuristic: word overlap between question and answer
        prev_words = set(previous['text'].lower().split())
        curr_words = set(current['text'].lower().split())
        
        # Remove stop words (simplified)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it'}
        prev_words = prev_words - stop_words
        curr_words = curr_words - stop_words
        
        # Jaccard similarity
        if len(prev_words | curr_words) > 0:
            coherence = len(prev_words & curr_words) / len(prev_words | curr_words)
        else:
            coherence = 0.0
            
        # Check for evasive indicators
        evasive_phrases = ['i dont remember', "don't remember", 'not sure', 'dont know', "don't know",
                          'maybe', 'i guess', 'possibly']
        
        is_evasive = any(phrase in current['text'].lower() for phrase in evasive_phrases)
        
        return {
            'response_coherence': float(coherence),
            'is_evasive': is_evasive
        }

