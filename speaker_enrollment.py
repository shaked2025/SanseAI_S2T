"""
Speaker Enrollment System for Interview Transcription
Designed for interrogation/interview scenarios with known participants
"""

import numpy as np
import pickle
import os
from datetime import datetime
from collections import deque
import threading


class SpeakerEnrollment:
    """Manages speaker enrollment with voice samples"""
    
    def __init__(self, embedding_extractor):
        """
        Initialize enrollment system
        
        Args:
            embedding_extractor: Object with extract_embedding(audio, sr) method
        """
        self.embedding_extractor = embedding_extractor
        self.enrolled_speakers = {}
        self.enrollment_samples_required = 5  # Minimum samples per speaker
        self.lock = threading.Lock()
        
    def start_enrollment(self, speaker_key, name, role):
        """
        Start enrollment for a speaker
        
        Args:
            speaker_key: Unique identifier (e.g., 'interviewer', 'interviewee_1')
            name: Full name
            role: Role (Interviewer, Interviewee, Observer, etc.)
        """
        with self.lock:
            self.enrolled_speakers[speaker_key] = {
                'key': speaker_key,
                'name': name,
                'role': role,
                'embeddings': [],
                'mean_embedding': None,
                'covariance': None,
                'std': None,
                'threshold': 0.85,  # Will be calculated
                'quality': 0.0,
                'enrolled': False,
                'total_utterances': 0,
                'correct_identifications': 0,
                'enrollment_start': datetime.now()
            }
            
    def add_enrollment_sample(self, speaker_key, audio_data, sample_rate=16000):
        """
        Add voice sample for enrollment
        
        Args:
            speaker_key: Speaker identifier
            audio_data: Audio numpy array
            sample_rate: Sample rate
            
        Returns:
            (success, quality_score, message)
        """
        try:
            if speaker_key not in self.enrolled_speakers:
                return False, 0.0, "Speaker not initialized"
                
            # Extract embedding
            embedding = self.embedding_extractor.extract_embedding(audio_data, sample_rate)
            
            # Check for valid embedding
            if np.allclose(embedding, 0):
                return False, 0.0, "Failed to extract voice features"
                
            with self.lock:
                speaker = self.enrolled_speakers[speaker_key]
                speaker['embeddings'].append(embedding)
                
                # Calculate quality after each sample
                if len(speaker['embeddings']) >= 2:
                    embeddings_array = np.array(speaker['embeddings'])
                    std = np.std(embeddings_array, axis=0).mean()
                    quality = 1.0 / (1.0 + std * 20)  # 0-1 score
                    speaker['quality'] = quality
                else:
                    quality = 0.9  # First sample, assume good
                    
                samples_collected = len(speaker['embeddings'])
                samples_needed = self.enrollment_samples_required
                
                message = f"Sample {samples_collected}/{samples_needed} collected (quality: {quality:.1%})"
                
                return True, quality, message
                
        except Exception as e:
            return False, 0.0, f"Error: {str(e)}"
            
    def complete_enrollment(self, speaker_key):
        """
        Complete enrollment and calculate final voiceprint
        
        Args:
            speaker_key: Speaker identifier
            
        Returns:
            (success, quality, message)
        """
        try:
            with self.lock:
                if speaker_key not in self.enrolled_speakers:
                    return False, 0.0, "Speaker not found"
                    
                speaker = self.enrolled_speakers[speaker_key]
                
                if len(speaker['embeddings']) < self.enrollment_samples_required:
                    return False, 0.0, f"Need {self.enrollment_samples_required} samples"
                    
                # Calculate statistics
                embeddings_array = np.array(speaker['embeddings'])
                
                # Mean embedding (voiceprint)
                mean = np.mean(embeddings_array, axis=0)
                mean_normalized = mean / (np.linalg.norm(mean) + 1e-10)
                
                # Standard deviation (consistency)
                std = np.std(embeddings_array, axis=0).mean()
                
                # Covariance matrix
                try:
                    cov = np.cov(embeddings_array.T)
                    # Regularize to ensure invertibility
                    cov += np.eye(len(cov)) * 1e-6
                except:
                    cov = np.eye(len(mean))
                    
                # Calculate quality score
                # Lower std = more consistent = higher quality
                quality = 1.0 / (1.0 + std * 15)
                
                # Set threshold based on quality
                # Higher quality = stricter threshold
                threshold = 0.92 - (std * 8)
                threshold = np.clip(threshold, 0.82, 0.95)
                
                # Update speaker profile
                speaker['mean_embedding'] = mean_normalized
                speaker['covariance'] = cov
                speaker['std'] = std
                speaker['threshold'] = threshold
                speaker['quality'] = quality
                speaker['enrolled'] = True
                speaker['total_utterances'] = len(speaker['embeddings'])
                speaker['enrollment_end'] = datetime.now()
                
                if quality >= 0.80:
                    status = "excellent"
                elif quality >= 0.70:
                    status = "good"
                else:
                    status = "acceptable"
                    
                message = f"Enrollment complete! Quality: {quality:.1%} ({status}), Threshold: {threshold:.2f}"
                
                print(f"✅ {speaker['name']} enrolled successfully")
                print(f"   Quality: {quality:.1%}")
                print(f"   Threshold: {threshold:.2f}")
                print(f"   Samples: {len(speaker['embeddings'])}")
                
                return True, quality, message
                
        except Exception as e:
            return False, 0.0, f"Error completing enrollment: {str(e)}"
            
    def test_speaker_separation(self):
        """
        Test how well enrolled speakers can be distinguished
        
        Returns:
            Dictionary of separation metrics
        """
        with self.lock:
            enrolled = {k: v for k, v in self.enrolled_speakers.items() if v.get('enrolled', False)}
            
            if len(enrolled) < 2:
                return {"error": "Need at least 2 enrolled speakers"}
                
            # Test all pairs
            separations = {}
            
            for key1, speaker1 in enrolled.items():
                for key2, speaker2 in enrolled.items():
                    if key1 >= key2:  # Avoid duplicates
                        continue
                        
                    # Calculate similarity between mean embeddings
                    similarity = np.dot(
                        speaker1['mean_embedding'],
                        speaker2['mean_embedding']
                    )
                    
                    # Lower similarity = better separation
                    separation = 1.0 - similarity
                    
                    pair_key = f"{speaker1['name']} vs {speaker2['name']}"
                    separations[pair_key] = {
                        'separation': separation,
                        'similarity': similarity,
                        'distinguishable': separation > 0.15  # >15% difference
                    }
                    
                    status = "✅" if separation > 0.15 else "⚠️"
                    print(f"{status} {pair_key}: {separation:.1%} separation (similarity: {similarity:.2f})")
                    
            return separations
            
    def get_enrolled_speakers(self):
        """Get list of enrolled speakers"""
        with self.lock:
            return {k: v for k, v in self.enrolled_speakers.items() if v.get('enrolled', False)}
            
    def save_enrollment(self, filepath):
        """Save enrollment data"""
        with self.lock:
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(self.enrolled_speakers, f)
                print(f"💾 Enrollment data saved: {len(self.enrolled_speakers)} speakers")
                return True
            except Exception as e:
                print(f"❌ Error saving enrollment: {e}")
                return False
                
    def load_enrollment(self, filepath):
        """Load enrollment data"""
        with self.lock:
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        self.enrolled_speakers = pickle.load(f)
                    enrolled = sum(1 for s in self.enrolled_speakers.values() if s.get('enrolled', False))
                    print(f"📂 Loaded {enrolled} enrolled speakers from {filepath}")
                    return True
                return False
            except Exception as e:
                print(f"❌ Error loading enrollment: {e}")
                return False


class SpeakerVerificationEngine:
    """
    Speaker verification for enrolled speakers
    Much more accurate than unsupervised diarization
    """
    
    def __init__(self, enrollment_system):
        """
        Initialize verification engine
        
        Args:
            enrollment_system: SpeakerEnrollment object
        """
        self.enrollment = enrollment_system
        self.context_tracker = InterviewContextTracker()
        
        # Statistics
        self.total_verifications = 0
        self.high_confidence_count = 0
        self.low_confidence_count = 0
        
    def verify_speaker(self, audio_data, sample_rate=16000, use_context=True):
        """
        Verify which enrolled speaker is speaking
        
        Args:
            audio_data: Audio sample
            sample_rate: Sample rate
            use_context: Use interview context for boosting
            
        Returns:
            (speaker_key, speaker_name, confidence, metadata)
        """
        try:
            # Extract embedding
            embedding = self.enrollment.embedding_extractor.extract_embedding(
                audio_data, sample_rate
            )
            
            # Get enrolled speakers
            enrolled = self.enrollment.get_enrolled_speakers()
            
            if not enrolled:
                return None, "Unknown", 0.0, {"error": "No enrolled speakers"}
                
            # Calculate similarity with each enrolled speaker
            similarities = {}
            mahalanobis_scores = {}
            
            for speaker_key, profile in enrolled.items():
                # Cosine similarity
                cosine_sim = np.dot(embedding, profile['mean_embedding'])
                similarities[speaker_key] = cosine_sim
                
                # Mahalanobis distance (if covariance available)
                try:
                    diff = embedding - profile['mean_embedding']
                    cov_inv = np.linalg.inv(profile['covariance'])
                    mahal_dist = np.sqrt(diff @ cov_inv @ diff)
                    # Convert to similarity (0-1, higher better)
                    mahal_sim = 1.0 / (1.0 + mahal_dist)
                    mahalanobis_scores[speaker_key] = mahal_sim
                except:
                    mahalanobis_scores[speaker_key] = cosine_sim
                    
            # Combined score (weighted average)
            combined_scores = {}
            for speaker_key in enrolled.keys():
                combined = 0.6 * similarities[speaker_key] + 0.4 * mahalanobis_scores[speaker_key]
                combined_scores[speaker_key] = combined
                
            # Get best match
            best_speaker = max(combined_scores.items(), key=lambda x: x[1])
            speaker_key, raw_confidence = best_speaker
            
            # Apply context boosting if enabled
            confidence = raw_confidence
            if use_context:
                expected_speaker, context_confidence = self.context_tracker.predict_next_speaker()
                if expected_speaker == speaker_key:
                    # Matches expected pattern, boost slightly
                    confidence = min(1.0, raw_confidence * 1.05)
                    
            # Get speaker profile
            profile = enrolled[speaker_key]
            
            # Check threshold
            threshold = profile['threshold']
            
            self.total_verifications += 1
            
            if confidence >= threshold:
                self.high_confidence_count += 1
                match_quality = "HIGH"
            elif confidence >= 0.75:
                match_quality = "MEDIUM"
            else:
                self.low_confidence_count += 1
                match_quality = "LOW"
                
            # Update statistics
            profile['correct_identifications'] += 1
            
            # Update context tracker
            self.context_tracker.add_turn(speaker_key, confidence)
            
            # Calculate running accuracy
            accuracy = (self.high_confidence_count / self.total_verifications * 100) if self.total_verifications > 0 else 0
            
            metadata = {
                'role': profile['role'],
                'match_quality': match_quality,
                'threshold': threshold,
                'accuracy': accuracy,
                'cosine_similarity': similarities[speaker_key],
                'mahalanobis_score': mahalanobis_scores[speaker_key],
                'all_similarities': similarities
            }
            
            return speaker_key, profile['name'], confidence, metadata
            
        except Exception as e:
            print(f"❌ Verification error: {e}")
            import traceback
            traceback.print_exc()
            return None, "Unknown", 0.0, {"error": str(e)}
            
    def get_statistics(self):
        """Get verification statistics"""
        accuracy = (self.high_confidence_count / self.total_verifications * 100) if self.total_verifications > 0 else 0
        
        return {
            'total_verifications': self.total_verifications,
            'high_confidence': self.high_confidence_count,
            'low_confidence': self.low_confidence_count,
            'accuracy': accuracy,
            'enrolled_speakers': len(self.enrollment.get_enrolled_speakers())
        }


class InterviewContextTracker:
    """Tracks interview patterns for context-aware verification"""
    
    def __init__(self):
        self.turn_history = deque(maxlen=30)  # Last 30 turns
        self.interviewer_key = None
        self.interviewee_keys = []
        
    def set_roles(self, interviewer_key, interviewee_keys):
        """Set the interviewer and interviewee(s)"""
        self.interviewer_key = interviewer_key
        self.interviewee_keys = interviewee_keys if isinstance(interviewee_keys, list) else [interviewee_keys]
        
    def add_turn(self, speaker_key, confidence):
        """Add a speaking turn to history"""
        self.turn_history.append({
            'speaker': speaker_key,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
    def predict_next_speaker(self):
        """
        Predict who is likely to speak next based on Q&A pattern
        
        Returns:
            (speaker_key, confidence) or (None, 0.0)
        """
        if not self.turn_history or not self.interviewer_key:
            return None, 0.0
            
        last_turn = self.turn_history[-1]
        last_speaker = last_turn['speaker']
        
        # Interview pattern: interviewer asks, interviewee answers
        if last_speaker == self.interviewer_key:
            # Interviewer spoke last, interviewee likely next
            if self.interviewee_keys:
                # If multiple interviewees, predict based on recent pattern
                return self.interviewee_keys[0], 0.75
        else:
            # Interviewee spoke, interviewer likely to follow up
            return self.interviewer_key, 0.80
            
        return None, 0.0
        
    def get_speaker_stats(self):
        """Get speaking statistics"""
        if not self.turn_history:
            return {}
            
        stats = {}
        for turn in self.turn_history:
            speaker = turn['speaker']
            if speaker not in stats:
                stats[speaker] = {'count': 0, 'avg_confidence': []}
            stats[speaker]['count'] += 1
            stats[speaker]['avg_confidence'].append(turn['confidence'])
            
        # Calculate averages
        for speaker, data in stats.items():
            data['avg_confidence'] = np.mean(data['avg_confidence'])
            
        return stats

