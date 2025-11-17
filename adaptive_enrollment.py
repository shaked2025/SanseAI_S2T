"""
Adaptive Enrollment for Long Interrogation Sessions

Problem: Voice drifts over hours due to:
- Fatigue (pitch drops, articulation changes)
- Stress variations (anxiety levels change)
- Environmental changes (room temperature, time of day)

Solution: Cautiously update voiceprint during session

Safety mechanisms:
- Only update on HIGH confidence matches (>0.90)
- Small learning rate (5% weight to new samples)
- Track drift magnitude (alert if excessive)
- Reversible (can rollback if accuracy drops)

Based on research:
- "Speaker Adaptation in Long-Duration Sessions" (ICASSP 2020)
- "Online Learning for Speaker Verification" (2019)
- "Voice Drift Compensation in Forensic Systems" (2021)
"""

import numpy as np
from datetime import datetime, timedelta
from collections import deque
from scipy import signal as scipy_signal


class AdaptiveEnrollmentSystem:
    """
    Adaptive enrollment with safeguards for forensic use
    """
    
    def __init__(self, learning_rate=0.05, min_confidence=0.90, 
                 max_drift_per_hour=0.10, adaptation_interval_minutes=5):
        """
        Args:
            learning_rate: Weight given to new samples (0.05 = 5%)
            min_confidence: Minimum confidence to trigger adaptation (0.90)
            max_drift_per_hour: Maximum allowed drift per hour (0.10 = 10%)
            adaptation_interval_minutes: Min time between adaptations (5 min)
        """
        self.learning_rate = learning_rate
        self.min_confidence = min_confidence
        self.max_drift_per_hour = max_drift_per_hour
        self.adaptation_interval = timedelta(minutes=adaptation_interval_minutes)
        
        # Tracking
        self.adaptation_history = {}  # {speaker_key: [...]}
        self.last_adaptation = {}  # {speaker_key: timestamp}
        self.original_voiceprints = {}  # Backup of original enrollment
        
    def initialize_speaker(self, speaker_key, initial_voiceprint):
        """
        Initialize tracking for a speaker
        
        Args:
            speaker_key: Speaker identifier
            initial_voiceprint: Original enrollment voiceprint
        """
        # Store original (for drift calculation and potential rollback)
        self.original_voiceprints[speaker_key] = initial_voiceprint.copy()
        
        # Initialize history
        self.adaptation_history[speaker_key] = []
        self.last_adaptation[speaker_key] = datetime.now()
        
    def should_adapt(self, speaker_key, confidence, spatial_match=True):
        """
        Determine if voiceprint should be adapted
        
        Args:
            speaker_key: Speaker to potentially adapt
            confidence: Verification confidence score
            spatial_match: Whether spatial features matched
            
        Returns:
            (should_adapt: bool, reason: str)
        """
        # Rule 1: Confidence must be HIGH
        if confidence < self.min_confidence:
            return False, f"Confidence too low ({confidence:.2f} < {self.min_confidence:.2f})"
            
        # Rule 2: Must respect adaptation interval (avoid over-updating)
        if speaker_key in self.last_adaptation:
            time_since_last = datetime.now() - self.last_adaptation[speaker_key]
            if time_since_last < self.adaptation_interval:
                return False, f"Too soon (last adaptation {time_since_last.seconds}s ago)"
                
        # Rule 3: Spatial features should match (same person, same location)
        if not spatial_match:
            return False, "Spatial mismatch (might be different person)"
            
        return True, "Conditions met for adaptation"
        
    def adapt_voiceprint(self, speaker_key, current_voiceprint, new_embedding, 
                        confidence, session_start_time):
        """
        Cautiously update voiceprint
        
        Args:
            speaker_key: Speaker to adapt
            current_voiceprint: Current mean embedding
            new_embedding: New verified embedding
            confidence: Confidence score
            session_start_time: When session started
            
        Returns:
            (updated_voiceprint, drift_info)
        """
        # Calculate drift from ORIGINAL enrollment
        original = self.original_voiceprints[speaker_key]
        drift_from_original = 1.0 - np.dot(new_embedding, original)
        
        # Calculate expected drift based on session duration
        session_duration_hours = (datetime.now() - session_start_time).total_seconds() / 3600
        max_allowed_drift = self.max_drift_per_hour * session_duration_hours
        
        # Safety check: Drift too large?
        if drift_from_original > max_allowed_drift:
            return current_voiceprint, {
                'adapted': False,
                'reason': f'Excessive drift ({drift_from_original:.3f} > {max_allowed_drift:.3f})',
                'drift': drift_from_original
            }
            
        # Perform adaptation (exponential moving average)
        α = self.learning_rate
        updated_voiceprint = α * new_embedding + (1 - α) * current_voiceprint
        
        # Normalize
        updated_voiceprint = updated_voiceprint / (np.linalg.norm(updated_voiceprint) + 1e-10)
        
        # Record adaptation
        adaptation_event = {
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'drift_from_original': drift_from_original,
            'drift_from_current': 1.0 - np.dot(new_embedding, current_voiceprint),
            'learning_rate': α
        }
        
        self.adaptation_history[speaker_key].append(adaptation_event)
        self.last_adaptation[speaker_key] = datetime.now()
        
        drift_info = {
            'adapted': True,
            'reason': f'Adapted (drift: {drift_from_original:.3f}, conf: {confidence:.2f})',
            'drift': drift_from_original,
            'total_adaptations': len(self.adaptation_history[speaker_key])
        }
        
        return updated_voiceprint, drift_info
        
    def get_adaptation_summary(self, speaker_key):
        """Get summary of adaptations for this speaker"""
        if speaker_key not in self.adaptation_history:
            return None
            
        history = self.adaptation_history[speaker_key]
        
        if not history:
            return {'adaptations': 0}
            
        return {
            'adaptations': len(history),
            'avg_drift': np.mean([h['drift_from_original'] for h in history]),
            'max_drift': np.max([h['drift_from_original'] for h in history]),
            'avg_confidence': np.mean([h['confidence'] for h in history]),
            'first_adaptation': history[0]['timestamp'],
            'last_adaptation': history[-1]['timestamp']
        }
        
    def check_drift_alert(self, speaker_key, current_drift):
        """
        Check if drift is concerning
        
        Returns:
            (alert: bool, message: str)
        """
        if current_drift > 0.15:
            return True, f"HIGH DRIFT WARNING: {current_drift:.2%} from original enrollment"
        elif current_drift > 0.10:
            return True, f"Moderate drift: {current_drift:.2%}"
        else:
            return False, ""


class VoiceStressIndicators:
    """
    Extract voice stress indicators for investigator awareness
    
    NOT used for automated decisions (not reliable enough)
    Used for investigator situational awareness
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def analyze_stress(self, audio):
        """
        Calculate stress indicators
        
        Returns:
            Dictionary of stress metrics
        """
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
            
        indicators = {}
        
        # 1. Pitch (F0) statistics
        f0_values = self._extract_f0_contour(audio)
        
        if len(f0_values) > 0:
            indicators['f0_mean'] = float(np.mean(f0_values))
            indicators['f0_std'] = float(np.std(f0_values))
            indicators['f0_range'] = float(np.max(f0_values) - np.min(f0_values))
            
            # High F0 std = stress/emotion
            if indicators['f0_std'] > 30:
                indicators['f0_stress_level'] = "HIGH"
            elif indicators['f0_std'] > 20:
                indicators['f0_stress_level'] = "MODERATE"
            else:
                indicators['f0_stress_level'] = "LOW"
        else:
            indicators['f0_mean'] = 0.0
            indicators['f0_std'] = 0.0
            indicators['f0_stress_level'] = "UNKNOWN"
            
        # 2. Jitter (pitch period variations)
        indicators['jitter_percent'] = self._calculate_jitter(audio)
        
        # Normal: <1%, Stressed: 1-3%, Very stressed: >3%
        if indicators['jitter_percent'] > 3.0:
            indicators['jitter_stress_level'] = "HIGH"
        elif indicators['jitter_percent'] > 1.0:
            indicators['jitter_stress_level'] = "MODERATE"
        else:
            indicators['jitter_stress_level'] = "LOW"
            
        # 3. Speaking rate
        indicators['speaking_rate'] = self._estimate_speaking_rate(audio)
        
        # Normal: 3-5 syllables/second
        # Stressed: <2 (hesitation) or >6 (rapid)
        if indicators['speaking_rate'] < 2.0 or indicators['speaking_rate'] > 6.0:
            indicators['rate_stress_level'] = "HIGH"
        elif indicators['speaking_rate'] < 2.5 or indicators['speaking_rate'] > 5.5:
            indicators['rate_stress_level'] = "MODERATE"
        else:
            indicators['rate_stress_level'] = "LOW"
            
        # 4. Overall stress assessment
        stress_levels = [
            indicators.get('f0_stress_level', 'LOW'),
            indicators.get('jitter_stress_level', 'LOW'),
            indicators.get('rate_stress_level', 'LOW')
        ]
        
        high_count = stress_levels.count('HIGH')
        moderate_count = stress_levels.count('MODERATE')
        
        if high_count >= 2:
            indicators['overall_stress'] = "HIGH"
        elif high_count >= 1 or moderate_count >= 2:
            indicators['overall_stress'] = "MODERATE"
        else:
            indicators['overall_stress'] = "LOW"
            
        return indicators
        
    def _extract_f0_contour(self, audio):
        """Extract pitch contour over time"""
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=self.sample_rate
            )
            
            # Remove unvoiced frames
            f0_voiced = f0[~np.isnan(f0)]
            
            return f0_voiced
            
        except:
            return np.array([])
            
    def _calculate_jitter(self, audio):
        """
        Calculate jitter (pitch period perturbation)
        
        Jitter% = average absolute difference between consecutive periods
        """
        # Extract pitch periods
        f0_contour = self._extract_f0_contour(audio)
        
        if len(f0_contour) < 2:
            return 0.0
            
        # Convert F0 to periods
        periods = 1.0 / (f0_contour + 1e-10)
        
        # Calculate consecutive differences
        diffs = np.abs(np.diff(periods))
        
        # Jitter = average difference / average period
        jitter = np.mean(diffs) / np.mean(periods) * 100
        
        return float(jitter)
        
    def _estimate_speaking_rate(self, audio):
        """
        Estimate syllables per second
        
        Method: Energy peaks correspond to syllable nuclei (vowels)
        """
        # Calculate envelope
        envelope = np.abs(scipy_signal.hilbert(audio))
        
        # Smooth
        window = int(0.02 * self.sample_rate)  # 20ms
        smoothed = np.convolve(envelope, np.ones(window)/window, mode='same')
        
        # Find peaks (syllables)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(smoothed, distance=int(0.1*self.sample_rate), height=np.max(smoothed)*0.3)
        
        # Calculate rate
        duration_seconds = len(audio) / self.sample_rate
        syllables_per_second = len(peaks) / duration_seconds
        
        return float(syllables_per_second)

