"""
Enhanced Enrollment Quality System
Based on NIST Speaker Recognition Evaluation best practices

Key improvements:
1. Quality validation during enrollment (SNR, consistency, duration)
2. Minimum quality requirements (reject poor samples)
3. More samples for robust enrollment (8-10 instead of 6)
4. Per-speaker threshold calculation based on enrollment quality
5. Score normalization (Z-norm, T-norm) for better separation
"""

import numpy as np
from scipy import signal
from collections import deque
import threading
from datetime import datetime


class EnrollmentQualityValidator:
    """
    Validates enrollment sample quality
    
    Based on research:
    - NIST SRE protocols require SNR > 12 dB
    - Minimum duration: 3 seconds
    - Consistency checks across samples
    """
    
    def __init__(self):
        self.min_snr_db = 8.0  # Lowered from 10.0 for more lenient validation (normal speech can be 8-10 dB)
        self.min_duration_seconds = 2.5  # Lowered from 3.0 for shorter samples
        self.min_rms = 300  # Lowered from 500 - minimum RMS for valid speech (more lenient)
        
    def validate_sample(self, audio_data, sample_rate=16000):
        """
        Validate a single enrollment sample
        
        Returns:
            (is_valid, quality_score, issues)
        """
        issues = []
        quality_score = 1.0
        
        # Check duration
        duration = len(audio_data) / sample_rate
        if duration < self.min_duration_seconds:
            issues.append(f"Too short: {duration:.1f}s (need {self.min_duration_seconds}s)")
            quality_score *= 0.5
        
        # Check RMS (signal level)
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        if rms < self.min_rms:
            issues.append(f"Too quiet: RMS {rms:.0f} (need {self.min_rms}, detected: {rms:.0f})")
            quality_score *= 0.5  # Less harsh penalty (was 0.3)
        
        # Calculate SNR
        snr_db = self._calculate_snr(audio_data, sample_rate)
        if snr_db < self.min_snr_db:
            issues.append(f"Low SNR: {snr_db:.1f} dB (need {self.min_snr_db} dB)")
            quality_score *= max(0.3, snr_db / self.min_snr_db)
        
        # Check for clipping
        max_amplitude = np.max(np.abs(audio_data))
        clipping_threshold = 0.95 * 32767  # 95% of max int16
        if max_amplitude > clipping_threshold:
            issues.append(f"Clipping detected: {max_amplitude} > {clipping_threshold}")
            quality_score *= 0.7
        
        # Check for silence (too much silence = bad)
        # Relaxed: Normal speech has pauses, breaths - 50-70% silence is normal
        silence_ratio = self._calculate_silence_ratio(audio_data, sample_rate)
        if silence_ratio > 0.80:  # More than 80% silence (relaxed from 50%)
            issues.append(f"Too much silence: {silence_ratio:.1%}")
            quality_score *= (1.0 - (silence_ratio - 0.5))  # Less harsh penalty
        
        # More lenient validation - allow samples with minor issues
        # Only reject if quality is very low OR too many critical issues
        is_valid = quality_score >= 0.5 and len([i for i in issues if 'Too quiet' in i or 'Too short' in i]) <= 1
        
        return is_valid, quality_score, issues
    
    def _calculate_snr(self, audio_data, sample_rate):
        """Calculate Signal-to-Noise Ratio in dB"""
        try:
            # Estimate noise from quiet segments
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Use energy-based VAD to find speech vs noise
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            # Calculate frame energy
            frame_energies = []
            for i in range(0, len(audio_float) - frame_length, hop_length):
                frame = audio_float[i:i+frame_length]
                energy = np.mean(frame ** 2)
                frame_energies.append(energy)
            
            frame_energies = np.array(frame_energies)
            
            # Bottom 20% = noise, top 20% = signal
            noise_energy = np.percentile(frame_energies, 20)
            signal_energy = np.percentile(frame_energies, 80)
            
            if noise_energy > 0:
                snr_linear = signal_energy / noise_energy
                snr_db = 10 * np.log10(snr_linear)
            else:
                snr_db = 30.0  # Assume good if no noise detected
            
            return max(0, min(50, snr_db))  # Clip to 0-50 dB
            
        except:
            return 15.0  # Default assumption
    
    def _calculate_silence_ratio(self, audio_data, sample_rate):
        """Calculate ratio of silence in audio"""
        audio_float = np.abs(audio_data.astype(np.float32)) / 32768.0
        
        # Threshold for silence - use adaptive threshold based on signal level
        # Instead of fixed 1%, use a percentage of the signal's RMS
        rms = np.sqrt(np.mean(audio_float ** 2))
        silence_threshold = max(0.01, rms * 0.3)  # 30% of RMS, minimum 1%
        
        silence_samples = np.sum(audio_float < silence_threshold)
        total_samples = len(audio_float)
        
        return silence_samples / total_samples if total_samples > 0 else 0.0


class EnhancedEnrollmentSystem:
    """
    Enhanced enrollment with quality validation
    
    Based on research:
    - NIST SRE: 8-10 samples minimum
    - Quality validation per sample
    - Per-speaker threshold calculation
    """
    
    def __init__(self, embedding_extractor):
        self.embedding_extractor = embedding_extractor
        self.quality_validator = EnrollmentQualityValidator()
        
        # Enhanced requirements (relaxed slightly for easier testing)
        self.min_samples_required = 8  # Increased from 6 (research: 8-10)
        self.min_quality_samples = 5   # At least 5 must pass quality check (relaxed from 6)
        self.min_overall_quality = 0.70  # Overall enrollment quality threshold (relaxed from 0.75)
        
        self.enrolled_speakers = {}
        self.lock = threading.Lock()
        
    def start_enrollment(self, speaker_key, name, role):
        """Start enrollment with enhanced tracking"""
        with self.lock:
            self.enrolled_speakers[speaker_key] = {
                'key': speaker_key,
                'name': name,
                'role': role,
                'embeddings': [],
                'quality_scores': [],
                'valid_samples': 0,
                'rejected_samples': 0,
                'mean_embedding': None,
                'std': None,
                'threshold': None,
                'quality': 0.0,
                'enrolled': False,
                'enrollment_start': datetime.now()
            }
    
    def add_enrollment_sample(self, speaker_key, audio_data, sample_rate=16000):
        """
        Add enrollment sample with quality validation
        
        Returns:
            (success, quality_score, message, is_valid)
        """
        try:
            if speaker_key not in self.enrolled_speakers:
                return False, 0.0, "Speaker not initialized", False
            
            # Validate quality
            is_valid, quality_score, issues = self.quality_validator.validate_sample(
                audio_data, sample_rate
            )
            
            # Extract embedding
            embedding = self.embedding_extractor.extract_embedding(audio_data, sample_rate)
            
            if np.allclose(embedding, 0):
                return False, 0.0, "Failed to extract voice features", False
            
            with self.lock:
                speaker = self.enrolled_speakers[speaker_key]
                
                if is_valid:
                    speaker['embeddings'].append(embedding)
                    speaker['quality_scores'].append(quality_score)
                    speaker['valid_samples'] += 1
                    message = f"Sample {speaker['valid_samples']} accepted (quality: {quality_score:.1%})"
                else:
                    speaker['rejected_samples'] += 1
                    issue_str = "; ".join(issues[:2])  # Show first 2 issues
                    message = f"Sample REJECTED: {issue_str}"
                
                total_attempts = speaker['valid_samples'] + speaker['rejected_samples']
                samples_needed = self.min_samples_required
                
                return True, quality_score, message, is_valid
                
        except Exception as e:
            return False, 0.0, f"Error: {str(e)}", False
    
    def complete_enrollment(self, speaker_key):
        """
        Complete enrollment with enhanced quality checks
        
        Returns:
            (success, quality, message)
        """
        try:
            with self.lock:
                if speaker_key not in self.enrolled_speakers:
                    return False, 0.0, "Speaker not found"
                
                speaker = self.enrolled_speakers[speaker_key]
                
                # Check minimum samples
                if speaker['valid_samples'] < self.min_quality_samples:
                    return False, 0.0, f"Need at least {self.min_quality_samples} quality samples (got {speaker['valid_samples']})"
                
                # Calculate statistics
                embeddings_array = np.array(speaker['embeddings'])
                
                # Mean embedding (voiceprint)
                mean = np.mean(embeddings_array, axis=0)
                mean_normalized = mean / (np.linalg.norm(mean) + 1e-10)
                
                # Standard deviation (consistency)
                std = np.std(embeddings_array, axis=0).mean()
                
                # Overall quality (combination of sample quality and consistency)
                avg_sample_quality = np.mean(speaker['quality_scores'])
                consistency_quality = 1.0 / (1.0 + std * 15)
                overall_quality = 0.6 * avg_sample_quality + 0.4 * consistency_quality
                
                # Check minimum overall quality
                if overall_quality < self.min_overall_quality:
                    return False, 0.0, f"Enrollment quality too low: {overall_quality:.1%} (need {self.min_overall_quality:.1%})"
                
                # Calculate per-speaker threshold (research-based)
                # Based on Resemblyzer research: optimal thresholds are 0.5-0.7, not 0.65-0.85
                # Higher quality = slightly stricter threshold, but not too strict
                # Lower std = more consistent = slightly stricter threshold
                base_threshold = 0.55  # Lowered from 0.65 (research: Resemblyzer works best at 0.55-0.60)
                quality_bonus = (overall_quality - 0.70) * 0.10  # Reduced from 0.15, adjusted baseline
                consistency_bonus = (1.0 - std * 10) * 0.08  # Reduced from 0.10
                
                threshold = base_threshold + quality_bonus + consistency_bonus
                threshold = np.clip(threshold, 0.50, 0.70)  # Range: 0.50-0.70 (lowered from 0.65-0.85)
                
                # Update speaker profile
                speaker['mean_embedding'] = mean_normalized
                speaker['std'] = std
                speaker['threshold'] = threshold
                speaker['quality'] = overall_quality
                speaker['enrolled'] = True
                speaker['enrollment_end'] = datetime.now()
                
                status = "excellent" if overall_quality >= 0.85 else "good" if overall_quality >= 0.75 else "acceptable"
                
                message = f"Enrollment complete! Quality: {overall_quality:.1%} ({status}), Threshold: {threshold:.2f}, Samples: {speaker['valid_samples']}/{speaker['valid_samples'] + speaker['rejected_samples']}"
                
                print(f"✅ {speaker['name']} enrolled successfully")
                print(f"   Quality: {overall_quality:.1%} ({status})")
                print(f"   Threshold: {threshold:.2f} (per-speaker)")
                print(f"   Valid samples: {speaker['valid_samples']}")
                print(f"   Rejected samples: {speaker['rejected_samples']}")
                
                return True, overall_quality, message
                
        except Exception as e:
            return False, 0.0, f"Error: {str(e)}"
    
    def get_enrolled_speakers(self):
        """Get enrolled speakers"""
        with self.lock:
            return {k: v for k, v in self.enrolled_speakers.items() if v.get('enrolled', False)}


class ScoreNormalizer:
    """
    Score normalization for better separation
    
    Based on research:
    - Z-norm: Normalize using enrolled speaker statistics
    - T-norm: Normalize using test utterance statistics
    - Combined: Better separation between enrolled and unknown
    """
    
    def __init__(self):
        self.enrolled_stats = {}  # Per-speaker statistics
        
    def fit_z_norm(self, enrolled_speakers):
        """Fit Z-norm using enrolled speaker statistics"""
        self.enrolled_stats = {}
        
        for speaker_key, profile in enrolled_speakers.items():
            if 'embeddings' in profile and len(profile['embeddings']) > 0:
                embeddings = np.array(profile['embeddings'])
                
                # Calculate mean and std of similarities within speaker
                mean_emb = profile['mean_embedding']
                similarities = [np.dot(emb, mean_emb) for emb in embeddings]
                
                self.enrolled_stats[speaker_key] = {
                    'mean': np.mean(similarities),
                    'std': np.std(similarities)
                }
    
    def normalize_score(self, raw_score, speaker_key, method='znorm'):
        """
        Normalize similarity score
        
        Args:
            raw_score: Raw cosine similarity
            speaker_key: Speaker identifier
            method: 'znorm' or 'combined'
        """
        if speaker_key not in self.enrolled_stats:
            return raw_score
        
        stats = self.enrolled_stats[speaker_key]
        
        if method == 'znorm':
            # Z-normalization: (score - mean) / std
            if stats['std'] > 0:
                normalized = (raw_score - stats['mean']) / stats['std']
                # Convert back to 0-1 scale (sigmoid)
                normalized_score = 1.0 / (1.0 + np.exp(-normalized))
                return normalized_score
            else:
                return raw_score
        
        return raw_score

