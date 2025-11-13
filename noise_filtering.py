"""
Noise Filtering and Unknown Speaker Rejection
Prevents background speakers and noise from being identified as enrolled participants
"""

import numpy as np
from scipy import signal as scipy_signal


class BackgroundSpeakerFilter:
    """
    Filters out speakers who are not part of the enrolled set
    Critical for interview/interrogation scenarios
    """
    
    def __init__(self, min_confidence=0.75, min_energy=1000, max_distance_ratio=2.5):
        """
        Initialize background filter
        
        Args:
            min_confidence: Minimum confidence to accept ANY speaker
            min_energy: Minimum audio energy (filters distant/quiet speakers)
            max_distance_ratio: Max ratio of distances to closest speaker (filters unknowns)
        """
        self.min_confidence = min_confidence
        self.min_energy = min_energy
        self.max_distance_ratio = max_distance_ratio
        
        # Statistics
        self.total_segments = 0
        self.accepted_segments = 0
        self.rejected_low_energy = 0
        self.rejected_low_confidence = 0
        self.rejected_unknown = 0
        
    def should_accept_speaker(self, audio_data, speaker_key, confidence, all_similarities, sample_rate=16000):
        """
        Determine if identified speaker should be accepted or rejected as noise
        
        Args:
            audio_data: Audio segment
            speaker_key: Identified speaker
            confidence: Confidence score
            all_similarities: Dict of {speaker_key: similarity} for all enrolled speakers
            sample_rate: Sample rate
            
        Returns:
            (accept: bool, reason: str, quality_score: float)
        """
        self.total_segments += 1
        
        # Check 1: Energy level (filter distant/background speakers)
        energy = self._calculate_energy(audio_data)
        
        if energy < self.min_energy:
            self.rejected_low_energy += 1
            reason = f"Low energy ({energy:.0f} < {self.min_energy}) - likely background/distant speaker"
            return False, reason, 0.0
            
        # Check 2: Minimum confidence (must be above threshold)
        if confidence < self.min_confidence:
            self.rejected_low_confidence += 1
            reason = f"Low confidence ({confidence:.2f} < {self.min_confidence}) - not a clear match"
            return False, reason, 0.0
            
        # Check 3: Unknown speaker detection
        # Compare best match to second-best match
        # If all similarities are similar (no clear winner), likely unknown speaker
        
        similarities_sorted = sorted(all_similarities.values(), reverse=True)
        
        if len(similarities_sorted) >= 2:
            best_sim = similarities_sorted[0]
            second_sim = similarities_sorted[1]
            
            # If best and second are too close, speaker doesn't match enrolled set well
            similarity_ratio = best_sim / (second_sim + 1e-6)
            
            if similarity_ratio < 1.2:  # Less than 20% better than second choice
                self.rejected_unknown += 1
                reason = f"Ambiguous match (ratio: {similarity_ratio:.2f}) - likely unknown speaker"
                return False, reason, 0.0
                
        # Check 4: Distance from enrolled speaker set
        # Calculate "distance" to all enrolled speakers
        avg_similarity = np.mean(list(all_similarities.values()))
        
        # If average similarity is too low, speaker is unlike anyone enrolled
        if avg_similarity < 0.40:  # Very different from all enrolled speakers
            self.rejected_unknown += 1
            reason = f"Low avg similarity ({avg_similarity:.2f}) - not from enrolled set"
            return False, reason, 0.0
            
        # Check 5: Audio quality indicators
        quality_score = self._calculate_quality_score(audio_data, sample_rate)
        
        if quality_score < 0.5:
            self.rejected_low_confidence += 1
            reason = f"Low quality score ({quality_score:.2f}) - likely noise/interference"
            return False, reason, quality_score
            
        # All checks passed!
        self.accepted_segments += 1
        
        acceptance_rate = (self.accepted_segments / self.total_segments * 100) if self.total_segments > 0 else 0
        reason = f"ACCEPTED (energy: {energy:.0f}, conf: {confidence:.2f}, quality: {quality_score:.2f}, rate: {acceptance_rate:.1f}%)"
        
        return True, reason, quality_score
        
    def _calculate_energy(self, audio_data):
        """Calculate RMS energy of audio segment"""
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32)
        else:
            audio_float = audio_data.astype(np.float32) * 32768.0
            
        rms = np.sqrt(np.mean(audio_float ** 2))
        return rms
        
    def _calculate_quality_score(self, audio_data, sample_rate):
        """
        Calculate audio quality score
        
        Indicators:
        - Signal-to-noise ratio estimate
        - Spectral clarity
        - Stable pitch (not random noise)
        
        Returns:
            Quality score 0-1 (higher = better)
        """
        try:
            if audio_data.dtype == np.int16:
                audio = audio_data.astype(np.float32) / 32768.0
            else:
                audio = audio_data.astype(np.float32)
                
            # 1. Signal energy
            energy = np.sqrt(np.mean(audio ** 2))
            energy_score = min(1.0, energy / 0.1)  # Normalize to 0-1
            
            # 2. Spectral flatness (noise is flat, speech is structured)
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            
            # Geometric mean / Arithmetic mean
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
            
            # Lower flatness = more structured (speech-like)
            structure_score = 1.0 - spectral_flatness
            
            # 3. Zero-crossing rate (noise has erratic ZCR, speech is more stable)
            zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
            zcr_score = 1.0 - min(1.0, abs(zcr - 0.1) / 0.1)  # Penalize very high/low
            
            # Combined quality score
            quality = 0.4 * energy_score + 0.4 * structure_score + 0.2 * zcr_score
            
            return quality
            
        except Exception as e:
            print(f"Error calculating quality: {e}")
            return 0.5  # Neutral score on error
            
    def get_statistics(self):
        """Get filtering statistics"""
        return {
            'total_segments': self.total_segments,
            'accepted': self.accepted_segments,
            'rejected_low_energy': self.rejected_low_energy,
            'rejected_low_confidence': self.rejected_low_confidence,
            'rejected_unknown': self.rejected_unknown,
            'acceptance_rate': (self.accepted_segments / self.total_segments * 100) if self.total_segments > 0 else 0
        }


class ProximityBasedFilter:
    """
    Filters speakers based on audio proximity/distance
    Assumes enrolled speakers are close to microphone, background speakers are far
    """
    
    def __init__(self, min_snr=10.0):
        """
        Initialize proximity filter
        
        Args:
            min_snr: Minimum signal-to-noise ratio (dB)
        """
        self.min_snr = min_snr
        self.noise_floor = None
        self.calibration_samples = []
        
    def calibrate_noise_floor(self, audio_data):
        """
        Calibrate noise floor from silence/background
        
        Args:
            audio_data: Audio segment of background noise (no speech)
        """
        if audio_data.dtype == np.int16:
            audio = audio_data.astype(np.float32) / 32768.0
        else:
            audio = audio_data.astype(np.float32)
            
        # Calculate noise floor (RMS of background)
        self.noise_floor = np.sqrt(np.mean(audio ** 2))
        print(f"📏 Noise floor calibrated: {self.noise_floor:.6f}")
        
    def estimate_snr(self, audio_data):
        """
        Estimate Signal-to-Noise Ratio
        
        Args:
            audio_data: Audio segment with speech
            
        Returns:
            SNR in dB
        """
        if audio_data.dtype == np.int16:
            audio = audio_data.astype(np.float32) / 32768.0
        else:
            audio = audio_data.astype(np.float32)
            
        # Signal power (RMS of speech)
        signal_power = np.sqrt(np.mean(audio ** 2))
        
        # Noise power (use calibrated or estimate)
        if self.noise_floor is not None:
            noise_power = self.noise_floor
        else:
            # Estimate from quietest 20% of frames
            frame_size = 1024
            num_frames = len(audio) // frame_size
            frame_energies = []
            
            for i in range(num_frames):
                start = i * frame_size
                end = start + frame_size
                frame = audio[start:end]
                energy = np.sqrt(np.mean(frame ** 2))
                frame_energies.append(energy)
                
            if frame_energies:
                # Noise floor = average of quietest 20%
                sorted_energies = sorted(frame_energies)
                quiet_count = max(1, len(sorted_energies) // 5)
                noise_power = np.mean(sorted_energies[:quiet_count])
            else:
                noise_power = 0.01
                
        # Calculate SNR in dB
        snr_db = 20 * np.log10((signal_power / (noise_power + 1e-10)))
        
        return snr_db
        
    def is_proximate_speaker(self, audio_data):
        """
        Determine if speaker is close (enrolled) or far (background)
        
        Args:
            audio_data: Audio segment
            
        Returns:
            (is_close: bool, snr: float)
        """
        snr = self.estimate_snr(audio_data)
        
        is_close = snr >= self.min_snr
        
        return is_close, snr


class EnrolledSpeakerOnlyVerifier:
    """
    Ensures ONLY enrolled speakers are identified
    Rejects unknown speakers, background voices, and noise
    """
    
    def __init__(self, verification_engine, background_filter, proximity_filter=None):
        """
        Initialize enrolled-speaker-only verifier
        
        Args:
            verification_engine: SpeakerVerificationEngine instance
            background_filter: BackgroundSpeakerFilter instance
            proximity_filter: Optional ProximityBasedFilter
        """
        self.verification = verification_engine
        self.background_filter = background_filter
        self.proximity_filter = proximity_filter
        
    def verify_with_filtering(self, audio_data, sample_rate=16000):
        """
        Verify speaker with comprehensive filtering
        
        Args:
            audio_data: Audio segment
            sample_rate: Sample rate
            
        Returns:
            (accepted: bool, speaker_key, speaker_name, confidence, reason)
        """
        # First, verify against enrolled speakers
        speaker_key, speaker_name, confidence, metadata = self.verification.verify_speaker(
            audio_data, sample_rate, use_context=True
        )
        
        # Get all similarities
        all_similarities = metadata.get('all_similarities', {})
        
        # Apply background speaker filter
        accept, reason, quality = self.background_filter.should_accept_speaker(
            audio_data,
            speaker_key,
            confidence,
            all_similarities,
            sample_rate
        )
        
        if not accept:
            return False, None, "FILTERED", 0.0, reason
            
        # Apply proximity filter if available
        if self.proximity_filter:
            is_close, snr = self.proximity_filter.is_proximate_speaker(audio_data)
            
            if not is_close:
                reason = f"Distant speaker (SNR: {snr:.1f}dB < {self.proximity_filter.min_snr}dB) - likely background"
                return False, None, "FILTERED", 0.0, reason
                
        # All filters passed - accept speaker
        return True, speaker_key, speaker_name, confidence, reason
        
    def get_filtering_statistics(self):
        """Get filtering statistics"""
        stats = self.background_filter.get_statistics()
        
        return {
            'total_segments': stats['total_segments'],
            'accepted': stats['accepted'],
            'rejected_total': stats['total_segments'] - stats['accepted'],
            'rejected_low_energy': stats['rejected_low_energy'],
            'rejected_low_confidence': stats['rejected_low_confidence'],
            'rejected_unknown': stats['rejected_unknown'],
            'acceptance_rate': stats['acceptance_rate'],
            'rejection_rate': 100 - stats['acceptance_rate']
        }

