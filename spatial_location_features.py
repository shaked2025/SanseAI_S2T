"""
Spatial Location Features for Fixed-Position Speaker Verification

Key insight: Speakers in FIXED positions have consistent spatial/acoustic signatures:
- Direct-to-Reverberant Ratio (DRR) - distance-dependent
- Spectral envelope - distance-dependent high-freq rolloff  
- SNR patterns - consistent for fixed location
- Reverberation characteristics - room position fingerprint

These create a "location fingerprint" that helps:
✅ Accept enrolled speakers (same position = same spatial signature)
✅ Reject passersby (different position = different spatial signature)

Based on research:
- "Spatial Features for Speaker Diarization" (ICASSP 2015)
- "Room-Aware Speaker Recognition" (Interspeech 2018)
- "Direct-to-Reverberant Ratio for Distance Estimation" (IEEE 2012)
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq


class SpatialLocationFeatures:
    """
    Extract spatial/location features from audio
    Works with SINGLE microphone (no array needed!)
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def extract_all_features(self, audio_data):
        """
        Extract comprehensive spatial features
        
        Returns:
            Dictionary of spatial features
        """
        if audio_data.dtype == np.int16:
            audio = audio_data.astype(np.float32) / 32768.0
        else:
            audio = audio_data.astype(np.float32)
            
        features = {}
        
        # Feature 1: Direct-to-Reverberant Ratio (DRR)
        features['drr'] = self._calculate_drr(audio)
        
        # Feature 2: Spectral Centroid & Rolloff (distance-dependent)
        features['spectral_centroid'] = self._spectral_centroid(audio)
        features['spectral_rolloff'] = self._spectral_rolloff(audio)
        
        # Feature 3: High-Frequency Energy Ratio (HF attenuates with distance)
        features['hf_ratio'] = self._high_frequency_ratio(audio)
        
        # Feature 4: Reverberation Time Estimate
        features['rt60_estimate'] = self._estimate_rt60(audio)
        
        # Feature 5: SNR Pattern (consistent for fixed position)
        features['snr_pattern'] = self._snr_pattern(audio)
        
        # Create spatial fingerprint vector
        spatial_vector = np.array([
            features['drr'],
            features['spectral_centroid'] / 4000.0,  # Normalize
            features['spectral_rolloff'] / 8000.0,
            features['hf_ratio'],
            features['rt60_estimate'] / 0.5,  # Normalize
            features['snr_pattern']
        ])
        
        # Normalize to unit length
        spatial_vector = spatial_vector / (np.linalg.norm(spatial_vector) + 1e-10)
        
        features['spatial_vector'] = spatial_vector
        
        return features
        
    def _calculate_drr(self, audio):
        """
        Calculate Direct-to-Reverberant Ratio
        
        Higher DRR = closer to microphone (more direct sound)
        Lower DRR = farther or more reverberant room
        
        Method: Compare early energy (direct + early reflections) to late energy (reverberation)
        """
        # Calculate envelope (energy over time)
        envelope = np.abs(scipy_signal.hilbert(audio))
        
        # Smooth envelope
        window_size = int(0.01 * self.sample_rate)  # 10ms
        smoothed = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Find peak (likely direct sound arrival)
        peak_idx = np.argmax(smoothed)
        
        # Direct energy: Within 50ms of peak
        direct_window = int(0.05 * self.sample_rate)
        direct_start = max(0, peak_idx - direct_window//2)
        direct_end = min(len(smoothed), peak_idx + direct_window//2)
        direct_energy = np.sum(smoothed[direct_start:direct_end] ** 2)
        
        # Reverberant energy: 50-200ms after peak
        reverb_start = peak_idx + direct_window
        reverb_end = min(len(smoothed), peak_idx + int(0.2 * self.sample_rate))
        if reverb_end > reverb_start:
            reverb_energy = np.sum(smoothed[reverb_start:reverb_end] ** 2)
        else:
            reverb_energy = 0.0
            
        # DRR in dB
        if reverb_energy > 0:
            drr_db = 10 * np.log10((direct_energy + 1e-10) / (reverb_energy + 1e-10))
        else:
            drr_db = 20.0  # High DRR (very close, no reverb)
            
        # Normalize to 0-1 range (typical: -5 to +15 dB)
        drr_normalized = (drr_db + 5) / 20.0
        drr_normalized = np.clip(drr_normalized, 0.0, 1.0)
        
        return drr_normalized
        
    def _spectral_centroid(self, audio):
        """
        Calculate spectral centroid (center of mass of spectrum)
        
        Centroid shifts lower with distance (high freq attenuated by air)
        """
        # FFT
        fft = rfft(audio)
        magnitude = np.abs(fft)
        freqs = rfftfreq(len(audio), 1/self.sample_rate)
        
        # Centroid = weighted average frequency
        centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        
        return centroid
        
    def _spectral_rolloff(self, audio, percentile=0.85):
        """
        Frequency below which 85% of spectral energy is contained
        
        Lower rolloff = farther (high frequencies absorbed)
        Higher rolloff = closer (high frequencies preserved)
        """
        fft = rfft(audio)
        magnitude = np.abs(fft) ** 2
        freqs = rfftfreq(len(audio), 1/self.sample_rate)
        
        # Cumulative energy
        cumsum = np.cumsum(magnitude)
        total_energy = cumsum[-1]
        
        # Find frequency where cumsum reaches 85% of total
        threshold = percentile * total_energy
        rolloff_idx = np.where(cumsum >= threshold)[0]
        
        if len(rolloff_idx) > 0:
            rolloff_freq = freqs[rolloff_idx[0]]
        else:
            rolloff_freq = freqs[-1]
            
        return rolloff_freq
        
    def _high_frequency_ratio(self, audio):
        """
        Ratio of high-frequency energy to total energy
        
        Higher ratio = closer (HF preserved)
        Lower ratio = farther (HF attenuated)
        """
        fft = rfft(audio)
        magnitude = np.abs(fft) ** 2
        freqs = rfftfreq(len(audio), 1/self.sample_rate)
        
        # High freq: >2000 Hz
        hf_mask = freqs > 2000
        hf_energy = np.sum(magnitude[hf_mask])
        total_energy = np.sum(magnitude)
        
        hf_ratio = hf_energy / (total_energy + 1e-10)
        
        return hf_ratio
        
    def _estimate_rt60(self, audio):
        """
        Estimate RT60 (reverberation time)
        
        Time for sound to decay 60 dB
        Characteristic of room and position
        """
        # Calculate energy decay curve
        envelope = np.abs(scipy_signal.hilbert(audio))
        envelope_db = 20 * np.log10(envelope + 1e-10)
        
        # Find peak
        peak_idx = np.argmax(envelope_db)
        peak_db = envelope_db[peak_idx]
        
        # Find where it drops 60 dB
        target_db = peak_db - 60
        
        decay_indices = np.where(envelope_db[peak_idx:] < target_db)[0]
        
        if len(decay_indices) > 0:
            decay_samples = decay_indices[0]
            rt60 = decay_samples / self.sample_rate
        else:
            rt60 = 0.3  # Typical for small room
            
        # Clip to reasonable range
        rt60 = np.clip(rt60, 0.05, 1.0)
        
        return rt60
        
    def _snr_pattern(self, audio):
        """
        SNR pattern (ratio of speech to background)
        
        Fixed position = consistent SNR
        Moving speaker = variable SNR
        """
        # Split into frames
        frame_size = int(0.03 * self.sample_rate)  # 30ms
        num_frames = len(audio) // frame_size
        
        frame_energies = []
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio[start:end]
            energy = np.sqrt(np.mean(frame ** 2))
            frame_energies.append(energy)
            
        if not frame_energies:
            return 0.5
            
        # Noise floor = bottom 20%
        sorted_energies = sorted(frame_energies)
        noise_floor = np.mean(sorted_energies[:max(1, len(sorted_energies)//5)])
        
        # Signal = top 20%
        signal_level = np.mean(sorted_energies[-len(sorted_energies)//5:])
        
        # SNR
        if noise_floor > 0:
            snr = signal_level / noise_floor
        else:
            snr = 10.0
            
        # Normalize to 0-1 (typical: 2-10)
        snr_normalized = (snr - 2) / 8.0
        snr_normalized = np.clip(snr_normalized, 0.0, 1.0)
        
        return snr_normalized


class LocationAwareVerifier:
    """
    Enhanced verifier that uses BOTH voice AND location features
    
    For fixed-position speakers:
    - Voice embedding should match (WHO)
    - Spatial features should match (WHERE)
    
    For passersby/moving speakers:
    - Voice might be similar (another male)
    - Spatial features will differ (different location)
    → REJECT!
    """
    
    def __init__(self, base_verifier, spatial_weight=0.15):
        """
        Args:
            base_verifier: SimpleRobustVerifier instance
            spatial_weight: How much to weight spatial features (0-1)
                           0.15 = 15% spatial, 85% voice (recommended)
        """
        self.base_verifier = base_verifier
        self.spatial_weight = spatial_weight
        self.spatial_extractor = SpatialLocationFeatures()
        
        # Will store spatial fingerprints during enrollment
        self.spatial_profiles = {}  # {speaker_key: spatial_vector}
        
    def enroll_spatial_profile(self, speaker_key, audio_chunks):
        """
        Create spatial fingerprint from enrollment chunks
        
        Args:
            speaker_key: Speaker identifier
            audio_chunks: List of audio chunks from enrollment
        """
        print(f"   📍 Creating spatial location fingerprint for {speaker_key}...")
        
        # Extract spatial features from each chunk
        spatial_vectors = []
        
        for chunk in audio_chunks:
            features = self.spatial_extractor.extract_all_features(chunk)
            spatial_vectors.append(features['spatial_vector'])
            
        # Average spatial vector (location fingerprint)
        spatial_fingerprint = np.mean(spatial_vectors, axis=0)
        spatial_fingerprint = spatial_fingerprint / (np.linalg.norm(spatial_fingerprint) + 1e-10)
        
        # Calculate consistency
        spatial_std = np.std(np.array(spatial_vectors), axis=0).mean()
        
        self.spatial_profiles[speaker_key] = {
            'spatial_vector': spatial_fingerprint,
            'spatial_std': spatial_std,
            'features_history': spatial_vectors
        }
        
        print(f"      DRR: {features['drr']:.2f}, HF ratio: {features['hf_ratio']:.2f}")
        print(f"      Spatial consistency: {1.0/(1.0+spatial_std*10):.1%}")
        
    def verify_with_location(self, test_embedding, test_audio, enrolled_speakers):
        """
        Verify speaker using BOTH voice and location
        
        Args:
            test_embedding: Voice embedding (256-D)
            test_audio: Raw audio for spatial analysis
            enrolled_speakers: Dict of enrolled speaker profiles
            
        Returns:
            (accept, speaker_key, speaker_name, combined_score, details)
        """
        # === VOICE VERIFICATION ===
        voice_accept, speaker_key, speaker_name, voice_similarity, voice_reason = self.base_verifier.verify_speaker(
            test_embedding,
            enrolled_speakers,
            audio_quality=0.8
        )
        
        # If voice strongly rejects, don't bother with spatial
        if voice_similarity < 0.55:
            return False, speaker_key, speaker_name, voice_similarity, f"Voice rejected: {voice_reason}"
            
        # === SPATIAL VERIFICATION ===
        if speaker_key in self.spatial_profiles:
            # Extract spatial features from test audio
            test_spatial_features = self.spatial_extractor.extract_all_features(test_audio)
            test_spatial_vector = test_spatial_features['spatial_vector']
            
            # Compare with enrolled spatial fingerprint
            enrolled_spatial = self.spatial_profiles[speaker_key]['spatial_vector']
            
            # Spatial similarity (cosine similarity of spatial features)
            spatial_similarity = np.dot(test_spatial_vector, enrolled_spatial)
            
            # === COMBINED SCORE ===
            # Weight voice more heavily (85%), spatial as confirmation (15%)
            combined_score = (
                (1 - self.spatial_weight) * voice_similarity +
                self.spatial_weight * spatial_similarity
            )
            
            details = {
                'voice_similarity': voice_similarity,
                'spatial_similarity': spatial_similarity,
                'combined_score': combined_score,
                'voice_reason': voice_reason,
                'drr_test': test_spatial_features['drr'],
                'drr_enrolled': self.spatial_profiles[speaker_key].get('avg_drr', 0),
            }
            
            # Enhanced decision logic
            # Both voice AND spatial must be reasonable
            
            # Strict spatial check: should be >0.70 for fixed position
            if spatial_similarity < 0.70:
                # Spatial mismatch - different location!
                return False, speaker_key, speaker_name, combined_score, \
                       f"Spatial mismatch (voice:{voice_similarity:.2f}, spatial:{spatial_similarity:.2f}) - likely different location"
                       
            # Voice passed basic check, spatial matches - combine scores
            threshold = 0.62  # Slightly lower than voice-only (spatial helps)
            
            if combined_score >= threshold:
                if voice_accept:
                    return True, speaker_key, speaker_name, combined_score, \
                           f"Accepted (voice:{voice_similarity:.2f}, spatial:{spatial_similarity:.2f})"
                else:
                    # Voice was borderline but spatial helps
                    if spatial_similarity >= 0.85:  # Strong spatial match
                        return True, speaker_key, speaker_name, combined_score, \
                               f"Accepted via spatial boost (voice:{voice_similarity:.2f}→{combined_score:.2f})"
                    else:
                        return False, speaker_key, speaker_name, combined_score, voice_reason
            else:
                return False, speaker_key, speaker_name, combined_score, \
                       f"Combined score too low ({combined_score:.2f} < {threshold:.2f})"
                       
        else:
            # No spatial profile - fall back to voice only
            return voice_accept, speaker_key, speaker_name, voice_similarity, voice_reason
            
    def calculate_spatial_similarity(self, spatial_vec1, spatial_vec2):
        """Calculate similarity between spatial fingerprints"""
        return np.dot(spatial_vec1, spatial_vec2)


def save_spatial_profiles(location_verifier, filepath="spatial_profiles.pkl"):
    """Save spatial profiles to disk"""
    import pickle
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(location_verifier.spatial_profiles, f)
        print(f"💾 Spatial profiles saved to {filepath}")
    except Exception as e:
        print(f"Error saving spatial profiles: {e}")


def load_spatial_profiles(location_verifier, filepath="spatial_profiles.pkl"):
    """Load spatial profiles from disk"""
    import pickle
    import os
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                location_verifier.spatial_profiles = pickle.load(f)
            print(f"📂 Loaded spatial profiles from {filepath}")
            return True
    except Exception as e:
        print(f"Error loading spatial profiles: {e}")
    return False

