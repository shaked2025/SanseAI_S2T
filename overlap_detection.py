"""
Overlapping Speech Detection
Identifies when multiple speakers are talking simultaneously
"""

import numpy as np
from scipy import signal
from collections import deque


class OverlappingSpeechDetector:
    """
    Detects when multiple speakers are talking at the same time
    Uses energy-based and spectral analysis
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def detect_overlap(self, audio_data, sample_rate=16000):
        """
        Detect if multiple speakers are present in audio
        
        Args:
            audio_data: Audio numpy array
            sample_rate: Sample rate
            
        Returns:
            Dictionary with overlap information
        """
        try:
            # Convert to float
            if audio_data.dtype == np.int16:
                audio = audio_data.astype(np.float32) / 32768.0
            else:
                audio = audio_data.astype(np.float32)
                
            # Calculate energy in different frequency bands
            # Overlapping speech shows energy in multiple frequency regions
            
            # Low frequency (male voices typically 85-180 Hz)
            low_energy = self._band_energy(audio, sample_rate, 50, 300)
            
            # Mid frequency (female voices typically 165-255 Hz)
            mid_energy = self._band_energy(audio, sample_rate, 200, 500)
            
            # High frequency (consonants, clarity)
            high_energy = self._band_energy(audio, sample_rate, 500, 2000)
            
            # Calculate spectral flux (change in spectrum)
            spectral_flux = self._spectral_flux(audio, sample_rate)
            
            # Calculate zero-crossing rate variability
            zcr_var = self._zcr_variability(audio, sample_rate)
            
            # Overlap indicators:
            # 1. High energy in multiple bands (different pitches)
            multi_band = (low_energy > 0.3 and mid_energy > 0.3) or (mid_energy > 0.3 and high_energy > 0.4)
            
            # 2. High spectral flux (rapid spectrum changes from multiple voices)
            high_flux = spectral_flux > 0.15
            
            # 3. High ZCR variability (multiple pitch sources)
            high_zcr_var = zcr_var > 0.08
            
            # Combine indicators
            overlap_score = 0.0
            if multi_band:
                overlap_score += 0.4
            if high_flux:
                overlap_score += 0.3
            if high_zcr_var:
                overlap_score += 0.3
                
            is_overlap = overlap_score > 0.5
            
            return {
                'is_overlap': is_overlap,
                'overlap_score': overlap_score,
                'low_energy': low_energy,
                'mid_energy': mid_energy,
                'high_energy': high_energy,
                'spectral_flux': spectral_flux,
                'zcr_variability': zcr_var
            }
            
        except Exception as e:
            print(f"Error detecting overlap: {e}")
            return {'is_overlap': False, 'overlap_score': 0.0}
            
    def _band_energy(self, audio, sample_rate, low_freq, high_freq):
        """Calculate energy in frequency band"""
        # FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
        
        # Find indices for band
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        
        # Calculate energy in band
        band_energy = np.sum(np.abs(fft[band_mask]) ** 2)
        total_energy = np.sum(np.abs(fft) ** 2) + 1e-10
        
        return band_energy / total_energy
        
    def _spectral_flux(self, audio, sample_rate, frame_size=2048, hop_size=512):
        """Calculate spectral flux (rate of spectral change)"""
        # Compute STFT
        f, t, Zxx = signal.stft(audio, sample_rate, nperseg=frame_size, noverlap=frame_size-hop_size)
        
        # Magnitude spectrum
        magnitude = np.abs(Zxx)
        
        # Spectral flux between consecutive frames
        flux = np.sum(np.abs(np.diff(magnitude, axis=1)), axis=0)
        
        # Average flux
        return np.mean(flux) / (np.max(flux) + 1e-10)
        
    def _zcr_variability(self, audio, sample_rate, frame_size=2048, hop_size=512):
        """Calculate zero-crossing rate variability"""
        # Split into frames
        num_frames = (len(audio) - frame_size) // hop_size + 1
        zcr_values = []
        
        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_size
            frame = audio[start:end]
            
            # Calculate ZCR for frame
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            zcr_values.append(zcr)
            
        # Variability = standard deviation of ZCR
        return np.std(zcr_values) if zcr_values else 0.0


class MultiSpeakerIdentifier:
    """
    Identifies multiple speakers in overlapping speech segments
    """
    
    def __init__(self, verification_engine, overlap_detector):
        self.verification_engine = verification_engine
        self.overlap_detector = overlap_detector
        
    def identify_speakers(self, audio_data, sample_rate=16000):
        """
        Identify all speakers in audio segment (may be multiple if overlapping)
        
        Args:
            audio_data: Audio numpy array
            sample_rate: Sample rate
            
        Returns:
            List of (speaker_key, speaker_name, confidence) tuples
        """
        # First check for overlap
        overlap_info = self.overlap_detector.detect_overlap(audio_data, sample_rate)
        
        if not overlap_info['is_overlap']:
            # Single speaker - use standard verification
            speaker_key, speaker_name, confidence, metadata = self.verification_engine.verify_speaker(
                audio_data, sample_rate
            )
            return [(speaker_key, speaker_name, confidence, False)]
        else:
            # Overlapping speech detected!
            # Overlap detected - reduced logging
            pass  # Will be shown in main app if needed
            
            # Try to identify multiple speakers
            speakers = self._identify_multiple_speakers(audio_data, sample_rate, overlap_info)
            
            if len(speakers) >= 2:
                return speakers
            else:
                # Fall back to single speaker
                speaker_key, speaker_name, confidence, metadata = self.verification_engine.verify_speaker(
                    audio_data, sample_rate
                )
                return [(speaker_key, speaker_name, confidence, True)]  # Mark as overlap
                
    def _identify_multiple_speakers(self, audio_data, sample_rate, overlap_info):
        """
        Attempt to identify multiple speakers in overlapping segment
        
        Strategy:
        1. Try full segment with each enrolled speaker
        2. Return top 2-3 matches above threshold
        """
        enrolled_speakers = self.verification_engine.enrollment.get_enrolled_speakers()
        
        if not enrolled_speakers:
            return []
            
        # Extract embedding from full segment
        embedding = self.verification_engine.enrollment.embedding_extractor.extract_embedding(
            audio_data, sample_rate
        )
        
        # Calculate similarity with each enrolled speaker
        similarities = {}
        
        for speaker_key, profile in enrolled_speakers.items():
            similarity = np.dot(embedding, profile['mean_embedding'])
            similarities[speaker_key] = similarity
            
        # Get speakers above threshold (likely multiple in overlap)
        speakers = []
        
        # Sort by similarity
        sorted_speakers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 2-3 if they exceed a lower threshold (for overlap)
        overlap_threshold = 0.65  # Lower threshold for overlap detection
        
        for speaker_key, similarity in sorted_speakers[:3]:
            if similarity >= overlap_threshold:
                profile = enrolled_speakers[speaker_key]
                speakers.append((
                    speaker_key,
                    profile['name'],
                    similarity,
                    True  # Mark as overlap
                ))
                
        return speakers if len(speakers) >= 2 else []

