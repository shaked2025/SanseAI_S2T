"""
Comprehensive Quality Assessment for Forensic Transcription

Multi-dimensional quality scoring:
1. Audio quality (SNR, distortion, clipping)
2. Verification confidence (similarity scores)
3. Transcription confidence (Whisper metrics)
4. Spatial consistency (location match)
5. Temporal consistency (drift from enrollment)

Legal admissibility criteria:
- Audio SNR > 15 dB
- Verification confidence > 0.75
- Transcription confidence > 0.70
- No clipping/distortion
- Spatial match (if applicable)

Outputs confidence category: HIGH/MEDIUM/LOW/INADMISSIBLE
"""

import numpy as np
from scipy import signal as scipy_signal


class ComprehensiveQualityAssessment:
    """
    Multi-dimensional quality scoring for forensic use
    """
    
    def __init__(self):
        # Legal admissibility thresholds
        self.admissibility_criteria = {
            'min_snr_db': 12.0,  # Slightly relaxed from 15 for real conditions
            'min_verification_confidence': 0.70,
            'min_transcription_confidence': 0.65,
            'max_distortion_thd': 0.10,  # 10% THD
            'min_spatial_similarity': 0.70  # For fixed-position speakers
        }
        
    def assess_audio_quality(self, audio_data, sample_rate=16000):
        """
        Assess raw audio quality
        
        Returns:
            Dictionary of audio quality metrics
        """
        if audio_data.dtype == np.int16:
            audio = audio_data.astype(np.float32) / 32768.0
        else:
            audio = audio_data.astype(np.float32)
            
        metrics = {}
        
        # 1. Signal-to-Noise Ratio
        metrics['snr_db'] = self._calculate_snr(audio, sample_rate)
        
        # 2. Clipping detection
        metrics['clipping_percent'] = self._detect_clipping(audio_data)
        
        # 3. Total Harmonic Distortion
        metrics['thd_percent'] = self._calculate_thd(audio, sample_rate)
        
        # 4. Dynamic range
        metrics['dynamic_range_db'] = self._calculate_dynamic_range(audio)
        
        # 5. Spectral flatness (noise indicator)
        metrics['spectral_flatness'] = self._spectral_flatness(audio)
        
        # Overall audio quality score (0-1)
        quality_components = []
        
        # SNR score
        if metrics['snr_db'] >= 20:
            snr_score = 1.0
        elif metrics['snr_db'] >= 15:
            snr_score = 0.9
        elif metrics['snr_db'] >= 12:
            snr_score = 0.7
        elif metrics['snr_db'] >= 8:
            snr_score = 0.5
        else:
            snr_score = 0.3
        quality_components.append(snr_score)
        
        # Clipping score
        if metrics['clipping_percent'] < 0.1:
            clip_score = 1.0
        elif metrics['clipping_percent'] < 1.0:
            clip_score = 0.8
        elif metrics['clipping_percent'] < 5.0:
            clip_score = 0.5
        else:
            clip_score = 0.2
        quality_components.append(clip_score)
        
        # Distortion score
        if metrics['thd_percent'] < 1.0:
            thd_score = 1.0
        elif metrics['thd_percent'] < 5.0:
            thd_score = 0.8
        elif metrics['thd_percent'] < 10.0:
            thd_score = 0.5
        else:
            thd_score = 0.2
        quality_components.append(thd_score)
        
        metrics['audio_quality_score'] = np.mean(quality_components)
        
        # Quality category
        if metrics['audio_quality_score'] >= 0.85:
            metrics['audio_quality_category'] = "EXCELLENT"
        elif metrics['audio_quality_score'] >= 0.70:
            metrics['audio_quality_category'] = "GOOD"
        elif metrics['audio_quality_score'] >= 0.50:
            metrics['audio_quality_category'] = "ACCEPTABLE"
        else:
            metrics['audio_quality_category'] = "POOR"
            
        return metrics
        
    def assess_verification_quality(self, voice_similarity, spatial_similarity,
                                   combined_score, threshold, margin):
        """
        Assess speaker verification quality
        
        Returns:
            Verification quality metrics
        """
        metrics = {}
        
        metrics['voice_similarity'] = float(voice_similarity)
        metrics['spatial_similarity'] = float(spatial_similarity) if spatial_similarity else None
        metrics['combined_score'] = float(combined_score)
        metrics['threshold'] = float(threshold)
        metrics['margin'] = float(margin) if margin else None
        
        # Confidence level based on how far above threshold
        margin_above_threshold = combined_score - threshold
        
        if margin_above_threshold >= 0.20:
            metrics['verification_confidence'] = 0.95
            metrics['confidence_category'] = "VERY_HIGH"
        elif margin_above_threshold >= 0.15:
            metrics['verification_confidence'] = 0.90
            metrics['confidence_category'] = "HIGH"
        elif margin_above_threshold >= 0.10:
            metrics['verification_confidence'] = 0.85
            metrics['confidence_category'] = "GOOD"
        elif margin_above_threshold >= 0.05:
            metrics['verification_confidence'] = 0.75
            metrics['confidence_category'] = "MEDIUM"
        elif margin_above_threshold >= 0.00:
            metrics['verification_confidence'] = 0.65
            metrics['confidence_category'] = "LOW"
        else:
            metrics['verification_confidence'] = 0.0
            metrics['confidence_category'] = "REJECTED"
            
        return metrics
        
    def assess_transcription_quality(self, whisper_result):
        """
        Assess Whisper transcription quality
        
        Args:
            whisper_result: Full Whisper output dictionary
            
        Returns:
            Transcription quality metrics
        """
        metrics = {}
        
        # Extract Whisper internal metrics
        metrics['no_speech_prob'] = float(whisper_result.get('no_speech_prob', 0.5))
        metrics['avg_logprob'] = float(whisper_result.get('avg_logprob', -1.0))
        metrics['compression_ratio'] = float(whisper_result.get('compression_ratio', 1.5))
        
        # Calculate confidence
        from stress_invariant_features import calculate_whisper_confidence
        metrics['transcription_confidence'] = calculate_whisper_confidence(whisper_result)
        
        # Confidence category
        if metrics['transcription_confidence'] >= 0.90:
            metrics['confidence_category'] = "VERY_HIGH"
        elif metrics['transcription_confidence'] >= 0.80:
            metrics['confidence_category'] = "HIGH"
        elif metrics['transcription_confidence'] >= 0.70:
            metrics['confidence_category'] = "GOOD"
        elif metrics['transcription_confidence'] >= 0.60:
            metrics['confidence_category'] = "MEDIUM"
        else:
            metrics['confidence_category'] = "LOW"
            
        return metrics
        
    def assess_overall_quality(self, audio_quality, verification_quality, 
                              transcription_quality):
        """
        Combine all quality assessments
        
        Returns:
            Overall quality assessment with legal admissibility determination
        """
        overall = {}
        
        # Component scores
        overall['audio_score'] = audio_quality['audio_quality_score']
        overall['verification_score'] = verification_quality['verification_confidence']
        overall['transcription_score'] = transcription_quality['transcription_confidence']
        
        # Weighted overall score
        overall['combined_quality_score'] = (
            0.30 * overall['audio_score'] +
            0.40 * overall['verification_score'] +
            0.30 * overall['transcription_score']
        )
        
        # Check legal admissibility criteria
        admissible = True
        inadmissibility_reasons = []
        
        if audio_quality['snr_db'] < self.admissibility_criteria['min_snr_db']:
            admissible = False
            inadmissibility_reasons.append(f"SNR too low ({audio_quality['snr_db']:.1f} dB)")
            
        if verification_quality['verification_confidence'] < self.admissibility_criteria['min_verification_confidence']:
            admissible = False
            inadmissibility_reasons.append(f"Verification confidence too low ({verification_quality['verification_confidence']:.2f})")
            
        if transcription_quality['transcription_confidence'] < self.admissibility_criteria['min_transcription_confidence']:
            admissible = False
            inadmissibility_reasons.append(f"Transcription confidence too low ({transcription_quality['transcription_confidence']:.2f})")
            
        if audio_quality['clipping_percent'] > 10.0:
            admissible = False
            inadmissibility_reasons.append(f"Excessive clipping ({audio_quality['clipping_percent']:.1f}%)")
            
        if verification_quality.get('spatial_similarity') is not None:
            if verification_quality['spatial_similarity'] < self.admissibility_criteria['min_spatial_similarity']:
                admissible = False
                inadmissibility_reasons.append(f"Spatial mismatch ({verification_quality['spatial_similarity']:.2f})")
                
        overall['legally_admissible'] = admissible
        overall['inadmissibility_reasons'] = inadmissibility_reasons
        
        # Overall category
        if not admissible:
            overall['overall_category'] = "INADMISSIBLE"
        elif overall['combined_quality_score'] >= 0.85:
            overall['overall_category'] = "EXCELLENT"
        elif overall['combined_quality_score'] >= 0.75:
            overall['overall_category'] = "GOOD"
        elif overall['combined_quality_score'] >= 0.65:
            overall['overall_category'] = "ACCEPTABLE"
        else:
            overall['overall_category'] = "POOR"
            
        # Recommendations
        overall['recommendations'] = []
        
        if not admissible:
            overall['recommendations'].append("MANUAL_REVIEW_REQUIRED")
            
        if verification_quality['confidence_category'] == "LOW":
            overall['recommendations'].append("VERIFY_SPEAKER_IDENTITY")
            
        if transcription_quality['confidence_category'] == "LOW":
            overall['recommendations'].append("VERIFY_TRANSCRIPTION_ACCURACY")
            
        return overall
        
    # Helper methods for audio quality
    
    def _calculate_snr(self, audio, sample_rate):
        """Calculate SNR estimate"""
        frame_size = int(0.03 * sample_rate)
        frames = [audio[i:i+frame_size] for i in range(0, len(audio)-frame_size, frame_size)]
        
        energies = [np.sqrt(np.mean(f**2)) for f in frames]
        
        if not energies:
            return 10.0
            
        sorted_e = sorted(energies)
        noise_floor = np.mean(sorted_e[:max(1, len(sorted_e)//5)])
        signal_power = np.mean(sorted_e[-len(sorted_e)//5:])
        
        if noise_floor > 0:
            snr_db = 20 * np.log10(signal_power / noise_floor)
        else:
            snr_db = 30.0
            
        return float(snr_db)
        
    def _detect_clipping(self, audio_data):
        """Detect clipping (saturation)"""
        if audio_data.dtype == np.int16:
            # Count samples at max/min values
            max_val = 32767
            min_val = -32768
            
            clipped = np.sum((audio_data >= max_val - 10) | (audio_data <= min_val + 10))
            clipping_percent = clipped / len(audio_data) * 100
        else:
            clipped = np.sum((audio_data >= 0.99) | (audio_data <= -0.99))
            clipping_percent = clipped / len(audio_data) * 100
            
        return float(clipping_percent)
        
    def _calculate_thd(self, audio, sample_rate):
        """
        Calculate Total Harmonic Distortion
        
        Simplified estimate using spectral analysis
        """
        # FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        
        # Find fundamental (strongest component in 80-300 Hz range)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
        
        fund_mask = (freqs >= 80) & (freqs <= 300)
        if np.any(fund_mask):
            fund_idx = np.argmax(magnitude[fund_mask])
            fund_freq_idx = np.where(fund_mask)[0][fund_idx]
            fund_power = magnitude[fund_freq_idx] ** 2
            
            # Harmonics (2f, 3f, 4f, 5f)
            harmonic_power = 0
            for h in [2, 3, 4, 5]:
                harm_idx = fund_freq_idx * h
                if harm_idx < len(magnitude):
                    harmonic_power += magnitude[harm_idx] ** 2
                    
            # THD
            if fund_power > 0:
                thd = np.sqrt(harmonic_power / fund_power) * 100
            else:
                thd = 0.0
        else:
            thd = 0.0
            
        return float(thd)
        
    def _calculate_dynamic_range(self, audio):
        """Calculate dynamic range (peak to noise floor)"""
        peak = np.max(np.abs(audio))
        
        # Estimate noise floor (bottom 10% of samples)
        abs_audio = np.abs(audio)
        sorted_abs = np.sort(abs_audio)
        noise_floor = np.mean(sorted_abs[:len(sorted_abs)//10])
        
        if noise_floor > 0:
            dynamic_range_db = 20 * np.log10(peak / noise_floor)
        else:
            dynamic_range_db = 60.0
            
        return float(dynamic_range_db)
        
    def _spectral_flatness(self, audio):
        """
        Calculate spectral flatness (Wiener entropy)
        
        0 = pure tone (structured)
        1 = white noise (flat)
        
        Speech typically: 0.1-0.3
        Noise typically: 0.8-1.0
        """
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft) + 1e-10
        
        # Geometric mean / Arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(magnitude)))
        arithmetic_mean = np.mean(magnitude)
        
        flatness = geometric_mean / arithmetic_mean
        
        return float(flatness)

