"""
Stress-Invariant Voice Features for Interrogation Scenarios

Problem: Voice changes under stress/emotion:
- Pitch increases 20-40 Hz (fear/anxiety)
- Voice tremor (jitter) increases
- Speaking rate changes
- Energy fluctuations

Solution: Normalize stress-sensitive parameters BEFORE embedding extraction

Based on research:
- "Emotion-Invariant Speaker Recognition" (Interspeech 2019)
- "Robust Speaker Verification Under Stress" (IEEE 2017)
- "Pitch Normalization for Forensic Speaker ID" (2018)
"""

import numpy as np
from scipy import signal as scipy_signal
import librosa


class StressInvariantProcessor:
    """
    Preprocess audio to be invariant to emotional stress
    
    Techniques:
    1. Pitch normalization (remove F0 variations)
    2. Energy normalization (consistent loudness)
    3. Speaking rate normalization (tempo adjustment)
    4. Formant-based features (less affected by stress)
    """
    
    def __init__(self, target_pitch=150, target_rms=1000):
        """
        Args:
            target_pitch: Target fundamental frequency (Hz)
                         Male: 100-150 Hz, Female: 180-250 Hz
                         Use 150 as neutral
            target_rms: Target RMS energy level
        """
        self.target_pitch = target_pitch
        self.target_rms = target_rms
        
    def normalize_audio(self, audio_data, sample_rate=16000):
        """
        Normalize audio for stress invariance
        
        Args:
            audio_data: Raw audio (int16 or float32)
            sample_rate: Sample rate
            
        Returns:
            Normalized audio (float32)
        """
        # Convert to float
        if audio_data.dtype == np.int16:
            audio = audio_data.astype(np.float32) / 32768.0
        else:
            audio = audio_data.astype(np.float32)
            
        # Step 1: Pitch normalization
        audio_pitch_norm = self._normalize_pitch(audio, sample_rate)
        
        # Step 2: Energy normalization  
        audio_energy_norm = self._normalize_energy(audio_pitch_norm)
        
        # Step 3: Gentle noise gate (remove very quiet parts)
        audio_gated = self._noise_gate(audio_energy_norm)
        
        return audio_gated
        
    def _normalize_pitch(self, audio, sample_rate):
        """
        Normalize pitch to target F0
        
        Method: Time-domain pitch shifting using phase vocoder
        """
        try:
            # Estimate current pitch
            current_pitch = self._estimate_pitch(audio, sample_rate)
            
            if current_pitch < 50 or current_pitch > 500:
                # Invalid pitch estimate, skip normalization
                return audio
                
            # Calculate shift ratio
            shift_ratio = self.target_pitch / current_pitch
            
            # Don't shift if already close
            if 0.95 <= shift_ratio <= 1.05:
                return audio
                
            # Limit shift to reasonable range (avoid artifacts)
            shift_ratio = np.clip(shift_ratio, 0.85, 1.15)
            
            # Pitch shift using librosa
            audio_shifted = librosa.effects.pitch_shift(
                audio, sr=sample_rate, n_steps=12*np.log2(shift_ratio)
            )
            
            return audio_shifted
            
        except Exception as e:
            # If pitch shift fails, return original
            return audio
            
    def _estimate_pitch(self, audio, sample_rate):
        """
        Estimate fundamental frequency (F0)
        
        Method: Autocorrelation
        """
        # Autocorrelation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find first peak after zero lag
        # Corresponding to fundamental period
        
        # Search range: 50-500 Hz → periods of 2-320 samples at 16kHz
        min_period = int(sample_rate / 500)  # 32 samples
        max_period = int(sample_rate / 50)   # 320 samples
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(min_period, min(max_period, len(autocorr)-1)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append((i, autocorr[i]))
                
        if peaks:
            # First (strongest) peak corresponds to pitch period
            pitch_period = max(peaks, key=lambda x: x[1])[0]
            pitch = sample_rate / pitch_period
            return pitch
        else:
            return self.target_pitch  # Default if no pitch found
            
    def _normalize_energy(self, audio):
        """Normalize RMS energy to target level"""
        current_rms = np.sqrt(np.mean(audio ** 2))
        
        if current_rms < 1e-6:
            return audio
            
        gain = self.target_rms / (current_rms * 32768.0)
        
        # Limit gain to prevent excessive amplification
        gain = np.clip(gain, 0.5, 2.0)
        
        normalized = audio * gain
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val * 0.95
            
        return normalized
        
    def _noise_gate(self, audio, threshold_db=-40):
        """
        Simple noise gate to remove very quiet background
        
        Args:
            threshold_db: Gate threshold in dB below peak
        """
        # Calculate envelope
        envelope = np.abs(scipy_signal.hilbert(audio))
        
        # Smooth envelope
        window_size = int(0.01 * 16000)  # 10ms
        envelope_smooth = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Calculate threshold
        peak = np.max(envelope_smooth)
        threshold_linear = peak * 10 ** (threshold_db / 20)
        
        # Apply gate (soft knee)
        gate = np.minimum(1.0, envelope_smooth / threshold_linear)
        
        # Apply to audio
        gated = audio * gate
        
        return gated


def calculate_whisper_confidence(whisper_result):
    """
    Calculate confidence score from Whisper output
    
    Whisper doesn't provide explicit confidence, so we estimate from:
    - no_speech_prob (lower is better)
    - avg_logprob (higher is better)
    - compression_ratio (closer to 1 is better)
    """
    # Extract metrics
    no_speech_prob = whisper_result.get('no_speech_prob', 0.5)
    avg_logprob = whisper_result.get('avg_logprob', -1.0)
    compression_ratio = whisper_result.get('compression_ratio', 1.5)
    
    # Calculate confidence components
    
    # 1. Speech confidence (inverse of no_speech_prob)
    speech_conf = 1.0 - no_speech_prob
    
    # 2. Log probability confidence
    # avg_logprob ranges from -∞ to 0
    # Typical: -0.2 to -1.0
    # Map to 0-1
    logprob_conf = np.exp(avg_logprob / 2)  # -1.0 → 0.61, -0.2 → 0.90
    
    # 3. Compression ratio confidence
    # Ideal: 1.0-2.0
    # Too high (>3): repetitive/hallucinatin
    # Too low (<0.8): missing words
    if 0.8 <= compression_ratio <= 2.0:
        compression_conf = 1.0
    elif compression_ratio > 2.0:
        compression_conf = max(0.0, 1.0 - (compression_ratio - 2.0) / 3.0)
    else:
        compression_conf = max(0.0, compression_ratio / 0.8)
        
    # Combined confidence (weighted average)
    confidence = (
        0.50 * speech_conf +
        0.35 * logprob_conf +
        0.15 * compression_conf
    )
    
    return float(np.clip(confidence, 0.0, 1.0))

