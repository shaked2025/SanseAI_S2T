"""
Audio data augmentation module for enhancing training datasets.
Provides various audio augmentation techniques including time shifting, noise addition,
pitch shifting, speed changes, filtering, and time masking.
"""

import numpy as np
import librosa
import random
from typing import List, Callable
from .config import CONFIG


class AudioAugmenter:
    """
    Class to perform various audio augmentations for data enhancement.
    
    Provides a comprehensive set of audio augmentation techniques that can be applied
    individually or in chains to increase dataset diversity and model robustness.
    """
    
    def __init__(self, sample_rate: int = None):
        """
        Initialize the audio augmenter.
        
        Args:
            sample_rate: Sampling rate of the audio
        """
        self.sample_rate = sample_rate or CONFIG.audio.sample_rate
        
    def time_shift(self, audio: np.ndarray, shift_factor: float = None) -> np.ndarray:
        """
        Shifts the audio in time by a random amount.
        
        Args:
            audio: Audio signal as numpy array
            shift_factor: Fraction of total length to shift (0.0 to 1.0)
        
        Returns:
            Time-shifted audio array
        """
        if shift_factor is None:
            shift_factor = CONFIG.augmentation.time_shift_factor
            
        shift_amount = int(len(audio) * shift_factor)
        if shift_amount > 0:
            direction = np.random.randint(0, 2)
            if direction == 1:
                # Shift right (add silence at beginning)
                augmented = np.concatenate((np.zeros(shift_amount), audio[:-shift_amount]))
            else:
                # Shift left (add silence at end)
                augmented = np.concatenate((audio[shift_amount:], np.zeros(shift_amount)))
            return augmented.copy()  # Return a copy to ensure positive strides
        return audio.copy()
    
    def add_noise(self, audio: np.ndarray, noise_level: float = None) -> np.ndarray:
        """
        Adds random Gaussian noise to the audio signal.
        
        Args:
            audio: Audio signal as numpy array
            noise_level: Level of noise to add (standard deviation)
        
        Returns:
            Audio with added noise
        """
        if noise_level is None:
            noise_level = CONFIG.augmentation.noise_level
            
        noise = np.random.randn(len(audio)) * noise_level
        return (audio + noise).copy()  # Return a copy to ensure positive strides
    
    def change_pitch(self, audio: np.ndarray, pitch_factor: int = None) -> np.ndarray:
        """
        Changes the pitch of the audio by shifting it in semitones.
        
        Args:
            audio: Audio signal as numpy array
            pitch_factor: Maximum pitch shift range in semitones
        
        Returns:
            Pitch-shifted audio array
        """
        if pitch_factor is None:
            pitch_factor = CONFIG.augmentation.pitch_shift_range
            
        steps = np.random.randint(-pitch_factor, pitch_factor)
        try:
            result = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=steps)
            # Make sure result is C-contiguous
            return np.ascontiguousarray(result)
        except Exception as e:
            print(f"Pitch shift failed: {e}, returning original audio")
            return audio.copy()
    
    def change_speed(self, audio: np.ndarray, speed_factor: float = None) -> np.ndarray:
        """
        Changes the speed of the audio without changing pitch.
        
        Args:
            audio: Audio signal as numpy array
            speed_factor: Maximum speed change factor (0.0 to 1.0)
        
        Returns:
            Speed-adjusted audio array
        """
        if speed_factor is None:
            speed_factor = CONFIG.augmentation.speed_change_factor
            
        speed_change = np.random.uniform(1-speed_factor, 1+speed_factor)
        try:
            # Use librosa's time stretch
            augmented = librosa.effects.time_stretch(audio, rate=speed_change)
            # Adjust length to match original if needed
            if len(augmented) > len(audio):
                augmented = augmented[:len(audio)]
            elif len(augmented) < len(audio):
                # Pad with zeros to match length
                augmented = np.pad(augmented, (0, len(audio) - len(augmented)))
            # Make sure result is C-contiguous
            return np.ascontiguousarray(augmented)
        except Exception as e:
            print(f"Speed change failed: {e}, returning original audio")
            return audio.copy()
    
    def apply_filters(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies random filtering to audio (high-pass or low-pass).
        
        Args:
            audio: Audio signal as numpy array
        
        Returns:
            Filtered audio array
        """
        filter_type = np.random.choice(['highpass', 'lowpass'])
        
        try:
            nyquist = self.sample_rate // 2
            if filter_type == 'highpass':
                cutoff = np.random.uniform(0.1, 0.5) * nyquist
                # Simple high-pass filter using FFT
                fft = np.fft.rfft(audio)
                freq = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
                fft[freq < cutoff] = 0
                result = np.fft.irfft(fft, len(audio))
            else:  # lowpass
                cutoff = np.random.uniform(0.5, 0.9) * nyquist
                # Simple low-pass filter using FFT
                fft = np.fft.rfft(audio)
                freq = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
                fft[freq > cutoff] = 0
                result = np.fft.irfft(fft, len(audio))
                
            # Ensure result is C-contiguous
            return np.ascontiguousarray(result)
        except Exception as e:
            print(f"Filter application failed: {e}, returning original audio")
            return audio.copy()
    
    def time_masking(self, audio: np.ndarray, mask_factor: float = None) -> np.ndarray:
        """
        Masks random segments of audio in time by setting them to zero.
        
        Args:
            audio: Audio signal as numpy array
            mask_factor: Maximum portion of audio to mask (0.0 to 1.0)
        
        Returns:
            Audio with time masking applied
        """
        if mask_factor is None:
            mask_factor = CONFIG.augmentation.mask_factor
            
        audio_len = len(audio)
        mask_len = int(audio_len * np.random.uniform(0, mask_factor))
        if mask_len == 0:
            return audio.copy()
            
        mask_start = np.random.randint(0, audio_len - mask_len)
        
        augmented = audio.copy()  # Create a copy to avoid modifying the original
        augmented[mask_start:mask_start+mask_len] = 0
        
        return augmented  # Already a copy
    
    def frequency_masking(self, audio: np.ndarray, mask_factor: float = 0.1) -> np.ndarray:
        """
        Masks random frequency bands in the spectrogram domain.
        
        Args:
            audio: Audio signal as numpy array
            mask_factor: Maximum portion of frequency bands to mask
        
        Returns:
            Audio with frequency masking applied
        """
        try:
            # Convert to spectrogram
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Apply frequency masking
            freq_bins = magnitude.shape[0]
            mask_size = int(freq_bins * np.random.uniform(0, mask_factor))
            if mask_size > 0:
                mask_start = np.random.randint(0, freq_bins - mask_size)
                magnitude[mask_start:mask_start+mask_size, :] = 0
            
            # Convert back to time domain
            masked_stft = magnitude * np.exp(1j * phase)
            result = librosa.istft(masked_stft, length=len(audio))
            
            return np.ascontiguousarray(result)
        except Exception as e:
            print(f"Frequency masking failed: {e}, returning original audio")
            return audio.copy()
    
    def apply_random_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies a single random augmentation technique to the audio.
        
        Args:
            audio: Audio signal as numpy array
        
        Returns:
            Augmented audio array
        """
        augmentation_methods = [
            self.time_shift,
            self.add_noise,
            self.change_pitch,
            self.change_speed,
            self.apply_filters,
            self.time_masking,
            lambda x: x.copy()  # No augmentation (identity)
        ]
        
        chosen_method = np.random.choice(augmentation_methods)
        return chosen_method(audio)
    
    def apply_augmentation_chain(self, audio: np.ndarray, num_augmentations: int = None) -> np.ndarray:
        """
        Applies a chain of random augmentations to the audio.
        
        Args:
            audio: Audio signal as numpy array
            num_augmentations: Number of augmentations to apply in sequence
        
        Returns:
            Augmented audio with positive strides
        """
        if num_augmentations is None:
            num_augmentations = np.random.randint(1, 4)  # Apply 1-3 augmentations by default
        
        # Start with a copy to avoid modifying the original
        augmented = np.ascontiguousarray(audio.copy())
        
        # List of possible augmentations (excluding identity)
        augmentations = [
            self.time_shift,
            self.add_noise,
            self.change_pitch,
            self.change_speed,
            self.apply_filters,
            self.time_masking
        ]
        
        # Randomly select augmentations
        if num_augmentations > len(augmentations):
            num_augmentations = len(augmentations)
            
        selected_augs = random.sample(augmentations, num_augmentations)
        
        # Apply each augmentation, ensuring positive strides after each step
        for aug_func in selected_augs:
            try:
                augmented = aug_func(augmented)
                # Ensure result is C-contiguous (positive strides)
                if not augmented.flags.c_contiguous:
                    augmented = np.ascontiguousarray(augmented)
            except Exception as e:
                print(f"Augmentation failed: {e}, continuing with current audio")
                # If any augmentation fails, make sure we have a contiguous array
                augmented = np.ascontiguousarray(augmented)
                
        return augmented
    
    def apply_stress_specific_augmentation(self, audio: np.ndarray, is_stress: bool = False) -> np.ndarray:
        """
        Applies augmentations tailored for stress detection.
        
        Args:
            audio: Audio signal as numpy array
            is_stress: Whether the audio represents a stress sample
        
        Returns:
            Augmented audio optimized for stress detection
        """
        if is_stress:
            # For stress samples, apply more aggressive augmentations
            # to increase robustness
            stress_augmentations = [
                self.add_noise,
                self.change_pitch,
                self.time_masking
            ]
            num_augs = np.random.randint(2, 4)
            selected_augs = random.sample(stress_augmentations, min(num_augs, len(stress_augmentations)))
        else:
            # For non-stress samples, use gentler augmentations
            normal_augmentations = [
                self.time_shift,
                self.change_speed,
                self.apply_filters
            ]
            num_augs = np.random.randint(1, 3)
            selected_augs = random.sample(normal_augmentations, min(num_augs, len(normal_augmentations)))
        
        # Apply selected augmentations
        augmented = np.ascontiguousarray(audio.copy())
        for aug_func in selected_augs:
            try:
                augmented = aug_func(augmented)
                if not augmented.flags.c_contiguous:
                    augmented = np.ascontiguousarray(augmented)
            except Exception as e:
                print(f"Stress-specific augmentation failed: {e}")
                augmented = np.ascontiguousarray(augmented)
        
        return augmented
    
    def get_available_augmentations(self) -> List[str]:
        """
        Returns a list of available augmentation methods.
        
        Returns:
            List of augmentation method names
        """
        return [
            'time_shift',
            'add_noise',
            'change_pitch',
            'change_speed',
            'apply_filters',
            'time_masking',
            'frequency_masking'
        ]
    
    def __repr__(self) -> str:
        """
        String representation of the AudioAugmenter.
        
        Returns:
            String representation
        """
        return f"AudioAugmenter(sample_rate={self.sample_rate})" 