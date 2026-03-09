"""
Audio feature extraction module for stress detection.
Extracts various audio features including RMS, zero crossing rate, pitch features,
MFCC coefficients, chroma, spectral contrast, and rhythm features.
"""

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
import warnings
import os
from typing import Tuple, List, Optional
from .config import CONFIG


class AudioFeatures:
    """
    Class for extracting comprehensive audio features from audio signals.
    
    Supports both basic and comprehensive feature sets for stress detection.
    Features include energy, spectral, pitch, and rhythm characteristics.
    """
    
    def __init__(self, 
                 sample_rate: int = None,
                 feature_set: str = None,
                 n_mfcc: int = None):
        """
        Initialize the AudioFeatures extractor.
        
        Args:
            sample_rate: Audio sampling rate
            feature_set: Type of features to extract ('basic' or 'comprehensive')
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.sample_rate = sample_rate or CONFIG.audio.sample_rate
        self.feature_set = feature_set or CONFIG.features.feature_set
        self.n_mfcc = n_mfcc or CONFIG.features.n_mfcc
        self.pitch_min = CONFIG.features.pitch_min
        self.pitch_max = CONFIG.features.pitch_max
        self.chroma_bins = CONFIG.features.chroma_bins
        
    def extract_basic_features(self, audio: np.ndarray) -> Tuple[dict, List[str]]:
        """
        Extracts basic audio features (RMS, ZCR).
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Tuple of (features_dict, feature_names_list)
        """
        features = {}
        feature_names = []
        
        # RMS (Root Mean Square) - Energy content
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        feature_names.extend(['rms_mean', 'rms_std'])
        
        # Zero Crossing Rate - Frequency content indicator
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        feature_names.extend(['zcr_mean', 'zcr_std'])
        
        return features, feature_names
    
    def extract_pitch_features(self, audio: np.ndarray) -> Tuple[dict, List[str]]:
        """
        Extracts pitch-related features using Praat.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Tuple of (features_dict, feature_names_list)
        """
        features = {}
        feature_names = ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max']
        
        try:
            # Ensure mono audio for pitch extraction
            if audio.ndim > 1:
                audio_mono = np.mean(audio, axis=1)
            else:
                audio_mono = audio.astype(np.float64)
            
            # Create Praat sound object
            sound = parselmouth.Sound(audio_mono, sampling_frequency=self.sample_rate)
            
            # Extract pitch using Praat
            pitch = call(sound, "To Pitch", 0.01, self.pitch_min, self.pitch_max)
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_min'] = np.min(pitch_values)
                features['pitch_max'] = np.max(pitch_values)
            else:
                # No voiced frames found
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_min'] = 0.0
                features['pitch_max'] = 0.0
                
        except Exception as e:
            warnings.warn(f"Pitch extraction failed: {e}")
            # Set default values on failure
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_min'] = 0.0
            features['pitch_max'] = 0.0
        
        return features, feature_names
    
    def extract_mfcc_features(self, audio: np.ndarray) -> Tuple[dict, List[str]]:
        """
        Extracts MFCC (Mel-frequency cepstral coefficients) features.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Tuple of (features_dict, feature_names_list)
        """
        features = {}
        feature_names = []
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Store MFCC features
        for i in range(self.n_mfcc):
            features[f'mfcc_{i+1}_mean'] = mfccs_mean[i]
            features[f'mfcc_{i+1}_std'] = mfccs_std[i]
            feature_names.extend([f'mfcc_{i+1}_mean', f'mfcc_{i+1}_std'])
        
        return features, feature_names
    
    def extract_chroma_features(self, audio: np.ndarray) -> Tuple[dict, List[str]]:
        """
        Extracts chroma features (pitch class representation).
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Tuple of (features_dict, feature_names_list)
        """
        features = {}
        feature_names = []
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Store chroma features
        for i in range(self.chroma_bins):
            features[f'chroma_{i+1}_mean'] = chroma_mean[i]
            features[f'chroma_{i+1}_std'] = chroma_std[i]
            feature_names.extend([f'chroma_{i+1}_mean', f'chroma_{i+1}_std'])
        
        return features, feature_names
    
    def extract_spectral_contrast_features(self, audio: np.ndarray) -> Tuple[dict, List[str]]:
        """
        Extracts spectral contrast features.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Tuple of (features_dict, feature_names_list)
        """
        features = {}
        feature_names = []
        
        # Extract spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        # Store contrast features
        for i in range(contrast.shape[0]):
            features[f'contrast_{i+1}_mean'] = contrast_mean[i]
            features[f'contrast_{i+1}_std'] = contrast_std[i]
            feature_names.extend([f'contrast_{i+1}_mean', f'contrast_{i+1}_std'])
        
        return features, feature_names
    
    def extract_rhythm_features(self, audio: np.ndarray) -> Tuple[dict, List[str]]:
        """
        Extracts rhythm-related features (tempo, onset strength).
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Tuple of (features_dict, feature_names_list)
        """
        features = {}
        feature_names = ['tempo', 'onset_strength_mean', 'onset_strength_std']
        
        # Calculate onset envelope (strength of onsets/beats in the signal)
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        
        # Estimate tempo from onset envelope
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
        
        # Store tempo, defaulting to 60.0 BPM if estimation fails
        # BPM = Beats Per Minute, 60 is standard for relaxing speaking rhythm
        features['tempo'] = tempo[0] if isinstance(tempo, np.ndarray) and len(tempo) > 0 else 60.0
        
        # Calculate and store onset envelope statistics
        features['onset_strength_mean'] = np.mean(onset_env)
        features['onset_strength_std'] = np.std(onset_env)
        
        return features, feature_names
    
    def extract_audio_features(self, audio: np.ndarray, 
                             sample_rate: int = None, 
                             feature_set: str = None) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        Extracts a comprehensive set of audio features from an audio signal.
        
        Args:
            audio: The audio time series as numpy array
            sample_rate: The sampling rate of the audio (optional, uses class default)
            feature_set: Can be 'basic' or 'comprehensive' (optional, uses class default)

        Returns:
            Tuple of (feature_vector as np.ndarray, feature_names as list)
            Returns (None, None) if extraction fails significantly
        """
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if feature_set is not None:
            self.feature_set = feature_set
            
        try:
            all_features = {}
            all_feature_names = []
            
            # --- Basic Features ---
            basic_features, basic_names = self.extract_basic_features(audio)
            all_features.update(basic_features)
            all_feature_names.extend(basic_names)
            
            # --- Pitch Features ---
            pitch_features, pitch_names = self.extract_pitch_features(audio)
            all_features.update(pitch_features)
            all_feature_names.extend(pitch_names)
            
            # --- MFCC Features ---
            mfcc_features, mfcc_names = self.extract_mfcc_features(audio)
            all_features.update(mfcc_features)
            all_feature_names.extend(mfcc_names)
            
            # --- Comprehensive Features (only if requested) ---
            if self.feature_set == 'comprehensive':
                # Chroma features
                chroma_features, chroma_names = self.extract_chroma_features(audio)
                all_features.update(chroma_features)
                all_feature_names.extend(chroma_names)
                
                # Spectral contrast features
                contrast_features, contrast_names = self.extract_spectral_contrast_features(audio)
                all_features.update(contrast_features)
                all_feature_names.extend(contrast_names)
            
            # --- Rhythm Features ---
            rhythm_features, rhythm_names = self.extract_rhythm_features(audio)
            all_features.update(rhythm_features)
            all_feature_names.extend(rhythm_names)
            
            # --- Build Final Feature Vector ---
            feature_vector = []
            for name in all_feature_names:
                value = all_features.get(name, 0.0)
                # Handle NaN values
                if np.isnan(value):
                    value = 0.0
                feature_vector.append(value)
            
            feature_vector = np.array(feature_vector, dtype=np.float32)
            
            # Clean up any remaining NaN/inf values
            np.nan_to_num(feature_vector, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Validation check
            if len(feature_vector) != len(all_feature_names):
                warnings.warn("Feature/name length mismatch")
                return None, None
            
            return feature_vector, all_feature_names
            
        except Exception as e:
            warnings.warn(f"Error during feature extraction: {e}. Returning None.", RuntimeWarning)
            return None, None
    
    def get_feature_dimension(self) -> int:
        """
        Returns the expected dimension of the feature vector for the current configuration.
        
        Returns:
            Expected number of features
        """
        # Calculate based on current configuration
        basic_features = 4  # RMS mean/std, ZCR mean/std
        pitch_features = 4  # pitch mean/std/min/max
        mfcc_features = self.n_mfcc * 2  # mean and std for each coefficient
        rhythm_features = 3  # tempo, onset strength mean/std
        
        if self.feature_set == 'comprehensive':
            chroma_features = self.chroma_bins * 2  # mean and std for each chroma bin
            contrast_features = 7 * 2  # typical spectral contrast bands * 2
            return basic_features + pitch_features + mfcc_features + chroma_features + contrast_features + rhythm_features
        else:
            return basic_features + pitch_features + mfcc_features + rhythm_features
    
    def save_features_names(self, feature_names: List[str], filename: str = None) -> bool:
        """
        Saves a list of feature names to a file.
        
        Args:
            feature_names: List of feature names to save
            filename: Output filename (optional, uses default path)
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not feature_names or len(feature_names) == 0:
            print("Warning: Feature names list provided to save_features_names was empty or None.")
            return False
            
        if filename is None:
            filename = CONFIG.predictor.feature_names_path
            
        try:
            # Ensure directory exists before writing
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                for name in feature_names:
                    f.write(f"{name}\n")
            print(f"Saved {len(feature_names)} feature names to {filename}")
            return True
            
        except Exception as e:
            print(f"Warning: Could not save feature names to file '{filename}': {e}")
            return False
    
    def load_feature_names(self, filename: str = None) -> Optional[List[str]]:
        """
        Loads feature names from a file.
        
        Args:
            filename: Input filename (optional, uses default path)
            
        Returns:
            List of feature names or None if loading fails
        """
        if filename is None:
            filename = CONFIG.predictor.feature_names_path
            
        if not os.path.exists(filename):
            print(f"ERROR: Feature names file not found: '{filename}'")
            return None
            
        try:
            with open(filename, 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]
            
            if len(feature_names) == 0:
                print(f"Warning: Feature names file '{filename}' seems empty.")
                return None
                
            print(f"Loaded {len(feature_names)} feature names from '{filename}'")
            return feature_names
            
        except Exception as e:
            print(f"Error reading feature names file '{filename}': {e}")
            return None
    
    def get_features_number(self, feature_filename: str = None) -> int:
        """
        Reads the feature names file and returns the number of features.

        Args:
            feature_filename: Path to the feature names text file

        Returns:
            The number of features found, or 0 if the file doesn't exist or is empty
        """
        feature_names = self.load_feature_names(feature_filename)
        return len(feature_names) if feature_names else 0
    
    def __repr__(self) -> str:
        """
        String representation of the AudioFeatures object.
        
        Returns:
            String representation
        """
        return (f"AudioFeatures(sample_rate={self.sample_rate}, "
                f"feature_set='{self.feature_set}', "
                f"n_mfcc={self.n_mfcc})")

