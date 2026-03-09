"""
Utility functions for audio processing and filtering.
Contains Voice Activity Detection (VAD) functions and other audio utilities.
"""

import numpy as np
import warnings
import librosa
from typing import Optional
from .config import CONFIG

try:
    import webrtcvad
except ImportError:
    warnings.warn("webrtcvad library not found. VAD filtering will be disabled. "
                  "Install with 'pip install webrtcvad-wheels' or 'pip install webrtcvad'.")
    webrtcvad = None


def is_chunk_speech(audio_chunk: np.ndarray, 
                   sample_rate: int, 
                   vad_aggressiveness: int = None, 
                   frame_duration_ms: int = None, 
                   speech_threshold: float = None,
                   check_electronic_noise: bool = True) -> bool:
    """
    Determines if a given audio chunk likely contains speech using WebRTC VAD.

    Args:
        audio_chunk: Mono audio chunk as float numpy array (-1.0 to 1.0)
        sample_rate: Sample rate (8000, 16000, 32000, 48000 Hz)
        vad_aggressiveness: VAD aggressiveness mode (0-3). 3 is most aggressive
        frame_duration_ms: Frame duration for VAD (10, 20, or 30 ms)
        speech_threshold: Proportion (0.0-1.0) of frames needed to be speech
                         for the chunk to be considered speech
        check_electronic_noise: Whether to check for electronic noise

    Returns:
        True if the chunk is considered speech, False otherwise.
        Returns True if webrtcvad is not installed.
    """
    # Use config defaults if parameters not provided
    if vad_aggressiveness is None:
        vad_aggressiveness = CONFIG.vad.aggressiveness
    if frame_duration_ms is None:
        frame_duration_ms = CONFIG.vad.frame_duration_ms
    if speech_threshold is None:
        speech_threshold = CONFIG.vad.speech_threshold
    
    if webrtcvad is None:
        # If library isn't installed, assume it's speech to avoid breaking pipeline
        return True

    # Validate VAD parameters
    if sample_rate not in [8000, 16000, 32000, 48000]:
        warnings.warn(f"VAD: Invalid sample rate {sample_rate}. Assuming speech.", RuntimeWarning)
        return True
    if frame_duration_ms not in [10, 20, 30]:
        warnings.warn(f"VAD: Invalid frame duration {frame_duration_ms}. Assuming speech.", RuntimeWarning)
        return True
    if not 0 <= vad_aggressiveness <= 3:
         warnings.warn(f"VAD: Invalid aggressiveness {vad_aggressiveness}. Using default 1.", RuntimeWarning)
         vad_aggressiveness = 1

    try:
        vad = webrtcvad.Vad(vad_aggressiveness)
    except Exception as e:
        warnings.warn(f"VAD: Failed to initialize VAD: {e}. Assuming speech.", RuntimeWarning)
        return True

    # Convert float audio to 16-bit PCM bytes
    try:
        # Ensure mono (though input should ideally be)
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)

        # Check for near silence, VAD might behave unpredictably
        if np.max(np.abs(audio_chunk)) < 1e-5:
            return False # Treat silence as non-speech

        # Scale and convert
        int16_audio = (audio_chunk * 32767).astype(np.int16)
        audio_bytes = int16_audio.tobytes()
    except Exception as e:
         warnings.warn(f"VAD: Failed converting audio chunk to bytes: {e}. Assuming speech.", RuntimeWarning)
         return True

    frame_size_samples = int(sample_rate * frame_duration_ms / 1000)
    frame_size_bytes = frame_size_samples * 2 # 16-bit = 2 bytes/sample

    num_frames_total = len(audio_bytes) // frame_size_bytes
    num_speech_frames = 0

    if num_frames_total == 0:
        return False # Chunk is too short for even one frame

    # Process frames
    for i in range(num_frames_total):
        start_byte = i * frame_size_bytes
        frame = audio_bytes[start_byte : start_byte + frame_size_bytes]
        try:
            if vad.is_speech(frame, sample_rate):
                num_speech_frames += 1
        except Exception as e:
            # Handle potential VAD errors on specific frames
            warnings.warn(f"VAD: Error processing frame {i}: {e}", RuntimeWarning)
            # Decide how to treat error - skip frame? Assume non-speech?
            # Let's be conservative and not count it as speech
            pass

    speech_proportion = num_speech_frames / num_frames_total
    return speech_proportion >= speech_threshold


def calculate_rms_features(audio_chunk: np.ndarray, 
                          frame_length: int = None, 
                          hop_length: int = None) -> float:
    """
    Calculates RMS (Root Mean Square) energy of an audio chunk.
    
    Args:
        audio_chunk: Audio signal as numpy array
        frame_length: Frame length for RMS calculation
        hop_length: Hop length for RMS calculation
        
    Returns:
        Mean RMS value of the audio chunk
    """
    if frame_length is None:
        frame_length = CONFIG.audio.rms_frame_length
    if hop_length is None:
        hop_length = CONFIG.audio.rms_hop_length
        
    try:
        # Ensure chunk is long enough for at least one RMS frame
        if len(audio_chunk) < frame_length:
            return 0.0 # Treat very short chunks as having low energy
        
        rms_frames = librosa.feature.rms(y=audio_chunk, frame_length=frame_length, hop_length=hop_length)[0]
        if rms_frames.size > 0:
            return np.mean(rms_frames)
        else:
            return 0.0 # Should not happen if len > frame_length, but safe default
    except Exception as e:
        warnings.warn(f"RMS calculation failed: {e}", RuntimeWarning)
        return 0.0


def passes_filter(audio_chunk: np.ndarray, 
                 sample_rate: int,
                 vad_aggressiveness: int = None, 
                 frame_duration_ms: int = None, 
                 speech_threshold: float = None,
                 max_rms_threshold: float = None, 
                 rms_frame_length: int = None, 
                 rms_hop_length: int = None) -> bool:
    """
    Checks if an audio chunk passes VAD and optionally a Max RMS filter.

    Args:
        audio_chunk: Mono float audio chunk
        sample_rate: Sample rate
        vad_aggressiveness: VAD aggressiveness (0-3)
        frame_duration_ms: VAD frame duration (10, 20, 30 ms)
        speech_threshold: VAD speech proportion threshold (0.0-1.0)
        max_rms_threshold: If not None, chunk must have mean RMS <= this value to pass
        rms_frame_length: Frame length for RMS calculation
        rms_hop_length: Hop length for RMS calculation

    Returns:
        True if the chunk passes all active filters, False otherwise
    """
    # Use config defaults if parameters not provided
    if vad_aggressiveness is None:
        vad_aggressiveness = CONFIG.vad.aggressiveness
    if frame_duration_ms is None:
        frame_duration_ms = CONFIG.vad.frame_duration_ms
    if speech_threshold is None:
        speech_threshold = CONFIG.vad.speech_threshold
    if max_rms_threshold is None:
        max_rms_threshold = CONFIG.vad.max_rms_threshold
    if rms_frame_length is None:
        rms_frame_length = CONFIG.audio.rms_frame_length
    if rms_hop_length is None:
        rms_hop_length = CONFIG.audio.rms_hop_length

    # --- 1. VAD Check ---
    is_speech = is_chunk_speech(
        audio_chunk, 
        sample_rate, 
        vad_aggressiveness, 
        frame_duration_ms, 
        speech_threshold
    )

    if not is_speech:
        return False # Failed VAD, no need to check RMS

    # --- 2. Max RMS Check (only if threshold is provided) ---
    if max_rms_threshold is not None:
        try:
            chunk_rms = calculate_rms_features(audio_chunk, rms_frame_length, rms_hop_length)
            passes_rms_filter = chunk_rms <= max_rms_threshold
            
            if not passes_rms_filter:
                print(f"Failed RMS filter: {chunk_rms}")
                return False # Failed RMS check
        except Exception as e:
             warnings.warn(f"Filter: Error during RMS check: {e}. Skipping RMS filter for this chunk.", RuntimeWarning)
             # Decide behaviour: pass chunk or fail chunk if RMS check errors? Let's pass it.
             pass

    # --- If we reach here, all active checks passed ---
    return True


def ensure_audio_contiguous(audio: np.ndarray) -> np.ndarray:
    """
    Ensures that audio array is C-contiguous for efficient processing.
    
    Args:
        audio: Input audio array
        
    Returns:
        C-contiguous audio array
    """
    if not audio.flags.c_contiguous:
        return np.ascontiguousarray(audio)
    return audio


def validate_audio_chunk(audio_chunk: np.ndarray, 
                        min_length: int = 100) -> bool:
    """
    Validates if an audio chunk is suitable for processing.
    
    Args:
        audio_chunk: Audio chunk to validate
        min_length: Minimum required length
        
    Returns:
        True if chunk is valid, False otherwise
    """
    if audio_chunk is None:
        return False
    if len(audio_chunk) < min_length:
        return False
    if np.max(np.abs(audio_chunk)) < 1e-5:
        return False  # Too quiet
    return True


def normalize_audio(audio: np.ndarray, target_max: float = 1.0) -> np.ndarray:
    """
    Normalizes audio to a target maximum amplitude.
    
    Args:
        audio: Input audio array
        target_max: Target maximum amplitude
        
    Returns:
        Normalized audio array
    """
    max_val = np.max(np.abs(audio))
    if max_val > 1e-8:  # Avoid division by zero
        return audio * (target_max / max_val)
    return audio 