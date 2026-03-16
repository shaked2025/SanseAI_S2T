"""
Audio Capture Module with Silero-VAD
Real-time audio recording with neural voice activity detection
"""

import pyaudio
import numpy as np
import threading
import queue
import wave
from collections import deque
import time
import logging
import torch

log = logging.getLogger(__name__)


class AudioCapture:
    """Real-time audio capture with buffering"""

    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024, device_index=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.device_index = device_index

        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False

        self.audio_queue = queue.Queue()
        self.buffer = deque(maxlen=int(sample_rate * 10))  # 10 seconds max
        self.record_thread = None

    def start(self):
        if self.is_recording:
            return
        try:
            if self.device_index is not None:
                device_info = self.audio.get_device_info_by_index(self.device_index)
                log.info("Using microphone: %s (index %d)", device_info['name'], self.device_index)
            else:
                default_device = self.audio.get_default_input_device_info()
                log.info("Using default microphone: %s", default_device['name'])
                self.device_index = default_device['index']

            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            self.stream.start_stream()
            log.info("Audio capture started: %dHz, %d channel(s)", self.sample_rate, self.channels)
        except Exception as e:
            log.error("Error starting audio capture: %s", e)
            self.is_recording = False

    def stop(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.buffer.extend(audio_data)
        self.audio_queue.put(audio_data.copy())
        return (in_data, pyaudio.paContinue)

    def get_audio_chunk(self, timeout=0.1):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_buffer(self, duration=3.0):
        num_samples = int(self.sample_rate * duration)
        buf_len = len(self.buffer)
        if buf_len == 0:
            return np.array([], dtype=np.int16)
        num_samples = min(num_samples, buf_len)
        return np.array(list(self.buffer)[-num_samples:], dtype=np.int16)

    def clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def get_audio_level(self):
        if len(self.buffer) == 0:
            return 0
        recent = list(self.buffer)[-int(self.sample_rate * 0.1):]
        if not recent:
            return 0
        return float(np.sqrt(np.mean(np.square(np.array(recent, dtype=np.float64)))))

    def cleanup(self):
        self.stop()
        self.audio.terminate()


class SileroVAD:
    """
    Neural voice activity detection using Silero-VAD.
    Outperforms WebRTC-VAD by 30-50% across standard benchmarks
    (0.99 vs 0.81 AUC on LibriParty, 0.90 vs 0.66 on AVA).
    """

    def __init__(self, sample_rate=16000, threshold=0.45, min_speech_ms=250):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self._model = None
        self._utils = None
        self._load_model()

    def _load_model(self):
        try:
            self._model, self._utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True,
            )
            log.info("Silero-VAD loaded (threshold=%.2f)", self.threshold)
        except Exception as e:
            log.error("Silero-VAD load failed: %s — falling back to RMS-based VAD", e)
            self._model = None

    def is_speech(self, audio_data):
        """
        Returns True if speech is detected in the audio chunk.
        Falls back to RMS energy detection if Silero fails.
        """
        if self._model is None:
            return self._rms_fallback(audio_data)

        try:
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)

            wav = torch.tensor(audio_float)
            get_speech_timestamps = self._utils[0]
            speech_timestamps = get_speech_timestamps(
                wav, self._model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_ms,
            )
            return len(speech_timestamps) > 0

        except Exception as e:
            log.warning("Silero-VAD inference failed: %s", e)
            return self._rms_fallback(audio_data)

    def speech_probability(self, audio_data):
        """Return speech probability (0-1) for a ~250ms chunk."""
        if self._model is None:
            return 0.0

        try:
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)

            wav = torch.tensor(audio_float)
            prob = self._model(wav, self.sample_rate).item()
            return prob
        except Exception:
            return 0.0

    def get_speech_segments(self, audio_data):
        """Return list of (start_sample, end_sample) speech segments."""
        if self._model is None:
            return [(0, len(audio_data))] if self._rms_fallback(audio_data) else []

        try:
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)

            wav = torch.tensor(audio_float)
            get_speech_timestamps = self._utils[0]
            stamps = get_speech_timestamps(
                wav, self._model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_ms,
            )
            return [(s['start'], s['end']) for s in stamps]
        except Exception:
            return []

    @staticmethod
    def _rms_fallback(audio_data, threshold=300):
        rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        return rms > threshold


# Backwards-compatible alias
class VoiceActivityDetector:
    """Wrapper that delegates to SileroVAD for backward compat."""

    def __init__(self, sample_rate=16000, threshold=419, min_duration=0.3):
        self._silero = SileroVAD(
            sample_rate=sample_rate,
            threshold=0.45,
            min_speech_ms=int(min_duration * 1000),
        )

    def is_speech(self, audio_data):
        return self._silero.is_speech(audio_data)

    def get_speech_segments(self, audio_data, window_size=0.03):
        segs = self._silero.get_speech_segments(audio_data)
        if not segs:
            return [False] * (len(audio_data) // int(self._silero.sample_rate * window_size))
        mask = np.zeros(len(audio_data), dtype=bool)
        for start, end in segs:
            mask[start:end] = True
        frame_size = int(self._silero.sample_rate * window_size)
        return [bool(mask[i:i + frame_size].any())
                for i in range(0, len(audio_data), frame_size)]
