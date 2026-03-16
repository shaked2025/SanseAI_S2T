"""
Lightweight Replay / Spoofing Detection

Detects common audio spoofing attacks in forensic interview settings:
1. Replay attacks (playing back a recording through a speaker)
2. Cut-and-splice manipulation (edited audio segments)

Methods:
- Spectral flatness: replayed audio has flatter spectrum due to loudspeaker frequency response
- Modulation spectrum: replayed audio shows different temporal modulation patterns
- Pop noise / transient detector: live speech has micro-transients absent in replayed audio
- Channel consistency: live audio has consistent channel characteristics across segments

This is NOT a full ASVspoof countermeasure (would require dedicated neural CM model).
It provides a lightweight first line of defense suitable for the interview room scenario.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq
import logging

log = logging.getLogger(__name__)


class ReplayDetector:
    """
    Lightweight replay detection using acoustic analysis.
    Returns a spoof probability score (0 = definitely live, 1 = definitely spoofed).
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.live_stats = None  # Learned during enrollment

    def analyze(self, audio_data):
        """
        Analyze audio for replay artifacts.

        Args:
            audio_data: numpy array (int16 or float32)

        Returns:
            dict with 'spoof_probability' (0-1), 'features', and 'verdict'
        """
        if audio_data.dtype == np.int16:
            audio = audio_data.astype(np.float64) / 32768.0
        else:
            audio = audio_data.astype(np.float64)

        if len(audio) < self.sample_rate * 0.5:
            return {'spoof_probability': 0.0, 'features': {}, 'verdict': 'too_short'}

        features = {}
        features['spectral_flatness'] = self._spectral_flatness(audio)
        features['modulation_index'] = self._modulation_spectrum_energy(audio)
        features['transient_density'] = self._transient_density(audio)
        features['channel_variance'] = self._channel_consistency(audio)

        # Score each feature against expected live ranges
        scores = []

        # Spectral flatness: live speech 0.01-0.08; replayed 0.10-0.30
        sf = features['spectral_flatness']
        if sf > 0.15:
            scores.append(min((sf - 0.08) / 0.22, 1.0))
        else:
            scores.append(0.0)

        # Modulation index: live speech has strong 2-8 Hz modulation
        mi = features['modulation_index']
        if mi < 0.3:
            scores.append(0.6)  # Low modulation = suspicious
        else:
            scores.append(0.0)

        # Transient density: live speech has micro-transients from plosives
        td = features['transient_density']
        if td < 0.005:
            scores.append(0.4)  # Very few transients = suspicious
        else:
            scores.append(0.0)

        # Channel variance: replayed audio has more consistent (flat) channel
        cv = features['channel_variance']
        if cv < 0.01:
            scores.append(0.3)
        else:
            scores.append(0.0)

        # Adaptive scoring: compare against enrolled live stats if available
        if self.live_stats is not None:
            deviation = self._deviation_from_live(features)
            scores.append(min(deviation / 3.0, 0.5))

        spoof_prob = float(np.clip(np.mean(scores), 0.0, 1.0))

        verdict = 'live'
        if spoof_prob > 0.6:
            verdict = 'likely_spoof'
        elif spoof_prob > 0.3:
            verdict = 'suspicious'

        return {
            'spoof_probability': spoof_prob,
            'features': features,
            'verdict': verdict
        }

    def learn_live_stats(self, audio_samples):
        """
        Learn acoustic statistics from known-live enrollment samples.
        Call this during enrollment so we can compare against live baseline.
        """
        all_features = []
        for audio in audio_samples:
            result = self.analyze(audio)
            if result['features']:
                all_features.append(result['features'])

        if not all_features:
            return

        self.live_stats = {}
        for key in all_features[0]:
            vals = [f[key] for f in all_features]
            self.live_stats[key] = {
                'mean': float(np.mean(vals)),
                'std': max(float(np.std(vals)), 1e-6)
            }
        log.info("Learned live audio stats from %d samples", len(all_features))

    def _deviation_from_live(self, features):
        """Z-score deviation from learned live statistics."""
        if self.live_stats is None:
            return 0.0
        deviations = []
        for key, val in features.items():
            if key in self.live_stats:
                z = abs(val - self.live_stats[key]['mean']) / self.live_stats[key]['std']
                deviations.append(z)
        return float(np.mean(deviations)) if deviations else 0.0

    def _spectral_flatness(self, audio):
        """
        Wiener entropy / spectral flatness.
        Geometric mean / arithmetic mean of power spectrum.
        Live speech is spectrally shaped; replayed audio is flatter.
        """
        fft = rfft(audio)
        power = np.abs(fft) ** 2 + 1e-10

        log_mean = np.mean(np.log(power))
        geometric_mean = np.exp(log_mean)
        arithmetic_mean = np.mean(power)

        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        return float(np.clip(flatness, 0, 1))

    def _modulation_spectrum_energy(self, audio):
        """
        Energy in 2-8 Hz modulation band (syllable rate).
        Live speech has strong modulation at ~4 Hz.
        Replayed audio through a speaker has distorted modulation spectrum.
        """
        frame_size = int(0.025 * self.sample_rate)
        hop_size = int(0.010 * self.sample_rate)

        energies = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energies.append(np.sqrt(np.mean(frame ** 2)))

        if len(energies) < 10:
            return 0.5

        envelope = np.array(energies)
        envelope = envelope - np.mean(envelope)

        mod_rate = 1.0 / (hop_size / self.sample_rate)
        fft_env = rfft(envelope)
        freqs = rfftfreq(len(envelope), 1.0 / mod_rate)

        mask_2_8 = (freqs >= 2) & (freqs <= 8)
        total_energy = np.sum(np.abs(fft_env) ** 2) + 1e-10
        band_energy = np.sum(np.abs(fft_env[mask_2_8]) ** 2)

        return float(band_energy / total_energy)

    def _transient_density(self, audio):
        """
        Density of micro-transients (plosives, clicks).
        Live speech has more acoustic transients than replayed audio.
        """
        envelope = np.abs(scipy_signal.hilbert(audio))
        diff = np.diff(envelope)
        threshold = np.std(diff) * 3.0
        transients = np.sum(np.abs(diff) > threshold)
        density = transients / len(audio)
        return float(density)

    def _channel_consistency(self, audio):
        """
        Variance of spectral tilt across segments.
        Live audio from fixed mic has consistent tilt; replayed varies more.
        """
        seg_len = int(0.5 * self.sample_rate)
        tilts = []

        for i in range(0, len(audio) - seg_len, seg_len):
            seg = audio[i:i + seg_len]
            fft = rfft(seg)
            mag = np.abs(fft) + 1e-10
            freqs = rfftfreq(seg_len, 1.0 / self.sample_rate)

            if len(freqs) > 1:
                log_freqs = np.log(freqs[1:] + 1)
                log_mag = np.log(mag[1:])
                coeffs = np.polyfit(log_freqs, log_mag, 1)
                tilts.append(coeffs[0])

        if len(tilts) < 2:
            return 0.05

        return float(np.std(tilts))
