"""
Speaker Enrollment System for Interview Transcription
Quality-weighted centroid enrollment with test-time augmentation
"""

import numpy as np
import pickle
import os
from datetime import datetime
from collections import deque
import threading
import logging

from enrollment_augmentation import augment_enrollment_sample, compute_augmented_centroid

log = logging.getLogger(__name__)

EMBEDDING_DIM = 192


class SpeakerEnrollment:
    """Manages speaker enrollment with quality-weighted voiceprints"""

    def __init__(self, embedding_extractor, min_samples=4):
        """
        Args:
            embedding_extractor: Object with extract_embedding(audio, sr) method
            min_samples: Minimum enrollment samples (4 achieves reliable centroid
                        with ECAPA-TDNN; previously 5-6 with weaker Resemblyzer)
        """
        self.embedding_extractor = embedding_extractor
        self.enrolled_speakers = {}
        self.enrollment_samples_required = min_samples
        self.lock = threading.Lock()

    def start_enrollment(self, speaker_key, name, role):
        with self.lock:
            self.enrolled_speakers[speaker_key] = {
                'key': speaker_key,
                'name': name,
                'role': role,
                'embeddings': [],
                'raw_audio_samples': [],
                'qualities': [],
                'mean_embedding': None,
                'std': None,
                'threshold': 0.40,
                'quality': 0.0,
                'enrolled': False,
                'total_utterances': 0,
                'correct_identifications': 0,
                'enrollment_start': datetime.now()
            }

    def add_enrollment_sample(self, speaker_key, audio_data, sample_rate=16000):
        """Add a voice sample. Returns (success, quality_score, message)."""
        try:
            if speaker_key not in self.enrolled_speakers:
                return False, 0.0, "Speaker not initialized"

            embedding = self.embedding_extractor.extract_embedding(audio_data, sample_rate)
            if np.allclose(embedding, 0):
                return False, 0.0, "Failed to extract voice features"

            # Estimate sample quality from embedding norm pre-normalization and SNR
            sample_quality = self._estimate_sample_quality(audio_data)

            with self.lock:
                speaker = self.enrolled_speakers[speaker_key]
                speaker['embeddings'].append(embedding)
                speaker['raw_audio_samples'].append(audio_data.copy())
                speaker['qualities'].append(sample_quality)

                n = len(speaker['embeddings'])
                if n >= 2:
                    arr = np.array(speaker['embeddings'])
                    std = np.std(arr, axis=0).mean()
                    quality = 1.0 / (1.0 + std * 20)
                    speaker['quality'] = quality
                else:
                    quality = 0.9

                msg = f"Sample {n}/{self.enrollment_samples_required} (quality: {quality:.1%})"
                return True, quality, msg

        except Exception as e:
            return False, 0.0, f"Error: {e}"

    def complete_enrollment(self, speaker_key):
        """
        Finalize enrollment using quality-weighted centroid.
        Returns (success, quality, message).
        """
        try:
            with self.lock:
                if speaker_key not in self.enrolled_speakers:
                    return False, 0.0, "Speaker not found"

                speaker = self.enrolled_speakers[speaker_key]
                n = len(speaker['embeddings'])
                if n < self.enrollment_samples_required:
                    return False, 0.0, f"Need {self.enrollment_samples_required} samples, have {n}"

                raw_audio = speaker.get('raw_audio_samples', [])
                qualities = np.array(speaker['qualities']) if speaker['qualities'] else np.ones(n)

                # TTA: augment each sample -> 4x embeddings -> quality-weighted centroid
                if raw_audio:
                    centroid, all_embs, std = compute_augmented_centroid(
                        self.embedding_extractor, raw_audio,
                        sr=16000, quality_weights=qualities
                    )
                    if centroid is None:
                        return False, 0.0, "TTA centroid computation failed"
                else:
                    embeddings = np.array(speaker['embeddings'])
                    weights = qualities / (qualities.sum() + 1e-10)
                    centroid = np.average(embeddings, axis=0, weights=weights)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
                    std = np.std(embeddings, axis=0).mean()

                quality = 1.0 / (1.0 + std * 15)

                # ECAPA-TDNN operates with lower raw cosine scores than Resemblyzer;
                # genuine pairs typically > 0.40, impostors < 0.25
                threshold = 0.45 - (std * 5)
                threshold = float(np.clip(threshold, 0.30, 0.55))

                speaker['mean_embedding'] = centroid
                speaker['std'] = std
                speaker['threshold'] = threshold
                speaker['quality'] = quality
                speaker['enrolled'] = True
                speaker['total_utterances'] = n
                speaker['enrollment_end'] = datetime.now()

                status = "excellent" if quality >= 0.80 else "good" if quality >= 0.70 else "acceptable"
                msg = f"Enrollment complete! Quality: {quality:.1%} ({status}), Threshold: {threshold:.2f}"

                # Free raw audio to reclaim memory (embeddings are kept)
                speaker['raw_audio_samples'] = []

                log.info("%s enrolled — quality: %.1f%%, threshold: %.2f, samples: %d",
                         speaker['name'], quality * 100, threshold, n)
                return True, quality, msg

        except Exception as e:
            return False, 0.0, f"Error completing enrollment: {e}"

    def test_speaker_separation(self):
        with self.lock:
            enrolled = {k: v for k, v in self.enrolled_speakers.items() if v.get('enrolled', False)}
            if len(enrolled) < 2:
                return {"error": "Need at least 2 enrolled speakers"}

            separations = {}
            for key1, s1 in enrolled.items():
                for key2, s2 in enrolled.items():
                    if key1 >= key2:
                        continue
                    sim = float(np.dot(s1['mean_embedding'], s2['mean_embedding']))
                    sep = 1.0 - sim
                    pair = f"{s1['name']} vs {s2['name']}"
                    separations[pair] = {
                        'separation': sep, 'similarity': sim,
                        'distinguishable': sep > 0.10
                    }
                    log.info("  %s %s: %.1f%% separation (similarity: %.3f)",
                             "OK" if sep > 0.10 else "WARN", pair, sep * 100, sim)
            return separations

    def get_enrolled_speakers(self):
        with self.lock:
            return {k: v for k, v in self.enrolled_speakers.items() if v.get('enrolled', False)}

    def save_enrollment(self, filepath):
        with self.lock:
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(self.enrolled_speakers, f)
                return True
            except Exception as e:
                log.error("Save enrollment failed: %s", e)
                return False

    def load_enrollment(self, filepath):
        with self.lock:
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        self.enrolled_speakers = pickle.load(f)
                    enrolled = sum(1 for s in self.enrolled_speakers.values() if s.get('enrolled', False))
                    log.info("Loaded %d enrolled speakers from %s", enrolled, filepath)
                    return True
                return False
            except Exception as e:
                log.error("Load enrollment failed: %s", e)
                return False

    @staticmethod
    def _estimate_sample_quality(audio_data):
        """Lightweight quality estimate from raw audio (0-1)."""
        if audio_data.dtype == np.int16:
            audio = audio_data.astype(np.float64)
        else:
            audio = audio_data.astype(np.float64) * 32768.0

        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 100:
            return 0.2

        # SNR proxy: top 20% energy / bottom 20% energy
        frame_size = 480  # 30ms at 16kHz
        n_frames = max(1, len(audio) // frame_size)
        energies = []
        for i in range(n_frames):
            chunk = audio[i * frame_size:(i + 1) * frame_size]
            if len(chunk) == frame_size:
                energies.append(np.sqrt(np.mean(chunk ** 2)))

        if len(energies) < 4:
            return 0.5

        energies.sort()
        noise_floor = np.mean(energies[:max(1, len(energies) // 5)])
        signal_level = np.mean(energies[-max(1, len(energies) // 5):])
        snr = signal_level / (noise_floor + 1e-10)
        snr_score = float(np.clip((snr - 2) / 10.0, 0.0, 1.0))

        # Clipping penalty
        clip_ratio = np.mean(np.abs(audio) > 31000)
        clip_score = 1.0 - min(clip_ratio * 10, 1.0)

        return 0.6 * snr_score + 0.4 * clip_score


class InterviewContextTracker:
    """Tracks interview Q&A pattern for context-aware boosting"""

    def __init__(self):
        self.turn_history = deque(maxlen=30)
        self.interviewer_key = None
        self.interviewee_keys = []

    def set_roles(self, interviewer_key, interviewee_keys):
        self.interviewer_key = interviewer_key
        self.interviewee_keys = interviewee_keys if isinstance(interviewee_keys, list) else [interviewee_keys]

    def add_turn(self, speaker_key, confidence):
        self.turn_history.append({
            'speaker': speaker_key, 'confidence': confidence, 'timestamp': datetime.now()
        })

    def predict_next_speaker(self):
        if not self.turn_history or not self.interviewer_key:
            return None, 0.0
        last = self.turn_history[-1]['speaker']
        if last == self.interviewer_key and self.interviewee_keys:
            return self.interviewee_keys[0], 0.75
        elif last != self.interviewer_key:
            return self.interviewer_key, 0.80
        return None, 0.0

    def get_speaker_stats(self):
        if not self.turn_history:
            return {}
        stats = {}
        for turn in self.turn_history:
            sp = turn['speaker']
            if sp not in stats:
                stats[sp] = {'count': 0, 'avg_confidence': []}
            stats[sp]['count'] += 1
            stats[sp]['avg_confidence'].append(turn['confidence'])
        for data in stats.values():
            data['avg_confidence'] = float(np.mean(data['avg_confidence']))
        return stats
