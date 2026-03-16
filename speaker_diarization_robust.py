"""
Production Speaker Diarization using SpeechBrain ECAPA-TDNN
192-dimensional embeddings with 0.69% EER on VoxCeleb1
Replaces Resemblyzer (estimated 5-10%+ EER)
"""

import numpy as np
from collections import deque, defaultdict
from datetime import datetime
import threading
import pickle
import os
import logging
import torch

log = logging.getLogger(__name__)

import torchaudio_compat  # noqa: F401 — patches torchaudio for SpeechBrain

EMBEDDING_DIM = 192


class EcapaTdnnEmbeddings:
    """Extract speaker embeddings using SpeechBrain ECAPA-TDNN (0.69% EER on VoxCeleb1)"""

    def __init__(self, device=None):
        self.encoder = None
        self.lock = threading.Lock()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_encoder()

    def _load_encoder(self):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            with self.lock:
                self.encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="models/spkrec-ecapa-voxceleb",
                    run_opts={"device": self.device},
                    local_strategy=LocalStrategy.COPY,
                )
            log.info("ECAPA-TDNN encoder loaded (%d-dim embeddings, device=%s)", EMBEDDING_DIM, self.device)
        except Exception as e:
            log.error("Error loading ECAPA-TDNN: %s", e)
            raise

    def extract_embedding(self, audio_data, sample_rate=16000):
        """
        Extract 192-D speaker embedding from audio.

        Args:
            audio_data: numpy array (int16 or float32)
            sample_rate: sample rate (will resample to 16 kHz if needed)

        Returns:
            L2-normalized 192-dimensional numpy embedding
        """
        try:
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)

            if sample_rate != 16000:
                from scipy import signal
                num_samples = int(len(audio_float) * 16000 / sample_rate)
                audio_float = signal.resample(audio_float, num_samples)

            waveform = torch.tensor(audio_float).unsqueeze(0)

            with self.lock:
                embedding = self.encoder.encode_batch(waveform)

            emb_np = embedding.squeeze().cpu().numpy()
            emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-10)
            return emb_np

        except Exception as e:
            log.error("Embedding extraction failed: %s", e)
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)


# Backwards-compatible alias
ResemblyzerEmbeddings = EcapaTdnnEmbeddings


class RobustSpeakerDatabase:
    """Manages speaker profiles with multi-utterance enrollment"""

    def __init__(self, max_speakers=10, enrollment_size=3):
        self.max_speakers = max_speakers
        self.enrollment_size = enrollment_size
        self.speakers = {}
        self.next_speaker_id = 0
        self.lock = threading.Lock()

    def add_speaker(self, embedding, speaker_name=None):
        with self.lock:
            speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            self.speakers[speaker_id] = {
                'embeddings': [embedding],
                'mean_embedding': embedding.copy(),
                'std': 0.0,
                'count': 1,
                'name': speaker_name or f"Speaker {speaker_id + 1}",
                'enrolled': False,
                'confidence_history': [],
                'threshold': 0.75,
                'first_seen': datetime.now(),
                'last_seen': datetime.now()
            }
            log.info("New speaker created: %s (enrolling...)", self.speakers[speaker_id]['name'])
            return speaker_id

    def update_speaker(self, speaker_id, embedding, confidence=1.0):
        with self.lock:
            if speaker_id not in self.speakers:
                return
            speaker = self.speakers[speaker_id]

            speaker['embeddings'].append(embedding)
            if len(speaker['embeddings']) > 20:
                speaker['embeddings'].pop(0)

            alpha = 0.10
            speaker['mean_embedding'] = (
                alpha * embedding + (1 - alpha) * speaker['mean_embedding']
            )
            speaker['mean_embedding'] = speaker['mean_embedding'] / (
                np.linalg.norm(speaker['mean_embedding']) + 1e-10
            )

            if len(speaker['embeddings']) >= self.enrollment_size:
                embeddings_array = np.array(speaker['embeddings'])
                speaker['std'] = np.std(embeddings_array, axis=0).mean()
                if not speaker['enrolled']:
                    speaker['enrolled'] = True
                    speaker['threshold'] = max(0.70, 0.85 - speaker['std'] * 10)
                    log.info("%s enrolled (threshold: %.2f)", speaker['name'], speaker['threshold'])

            speaker['count'] += 1
            speaker['last_seen'] = datetime.now()
            speaker['confidence_history'].append(confidence)
            if len(speaker['confidence_history']) > 50:
                speaker['confidence_history'].pop(0)

    def find_speaker(self, embedding, require_enrolled=True):
        with self.lock:
            if not self.speakers:
                return None, 0.0

            best_speaker = None
            best_similarity = -1.0

            for speaker_id, speaker in self.speakers.items():
                if require_enrolled and not speaker['enrolled']:
                    continue
                similarity = float(np.dot(embedding, speaker['mean_embedding']))

                if speaker['enrolled']:
                    variance_boost = 1.0 / (1.0 + speaker['std'] * 5)
                    adjusted_similarity = similarity * variance_boost
                else:
                    adjusted_similarity = similarity

                if adjusted_similarity > best_similarity:
                    best_similarity = adjusted_similarity
                    best_speaker = speaker_id

            if best_speaker is None:
                return None, 0.0

            threshold = self.speakers[best_speaker]['threshold']
            if best_similarity >= threshold:
                return best_speaker, float(best_similarity)
            else:
                return None, float(best_similarity)

    def get_speaker_info(self, speaker_id):
        with self.lock:
            return self.speakers.get(speaker_id, None)

    def get_all_speakers(self):
        with self.lock:
            return list(self.speakers.keys())

    def save(self, filepath):
        with self.lock:
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(self.speakers, f)
                log.info("Speaker database saved (%d speakers)", len(self.speakers))
            except Exception as e:
                log.error("Error saving database: %s", e)

    def load(self, filepath):
        with self.lock:
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        self.speakers = pickle.load(f)
                    self.next_speaker_id = max(self.speakers.keys()) + 1 if self.speakers else 0
                    enrolled = sum(1 for s in self.speakers.values() if s.get('enrolled', False))
                    log.info("Loaded %d speakers (%d enrolled)", len(self.speakers), enrolled)
                    return True
            except Exception as e:
                log.error("Error loading database: %s", e)
        return False


class AdvancedTemporalSmoother:
    """Temporal smoothing with confidence weighting and recency bias"""

    def __init__(self, window_seconds=10, sample_interval=0.5):
        self.window_seconds = window_seconds
        self.sample_interval = sample_interval
        self.max_entries = int(window_seconds / sample_interval)
        self.history = deque(maxlen=self.max_entries)

    def smooth(self, speaker_id, confidence):
        now = datetime.now()
        self.history.append((speaker_id, confidence, now))

        if len(self.history) < 3:
            return speaker_id

        speaker_votes = defaultdict(float)
        for sid, conf, ts in self.history:
            age_seconds = (now - ts).total_seconds()
            recency_weight = 1.0 / (1.0 + age_seconds / 2.0)
            weight = conf * recency_weight
            speaker_votes[sid] += weight

        if speaker_votes:
            return max(speaker_votes.items(), key=lambda x: x[1])[0]
        return speaker_id

    def reset(self):
        self.history.clear()


class RobustSpeakerDiarization:
    """
    Production speaker diarization using ECAPA-TDNN (192-dim embeddings).
    0.69% EER on VoxCeleb1 vs ~5-10% with Resemblyzer.
    """

    def __init__(self, max_speakers=10, similarity_threshold=0.75):
        self.max_speakers = max_speakers
        self.base_threshold = similarity_threshold

        log.info("Initializing ECAPA-TDNN Speaker Diarization (192-dim, 0.69%% EER)")
        self.embedding_extractor = EcapaTdnnEmbeddings()
        self.speaker_db = RobustSpeakerDatabase(
            max_speakers=max_speakers, enrollment_size=3
        )
        self.temporal_smoother = AdvancedTemporalSmoother(window_seconds=10, sample_interval=0.5)

        self.total_identifications = 0
        self.successful_matches = 0
        self.new_speaker_count = 0

        self.db_path = "speaker_database_robust.pkl"
        self.speaker_db.load(self.db_path)

    def identify_speaker(self, audio_data, sample_rate=16000):
        try:
            embedding = self.embedding_extractor.extract_embedding(audio_data, sample_rate)
            if np.allclose(embedding, 0):
                return 0

            self.total_identifications += 1

            speaker_id, confidence = self.speaker_db.find_speaker(
                embedding, require_enrolled=False
            )

            if speaker_id is not None:
                self.speaker_db.update_speaker(speaker_id, embedding, confidence)
                self.successful_matches += 1
            else:
                if len(self.speaker_db.get_all_speakers()) >= self.max_speakers:
                    # Safety: reject unknown when at capacity (no force-assignment)
                    log.warning("Max speakers reached, rejecting unknown voice")
                    return -1
                else:
                    speaker_id = self.speaker_db.add_speaker(embedding)
                    confidence = 1.0
                    self.new_speaker_count += 1

            smoothed_id = self.temporal_smoother.smooth(speaker_id, confidence)
            return smoothed_id

        except Exception as e:
            log.error("Speaker identification error: %s", e, exc_info=True)
            return 0

    def get_speaker_count(self):
        return len(self.speaker_db.get_all_speakers())

    def get_active_speakers(self):
        return self.speaker_db.get_all_speakers()

    def reset(self):
        self.speaker_db = RobustSpeakerDatabase(max_speakers=self.max_speakers)
        self.temporal_smoother.reset()
        self.total_identifications = 0
        self.successful_matches = 0
        self.new_speaker_count = 0

    def save_database(self):
        self.speaker_db.save(self.db_path)

    def get_statistics(self):
        accuracy = (self.successful_matches / self.total_identifications * 100) if self.total_identifications > 0 else 0
        enrolled_count = sum(
            1 for sid in self.speaker_db.get_all_speakers()
            if self.speaker_db.get_speaker_info(sid).get('enrolled', False)
        )
        return {
            'total_identifications': self.total_identifications,
            'successful_matches': self.successful_matches,
            'accuracy': accuracy,
            'num_speakers': self.get_speaker_count(),
            'enrolled_speakers': enrolled_count,
            'new_speakers_created': self.new_speaker_count
        }
