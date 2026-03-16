"""
Test-Time Augmentation (TTA) for Speaker Enrollment

Augments each enrollment sample with realistic acoustic variations to produce
a more robust speaker centroid — without requiring additional recording time.

Each 5-second sample produces 4 embedding variants:
  1. Original (clean)
  2. Speed perturbation (+5% faster)
  3. Speed perturbation (-5% slower)
  4. Additive noise (SNR 20 dB)

With 4 recordings this yields 16 embeddings for centroid computation,
equivalent to 4x the enrollment data.
"""

import numpy as np
from scipy import signal
import logging

log = logging.getLogger(__name__)


def speed_perturb(audio, factor=1.05, sr=16000):
    """Speed perturbation via resampling. factor>1 = faster/higher pitch."""
    n_out = int(len(audio) / factor)
    return signal.resample(audio, n_out).astype(audio.dtype)


def add_noise(audio, snr_db=20.0):
    """Add white Gaussian noise at specified SNR."""
    audio_f = audio.astype(np.float64)
    rms_signal = np.sqrt(np.mean(audio_f ** 2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.randn(len(audio)) * rms_noise
    return (audio_f + noise).astype(audio.dtype)


def bandpass_filter(audio, low=100, high=7500, sr=16000, order=5):
    """Mild bandpass simulating telephone channel variation."""
    nyq = sr / 2
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
    return signal.filtfilt(b, a, audio.astype(np.float64)).astype(audio.dtype)


def augment_enrollment_sample(audio, sr=16000):
    """
    Produce augmented variants of a single enrollment audio sample.

    Returns:
        List of numpy arrays (each same dtype as input)
    """
    variants = [audio]  # original always included

    try:
        variants.append(speed_perturb(audio, factor=1.05, sr=sr))
    except Exception:
        log.debug("Speed perturb +5%% failed, skipping")

    try:
        variants.append(speed_perturb(audio, factor=0.95, sr=sr))
    except Exception:
        log.debug("Speed perturb -5%% failed, skipping")

    try:
        variants.append(add_noise(audio, snr_db=20.0))
    except Exception:
        log.debug("Noise augmentation failed, skipping")

    return variants


def compute_augmented_centroid(embedding_extractor, audio_samples, sr=16000, quality_weights=None):
    """
    Compute quality-weighted centroid using TTA on all enrollment samples.

    Args:
        embedding_extractor: Object with extract_embedding(audio, sr) method
        audio_samples: List of raw audio numpy arrays (one per enrollment recording)
        sr: Sample rate
        quality_weights: Optional per-sample quality weights (len = len(audio_samples))

    Returns:
        (centroid, all_embeddings, per_sample_std)
    """
    all_embeddings = []
    all_weights = []

    for idx, sample in enumerate(audio_samples):
        base_weight = quality_weights[idx] if quality_weights is not None else 1.0
        variants = augment_enrollment_sample(sample, sr=sr)

        for v_idx, variant in enumerate(variants):
            emb = embedding_extractor.extract_embedding(variant, sr)
            if not np.allclose(emb, 0):
                all_embeddings.append(emb)
                # Original sample gets full weight; augmentations get 70%
                w = base_weight if v_idx == 0 else base_weight * 0.7
                all_weights.append(w)

    if not all_embeddings:
        return None, [], 0.0

    embeddings_arr = np.array(all_embeddings)
    weights_arr = np.array(all_weights)
    weights_arr = weights_arr / (weights_arr.sum() + 1e-10)

    centroid = np.average(embeddings_arr, axis=0, weights=weights_arr)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

    std = np.std(embeddings_arr, axis=0).mean()

    log.info("TTA centroid: %d embeddings from %d samples (avg std: %.4f)",
             len(all_embeddings), len(audio_samples), std)

    return centroid, all_embeddings, float(std)
