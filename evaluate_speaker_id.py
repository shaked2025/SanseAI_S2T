"""
Speaker Verification Evaluation Framework

Computes EER, minDCF, and DET curves from real audio files.
Designed for proper train/test separation: enrollment data is NEVER used for testing.

Usage:
    python evaluate_speaker_id.py

Expects WAV files in the project directory (uses any .wav files present).
"""

import numpy as np
import os
import sys
import logging
import time
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

import torchaudio_compat  # noqa: F401

from speaker_diarization_robust import EcapaTdnnEmbeddings, EMBEDDING_DIM


def load_wav(filepath, target_sr=16000):
    """Load a WAV file and return int16 numpy array."""
    try:
        import soundfile as sf
        audio, sr = sf.read(filepath, dtype='int16')
        if sr != target_sr:
            from scipy import signal
            n_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, n_samples).astype(np.int16)
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio
    except Exception as e:
        log.error("Failed to load %s: %s", filepath, e)
        return None


def chunk_audio(audio, chunk_sec=5.0, hop_sec=2.5, sr=16000):
    """Split audio into overlapping chunks."""
    chunk_samples = int(chunk_sec * sr)
    hop_samples = int(hop_sec * sr)
    chunks = []
    for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
        chunks.append(audio[start:start + chunk_samples])
    return chunks


def compute_eer(genuine_scores, impostor_scores):
    """
    Compute Equal Error Rate from genuine and impostor score lists.
    Returns (eer, threshold_at_eer).
    """
    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)

    all_scores = np.concatenate([genuine, impostor])
    thresholds = np.sort(np.unique(all_scores))

    far_list = []
    frr_list = []

    for t in thresholds:
        far = np.mean(impostor >= t)   # False Accept Rate
        frr = np.mean(genuine < t)     # False Reject Rate
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    # Find crossing point
    diffs = far_arr - frr_arr
    idx = np.argmin(np.abs(diffs))

    eer = 0.5 * (far_arr[idx] + frr_arr[idx])
    threshold = thresholds[idx]

    return float(eer), float(threshold)


def compute_min_dcf(genuine_scores, impostor_scores, p_target=0.01, c_miss=1.0, c_fa=1.0):
    """Compute minimum Detection Cost Function (NIST SRE style)."""
    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)

    all_scores = np.concatenate([genuine, impostor])
    thresholds = np.sort(np.unique(all_scores))

    min_dcf = float('inf')

    for t in thresholds:
        fnr = np.mean(genuine < t)
        fpr = np.mean(impostor >= t)
        dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
        if dcf < min_dcf:
            min_dcf = dcf

    # Normalize by best possible cost
    c_default = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(min_dcf / c_default)


def run_evaluation():
    """Run full evaluation on available WAV files."""

    wav_files = sorted([f for f in os.listdir('.') if f.lower().endswith('.wav')])
    if len(wav_files) < 2:
        log.error("Need at least 2 WAV files for evaluation. Found: %s", wav_files)
        return

    log.info("Found %d WAV files for evaluation:", len(wav_files))
    for f in wav_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        log.info("  %s (%.1f MB)", f, size_mb)

    log.info("Loading ECAPA-TDNN embedder...")
    embedder = EcapaTdnnEmbeddings()

    # Extract embeddings from all files
    file_embeddings = {}
    for wav_file in wav_files:
        log.info("Processing: %s", wav_file)
        audio = load_wav(wav_file)
        if audio is None or len(audio) < 16000:
            log.warning("  Skipping (too short or failed)")
            continue

        chunks = chunk_audio(audio, chunk_sec=5.0, hop_sec=2.5)
        embeddings = []
        for chunk in chunks:
            emb = embedder.extract_embedding(chunk, 16000)
            if not np.allclose(emb, 0):
                embeddings.append(emb)

        if embeddings:
            file_embeddings[wav_file] = embeddings
            log.info("  Extracted %d embeddings from %d chunks", len(embeddings), len(chunks))

    if len(file_embeddings) < 2:
        log.error("Need embeddings from at least 2 files")
        return

    # Compute all pairwise scores
    files = list(file_embeddings.keys())
    log.info("\n=== Computing pairwise scores ===")

    genuine_scores = []
    impostor_scores = []

    for i, file_a in enumerate(files):
        embs_a = file_embeddings[file_a]

        # Split embeddings: first half = enrollment, second half = test
        mid = max(1, len(embs_a) // 2)
        enroll_a = embs_a[:mid]
        test_a = embs_a[mid:]

        # Enrollment centroid
        centroid_a = np.mean(enroll_a, axis=0)
        centroid_a = centroid_a / (np.linalg.norm(centroid_a) + 1e-10)

        # Genuine scores: test embeddings from same file vs enrollment centroid
        for emb in test_a:
            score = float(np.dot(emb, centroid_a))
            genuine_scores.append(score)

        # Impostor scores: test embeddings from OTHER files vs this centroid
        for j, file_b in enumerate(files):
            if i == j:
                continue
            embs_b = file_embeddings[file_b]
            mid_b = max(1, len(embs_b) // 2)
            test_b = embs_b[mid_b:]
            for emb in test_b:
                score = float(np.dot(emb, centroid_a))
                impostor_scores.append(score)

    log.info("Genuine trials: %d", len(genuine_scores))
    log.info("Impostor trials: %d", len(impostor_scores))

    if not genuine_scores or not impostor_scores:
        log.error("Not enough trials for evaluation")
        return

    # Compute metrics
    eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    min_dcf = compute_min_dcf(genuine_scores, impostor_scores)

    log.info("\n" + "=" * 60)
    log.info("EVALUATION RESULTS (ECAPA-TDNN 192-D)")
    log.info("=" * 60)
    log.info("  Equal Error Rate (EER):  %.2f%%", eer * 100)
    log.info("  EER Threshold:           %.4f", eer_threshold)
    log.info("  min DCF (p=0.01):        %.4f", min_dcf)
    log.info("")
    log.info("  Genuine scores:  mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
             np.mean(genuine_scores), np.std(genuine_scores),
             np.min(genuine_scores), np.max(genuine_scores))
    log.info("  Impostor scores: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
             np.mean(impostor_scores), np.std(impostor_scores),
             np.min(impostor_scores), np.max(impostor_scores))
    log.info("=" * 60)

    # Per-file analysis
    log.info("\nPer-file speaker similarity matrix:")
    for i, fi in enumerate(files):
        mid_i = max(1, len(file_embeddings[fi]) // 2)
        centroid_i = np.mean(file_embeddings[fi][:mid_i], axis=0)
        centroid_i = centroid_i / (np.linalg.norm(centroid_i) + 1e-10)
        row = []
        for j, fj in enumerate(files):
            mid_j = max(1, len(file_embeddings[fj]) // 2)
            centroid_j = np.mean(file_embeddings[fj][:mid_j], axis=0)
            centroid_j = centroid_j / (np.linalg.norm(centroid_j) + 1e-10)
            row.append(float(np.dot(centroid_i, centroid_j)))
        labels = [os.path.basename(f)[:15] for f in files]
        log.info("  %15s: %s", labels[i], "  ".join(f"{s:.3f}" for s in row))

    return {
        'eer': eer,
        'eer_threshold': eer_threshold,
        'min_dcf': min_dcf,
        'genuine_scores': genuine_scores,
        'impostor_scores': impostor_scores,
    }


if __name__ == "__main__":
    run_evaluation()
