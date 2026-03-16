"""
Offline Simulation Test — Full Speaker ID Pipeline

Simulates a real 2-person interview using actual recordings:
1. Loads WAV files, finds speech segments with Silero-VAD
2. Extracts ECAPA-TDNN embeddings per segment
3. Clusters Kavin Interview into 2 speakers via spectral clustering
4. Enrolls each speaker using first N segments (simulates enrollment phase)
5. Runs verification on remaining segments (simulates live interview)
6. Reports per-speaker accuracy, FAR, FRR, confusion matrix

Also tests unknown speaker rejection using a third recording (JiaJun).
"""

import numpy as np
import soundfile as sf
import torch
import logging
import sys
import time
from collections import defaultdict
from scipy import signal as scipy_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

import torchaudio_compat  # noqa: F401

from speaker_diarization_robust import EcapaTdnnEmbeddings, EMBEDDING_DIM
from speaker_enrollment import SpeakerEnrollment
from simple_robust_verification import SimpleRobustVerifier
from audio_capture import SileroVAD
from antispoof import ReplayDetector


def load_and_resample(filepath, target_sr=16000):
    """Load WAV, convert to mono float32 at target_sr."""
    audio, sr = sf.read(filepath, dtype='float32')
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        n = int(len(audio) * target_sr / sr)
        audio = scipy_signal.resample(audio, n).astype(np.float32)
    return audio, target_sr


def find_speech_segments(audio, sr, vad_model, vad_utils, min_dur_ms=1000, min_silence_ms=500):
    """Find speech segments using Silero-VAD."""
    wav = torch.tensor(audio)
    get_speech_timestamps = vad_utils[0]
    stamps = get_speech_timestamps(
        wav, vad_model, sampling_rate=sr,
        min_speech_duration_ms=min_dur_ms,
        min_silence_duration_ms=min_silence_ms,
    )
    segments = []
    for s in stamps:
        start, end = s['start'], s['end']
        dur = (end - start) / sr
        if dur >= 1.0:
            segments.append({
                'start': start, 'end': end,
                'start_sec': start / sr, 'end_sec': end / sr,
                'duration': dur,
                'audio': (audio[start:end] * 32768).astype(np.int16),
            })
    return segments


def extract_segment_embeddings(segments, embedder):
    """Extract embeddings for each segment."""
    for seg in segments:
        emb = embedder.extract_embedding(seg['audio'], 16000)
        seg['embedding'] = emb
        seg['valid'] = not np.allclose(emb, 0)
    return [s for s in segments if s['valid']]


def cluster_speakers(segments, n_speakers=2):
    """Cluster segments into speakers using cosine similarity + spectral clustering."""
    from sklearn.cluster import SpectralClustering

    embeddings = np.array([s['embedding'] for s in segments])
    n = len(embeddings)

    # Cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T
    # Ensure positive for spectral clustering
    sim_matrix = (sim_matrix + 1) / 2

    clustering = SpectralClustering(
        n_clusters=n_speakers, affinity='precomputed',
        random_state=42, n_init=10
    )
    labels = clustering.fit_predict(sim_matrix)

    for i, seg in enumerate(segments):
        seg['cluster'] = int(labels[i])

    # Report
    for c in range(n_speakers):
        cluster_segs = [s for s in segments if s['cluster'] == c]
        total_dur = sum(s['duration'] for s in cluster_segs)
        log.info("  Cluster %d: %d segments, %.1f seconds total", c, len(cluster_segs), total_dur)

    return segments


def run_simulation():
    log.info("=" * 70)
    log.info("OFFLINE SPEAKER ID SIMULATION TEST")
    log.info("=" * 70)

    # --- Load models ---
    log.info("Loading models...")
    t0 = time.time()

    embedder = EcapaTdnnEmbeddings()
    vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', model='silero_vad', trust_repo=True)

    log.info("Models loaded in %.1fs", time.time() - t0)

    # --- Load & segment Kavin Interview (2-speaker conversation) ---
    log.info("\n--- Loading Kavin Interview (2-speaker conversation) ---")
    kavin_audio, sr = load_and_resample("Kavin Interview77 (1).wav")
    kavin_segments = find_speech_segments(kavin_audio, sr, vad_model, vad_utils,
                                          min_dur_ms=1500, min_silence_ms=500)
    log.info("Found %d speech segments (>1.5s)", len(kavin_segments))

    log.info("Extracting embeddings...")
    t0 = time.time()
    kavin_segments = extract_segment_embeddings(kavin_segments, embedder)
    log.info("Extracted %d valid embeddings in %.1fs", len(kavin_segments), time.time() - t0)

    # --- Cluster into 2 speakers ---
    log.info("\nClustering into 2 speakers...")
    kavin_segments = cluster_speakers(kavin_segments, n_speakers=2)

    # --- Load JiaJun as unknown speaker ---
    log.info("\n--- Loading JiaJun (unknown speaker for rejection test) ---")
    jiajun_audio, sr2 = load_and_resample("JiaJun_video_3 1.wav")
    jiajun_segments = find_speech_segments(jiajun_audio, sr2, vad_model, vad_utils,
                                            min_dur_ms=1500, min_silence_ms=500)
    jiajun_segments = extract_segment_embeddings(jiajun_segments, embedder)
    log.info("JiaJun: %d valid segments", len(jiajun_segments))

    # --- Load temp_audio as another unknown ---
    log.info("\n--- Loading temp_audio (another unknown speaker) ---")
    temp_audio, sr3 = load_and_resample("temp_audio.wav")
    temp_segments = find_speech_segments(temp_audio, sr3, vad_model, vad_utils,
                                          min_dur_ms=1500, min_silence_ms=500)
    temp_segments = extract_segment_embeddings(temp_segments, embedder)
    log.info("temp_audio: %d valid segments", len(temp_segments))

    # =====================================================================
    # ENROLLMENT PHASE
    # =====================================================================
    log.info("\n" + "=" * 70)
    log.info("ENROLLMENT PHASE")
    log.info("=" * 70)

    # Split Kavin segments: first 4 segments per cluster for enrollment
    ENROLL_COUNT = 4

    cluster_0_segs = [s for s in kavin_segments if s['cluster'] == 0]
    cluster_1_segs = [s for s in kavin_segments if s['cluster'] == 1]

    if len(cluster_0_segs) < ENROLL_COUNT or len(cluster_1_segs) < ENROLL_COUNT:
        log.error("Not enough segments per cluster for enrollment")
        return

    enroll_0 = cluster_0_segs[:ENROLL_COUNT]
    enroll_1 = cluster_1_segs[:ENROLL_COUNT]
    test_0 = cluster_0_segs[ENROLL_COUNT:]
    test_1 = cluster_1_segs[ENROLL_COUNT:]

    log.info("Speaker A (cluster 0): %d enroll, %d test segments", len(enroll_0), len(test_0))
    log.info("Speaker B (cluster 1): %d enroll, %d test segments", len(enroll_1), len(test_1))

    # Enroll using the full pipeline
    enrollment = SpeakerEnrollment(embedder, min_samples=ENROLL_COUNT)

    for label, enroll_segs, name in [
        ('speaker_a', enroll_0, 'Speaker_A'),
        ('speaker_b', enroll_1, 'Speaker_B'),
    ]:
        enrollment.start_enrollment(label, name, 'Interviewee')
        for seg in enroll_segs:
            ok, q, msg = enrollment.add_enrollment_sample(label, seg['audio'], 16000)
            log.info("  %s enrollment: %s", name, msg)
        ok, quality, msg = enrollment.complete_enrollment(label)
        log.info("  %s: %s", name, msg)

    enrolled = enrollment.get_enrolled_speakers()

    # Show inter-speaker similarity
    emb_a = enrolled['speaker_a']['mean_embedding']
    emb_b = enrolled['speaker_b']['mean_embedding']
    inter_sim = float(np.dot(emb_a, emb_b))
    log.info("\nInter-speaker similarity: %.4f (lower = better separation)", inter_sim)

    # =====================================================================
    # VERIFICATION PHASE
    # =====================================================================
    log.info("\n" + "=" * 70)
    log.info("VERIFICATION PHASE")
    log.info("=" * 70)

    verifier = SimpleRobustVerifier()  # Uses calibrated default threshold

    results = {
        'speaker_a': {'correct': 0, 'wrong': 0, 'rejected': 0, 'scores': []},
        'speaker_b': {'correct': 0, 'wrong': 0, 'rejected': 0, 'scores': []},
        'unknown_jiajun': {'correctly_rejected': 0, 'falsely_accepted': 0, 'scores': []},
        'unknown_temp': {'correctly_rejected': 0, 'falsely_accepted': 0, 'scores': []},
    }

    # --- Test Speaker A segments ---
    log.info("\n--- Testing Speaker A (cluster 0) segments ---")
    for seg in test_0:
        accept, key, name, score, reason = verifier.verify_speaker(
            seg['embedding'], enrolled
        )
        results['speaker_a']['scores'].append(score)
        if accept and key == 'speaker_a':
            results['speaker_a']['correct'] += 1
        elif accept and key != 'speaker_a':
            results['speaker_a']['wrong'] += 1
            log.warning("  MISIDENTIFIED as %s at %.1fs (score: %.3f)", name, seg['start_sec'], score)
        else:
            results['speaker_a']['rejected'] += 1
            log.info("  FALSE REJECT at %.1fs (score: %.3f) — %s", seg['start_sec'], score, reason)

    # --- Test Speaker B segments ---
    log.info("\n--- Testing Speaker B (cluster 1) segments ---")
    for seg in test_1:
        accept, key, name, score, reason = verifier.verify_speaker(
            seg['embedding'], enrolled
        )
        results['speaker_b']['scores'].append(score)
        if accept and key == 'speaker_b':
            results['speaker_b']['correct'] += 1
        elif accept and key != 'speaker_b':
            results['speaker_b']['wrong'] += 1
            log.warning("  MISIDENTIFIED as %s at %.1fs (score: %.3f)", name, seg['start_sec'], score)
        else:
            results['speaker_b']['rejected'] += 1
            log.info("  FALSE REJECT at %.1fs (score: %.3f) — %s", seg['start_sec'], score, reason)

    # --- Test JiaJun (should be REJECTED) ---
    log.info("\n--- Testing JiaJun (UNKNOWN — should be rejected) ---")
    for seg in jiajun_segments:
        accept, key, name, score, reason = verifier.verify_speaker(
            seg['embedding'], enrolled
        )
        results['unknown_jiajun']['scores'].append(score)
        if accept:
            results['unknown_jiajun']['falsely_accepted'] += 1
            log.warning("  FALSE ACCEPT as %s at %.1fs (score: %.3f)", name, seg['start_sec'], score)
        else:
            results['unknown_jiajun']['correctly_rejected'] += 1

    # --- Test temp_audio (should be REJECTED) ---
    log.info("\n--- Testing temp_audio (UNKNOWN — should be rejected) ---")
    for seg in temp_segments:
        accept, key, name, score, reason = verifier.verify_speaker(
            seg['embedding'], enrolled
        )
        results['unknown_temp']['scores'].append(score)
        if accept:
            results['unknown_temp']['falsely_accepted'] += 1
            log.warning("  FALSE ACCEPT as %s at %.1fs (score: %.3f)", name, seg['start_sec'], score)
        else:
            results['unknown_temp']['correctly_rejected'] += 1

    # =====================================================================
    # RESULTS
    # =====================================================================
    log.info("\n" + "=" * 70)
    log.info("SIMULATION RESULTS")
    log.info("=" * 70)

    # Speaker A
    a = results['speaker_a']
    total_a = a['correct'] + a['wrong'] + a['rejected']
    acc_a = a['correct'] / total_a * 100 if total_a else 0
    log.info("\nSpeaker A (enrolled):")
    log.info("  Correct: %d/%d (%.1f%%)", a['correct'], total_a, acc_a)
    log.info("  Misidentified: %d", a['wrong'])
    log.info("  False Rejected: %d (FRR: %.1f%%)", a['rejected'], a['rejected'] / total_a * 100 if total_a else 0)
    if a['scores']:
        log.info("  Scores: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                 np.mean(a['scores']), np.std(a['scores']), np.min(a['scores']), np.max(a['scores']))

    # Speaker B
    b = results['speaker_b']
    total_b = b['correct'] + b['wrong'] + b['rejected']
    acc_b = b['correct'] / total_b * 100 if total_b else 0
    log.info("\nSpeaker B (enrolled):")
    log.info("  Correct: %d/%d (%.1f%%)", b['correct'], total_b, acc_b)
    log.info("  Misidentified: %d", b['wrong'])
    log.info("  False Rejected: %d (FRR: %.1f%%)", b['rejected'], b['rejected'] / total_b * 100 if total_b else 0)
    if b['scores']:
        log.info("  Scores: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                 np.mean(b['scores']), np.std(b['scores']), np.min(b['scores']), np.max(b['scores']))

    # Unknown rejection
    j = results['unknown_jiajun']
    total_j = j['correctly_rejected'] + j['falsely_accepted']
    log.info("\nJiaJun (UNKNOWN — should be 100%% rejected):")
    log.info("  Correctly rejected: %d/%d", j['correctly_rejected'], total_j)
    log.info("  Falsely accepted (FAR): %d (%.1f%%)", j['falsely_accepted'],
             j['falsely_accepted'] / total_j * 100 if total_j else 0)
    if j['scores']:
        log.info("  Scores: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                 np.mean(j['scores']), np.std(j['scores']), np.min(j['scores']), np.max(j['scores']))

    t = results['unknown_temp']
    total_t = t['correctly_rejected'] + t['falsely_accepted']
    log.info("\ntemp_audio (UNKNOWN — should be 100%% rejected):")
    log.info("  Correctly rejected: %d/%d", t['correctly_rejected'], total_t)
    log.info("  Falsely accepted (FAR): %d (%.1f%%)", t['falsely_accepted'],
             t['falsely_accepted'] / total_t * 100 if total_t else 0)
    if t['scores']:
        log.info("  Scores: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                 np.mean(t['scores']), np.std(t['scores']), np.min(t['scores']), np.max(t['scores']))

    # Overall
    total_enrolled_correct = a['correct'] + b['correct']
    total_enrolled = total_a + total_b
    total_unknown_rejected = j['correctly_rejected'] + t['correctly_rejected']
    total_unknown = total_j + total_t

    overall_acc = total_enrolled_correct / total_enrolled * 100 if total_enrolled else 0
    overall_rej = total_unknown_rejected / total_unknown * 100 if total_unknown else 0
    total_misid = a['wrong'] + b['wrong']
    total_false_accept = j['falsely_accepted'] + t['falsely_accepted']

    log.info("\n" + "=" * 70)
    log.info("OVERALL METRICS")
    log.info("=" * 70)
    log.info("  Enrolled speaker accuracy:   %d/%d = %.1f%%", total_enrolled_correct, total_enrolled, overall_acc)
    log.info("  Enrolled speaker FRR:        %d/%d = %.1f%%",
             a['rejected'] + b['rejected'], total_enrolled,
             (a['rejected'] + b['rejected']) / total_enrolled * 100 if total_enrolled else 0)
    log.info("  Misidentification rate:      %d/%d = %.1f%%", total_misid, total_enrolled,
             total_misid / total_enrolled * 100 if total_enrolled else 0)
    log.info("  Unknown rejection rate:      %d/%d = %.1f%%", total_unknown_rejected, total_unknown, overall_rej)
    log.info("  False acceptance rate (FAR): %d/%d = %.1f%%", total_false_accept, total_unknown,
             total_false_accept / total_unknown * 100 if total_unknown else 0)
    log.info("=" * 70)

    # Diagnostic: print all score distributions
    all_genuine = a['scores'] + b['scores']
    all_impostor = j['scores'] + t['scores']
    if all_genuine and all_impostor:
        log.info("\nScore distributions:")
        log.info("  Genuine  (enrolled speakers): mean=%.3f, std=%.3f, [%.3f — %.3f]",
                 np.mean(all_genuine), np.std(all_genuine), np.min(all_genuine), np.max(all_genuine))
        log.info("  Impostor (unknown speakers):  mean=%.3f, std=%.3f, [%.3f — %.3f]",
                 np.mean(all_impostor), np.std(all_impostor), np.min(all_impostor), np.max(all_impostor))

        # Suggest optimal threshold
        from evaluate_speaker_id import compute_eer
        eer, eer_thresh = compute_eer(all_genuine, all_impostor)
        log.info("\n  EER: %.2f%% at threshold %.4f", eer * 100, eer_thresh)
        log.info("  (Use this threshold as base_threshold for production)")

    return results


if __name__ == "__main__":
    run_simulation()
