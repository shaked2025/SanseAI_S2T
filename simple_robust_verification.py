"""
Production Speaker Verification — Direct Cosine Scoring

ECAPA-TDNN 192-D embeddings are L2-normalized. Cosine similarity = dot product.

Calibration from real data (offline_simulation_test.py, evaluate_speaker_id.py):
  Genuine pairs:  mean ~0.50, range [0.15 — 0.90]
  Impostor pairs: mean ~0.12, range [-0.15 — 0.25]
  EER threshold:  ~0.24

Decision pipeline:
1. Compute raw cosine similarity to each enrolled speaker
2. Apply quality-aware per-speaker threshold
3. Margin check between best and second-best match
4. Hard floor rejection for very low scores
"""

import numpy as np
import logging

log = logging.getLogger(__name__)

EMBEDDING_DIM = 192


class SimpleRobustVerifier:
    """
    Direct cosine verification for ECAPA-TDNN embeddings.

    No S-norm: with 2 enrolled speakers the cohort is too small for
    meaningful normalization. Raw cosine from ECAPA-TDNN already
    achieves 0.69% EER on VoxCeleb1.
    """

    def __init__(self, base_threshold=0.25):
        """
        Args:
            base_threshold: Cosine similarity threshold.
                Genuine pairs typically score 0.35-0.90.
                Impostor pairs typically score -0.15 to 0.21.
                0.25 gives strong unknown rejection with low FRR.
        """
        self.base_threshold = base_threshold
        self.rejection_stats = {
            'total_checks': 0, 'accepted': 0, 'rejected': 0, 'rejection_reasons': {}
        }

    def verify_speaker(self, test_embedding, enrolled_speakers, audio_quality=1.0):
        """
        Verify speaker identity using direct cosine similarity.

        Returns:
            (accept, best_speaker_key, best_speaker_name, best_score, reason)
        """
        self.rejection_stats['total_checks'] += 1

        if not enrolled_speakers:
            return False, None, "Unknown", 0.0, "No enrolled speakers"

        # Raw cosine similarity with each enrolled speaker
        scores = {}
        for speaker_key, profile in enrolled_speakers.items():
            scores[speaker_key] = float(np.dot(test_embedding, profile['mean_embedding']))

        best_key = max(scores, key=scores.get)
        best_score = scores[best_key]
        best_name = enrolled_speakers[best_key]['name']

        sorted_vals = sorted(scores.values(), reverse=True)
        second_best = sorted_vals[1] if len(sorted_vals) > 1 else -1.0
        margin = best_score - second_best

        # --- Decision rules (ordered from cheapest to most expensive) ---

        # Rule 1: Hard floor — reject clearly dissimilar embeddings
        absolute_min = 0.12 if audio_quality >= 0.5 else 0.18
        if best_score < absolute_min:
            return self._reject(best_key, best_name, best_score,
                                f"Below absolute min ({best_score:.3f} < {absolute_min:.3f})")

        # Rule 2: Per-speaker quality-aware threshold
        threshold = self._threshold(audio_quality, enrolled_speakers[best_key])
        if best_score < threshold:
            return self._reject(best_key, best_name, best_score,
                                f"Below threshold ({best_score:.3f} < {threshold:.3f})")

        # Rule 3: Margin check — only for very borderline scores near impostor range
        if len(enrolled_speakers) > 1 and best_score < 0.30:
            min_margin = 0.04
            if margin < min_margin:
                return self._reject(best_key, best_name, best_score,
                                    f"Small margin ({best_score:.3f}, margin: {margin:.3f} < {min_margin:.3f})")

        # Accepted
        self.rejection_stats['accepted'] += 1
        rate = self.rejection_stats['accepted'] / self.rejection_stats['total_checks'] * 100
        reason = f"Accepted (cos={best_score:.3f}, margin={margin:.3f}, q={audio_quality:.2f}, rate={rate:.0f}%)"
        return True, best_key, best_name, best_score, reason

    def _threshold(self, audio_quality, speaker_profile):
        """Adaptive threshold: base + quality adjustment + speaker consistency."""
        if audio_quality >= 0.8:
            t = self.base_threshold - 0.02
        elif audio_quality >= 0.6:
            t = self.base_threshold
        elif audio_quality >= 0.4:
            t = self.base_threshold + 0.03
        else:
            t = self.base_threshold + 0.05

        std = speaker_profile.get('std', 0.10)
        if std > 0.15:
            t -= 0.03  # High-variance speaker → be lenient
        elif std < 0.05:
            t += 0.02  # Very consistent speaker → can be stricter

        return float(np.clip(t, 0.18, 0.45))

    def _reject(self, key, name, score, reason):
        self.rejection_stats['rejected'] += 1
        rr = self.rejection_stats['rejection_reasons']
        rr[reason] = rr.get(reason, 0) + 1
        return False, key, name, score, reason

    def get_statistics(self):
        return self.rejection_stats
