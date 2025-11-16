"""
SIMPLE BUT ROBUST Speaker Verification
Based on what ACTUALLY WORKS from the logs

Key insight from testing:
- Direct cosine similarity WORKS (shows 0.91 for correct speaker!)
- Complex multi-metric fusion BREAKS IT (shows 0.54)
- SVM REJECTS EVERYTHING

Solution: Use SIMPLE direct cosine similarity with smart thresholding
This is what production systems actually use!
"""

import numpy as np


class SimpleRobustVerifier:
    """
    Simple but robust speaker verification
    Uses direct cosine similarity with adaptive thresholds
    PROVEN to work based on console logs
    """
    
    def __init__(self, base_threshold=0.64):
        """
        Args:
            base_threshold: Base similarity threshold (0.64 - middle of 0.60-0.70 perfect range)
        """
        self.base_threshold = base_threshold
        self.rejection_stats = {
            'total_checks': 0,
            'accepted': 0,
            'rejected': 0,
            'rejection_reasons': {}
        }
        
    def verify_speaker(self, test_embedding, enrolled_speakers, audio_quality=1.0):
        """
        Simple direct verification
        
        Args:
            test_embedding: Test embedding
            enrolled_speakers: Dict of enrolled speaker profiles
            audio_quality: Audio quality score (0-1)
            
        Returns:
            (accept: bool, best_speaker_key, best_speaker_name, similarity, reason)
        """
        self.rejection_stats['total_checks'] += 1
        
        if not enrolled_speakers:
            return False, None, "Unknown", 0.0, "No enrolled speakers"
            
        # Calculate DIRECT cosine similarity with each enrolled speaker
        similarities = {}
        
        for speaker_key, profile in enrolled_speakers.items():
            # Direct dot product (both embeddings are normalized)
            similarity = np.dot(test_embedding, profile['mean_embedding'])
            similarities[speaker_key] = similarity
            
        # Find best match
        best_speaker_key = max(similarities, key=similarities.get)
        best_similarity = similarities[best_speaker_key]
        best_profile = enrolled_speakers[best_speaker_key]
        best_name = best_profile['name']
        
        # Get second best for comparison
        sorted_sims = sorted(similarities.values(), reverse=True)
        second_best_sim = sorted_sims[1] if len(sorted_sims) > 1 else 0.0
        
        # === DECISION LOGIC ===
        
        # Rule 1: Must exceed base threshold
        threshold = self._calculate_threshold(audio_quality, best_profile)
        
        if best_similarity < threshold:
            reason = f"Below threshold ({best_similarity:.3f} < {threshold:.3f})"
            self.rejection_stats['rejected'] += 1
            self._record_rejection(reason)
            return False, best_speaker_key, best_name, best_similarity, reason
            
        # Rule 2: Must be clearly better than second choice (margin requirement)
        # This prevents accepting when all similarities are similar (unknown speaker)
        margin = best_similarity - second_best_sim
        
        # PRAGMATIC APPROACH based on real data:
        # If similarity is clearly above unknown range (>0.68), accept regardless of margin
        # Only check margin for borderline cases
        
        if best_similarity < 0.68 and len(enrolled_speakers) > 1:
            # Only enforce margin check for borderline similarities
            min_margin = 0.08
            
            if margin < min_margin:
                reason = f"Borderline with small margin ({best_similarity:.3f}, margin: {margin:.3f})"
                self.rejection_stats['rejected'] += 1
                self._record_rejection(reason)
                return False, best_speaker_key, best_name, best_similarity, reason
        
        # High similarity (>=0.68) bypasses margin check (clearly enrolled speaker)
            
        # Rule 3: Absolute minimum threshold (even with good quality)
        absolute_min = 0.58  # Slightly lower to account for gender/voice differences
        
        if best_similarity < absolute_min:
            reason = f"Below absolute minimum ({best_similarity:.3f} < {absolute_min:.3f})"
            self.rejection_stats['rejected'] += 1
            self._record_rejection(reason)
            return False, best_speaker_key, best_name, best_similarity, reason
            
        # All checks passed - ACCEPT
        self.rejection_stats['accepted'] += 1
        
        acceptance_rate = self.rejection_stats['accepted'] / self.rejection_stats['total_checks'] * 100
        reason = f"Accepted (margin: {margin:.3f}, quality: {audio_quality:.2f}, rate: {acceptance_rate:.1f}%)"
        
        return True, best_speaker_key, best_name, best_similarity, reason
        
    def _calculate_threshold(self, audio_quality, speaker_profile):
        """Calculate adaptive threshold based on quality and speaker consistency"""
        
        # Base threshold adjusted for quality
        # Use CONSERVATIVE adjustments to ensure robustness
        if audio_quality >= 0.8:
            quality_threshold = self.base_threshold - 0.02  # 0.62 - slightly lenient for good quality
        elif audio_quality >= 0.6:
            quality_threshold = self.base_threshold  # 0.64 - optimal base
        elif audio_quality >= 0.4:
            quality_threshold = self.base_threshold + 0.02  # 0.66 - slightly stricter
        else:
            quality_threshold = self.base_threshold + 0.04  # 0.68 - stricter for bad quality
            
        # Adjust for speaker consistency (if available)
        speaker_std = speaker_profile.get('std', 0.10)
        
        # If speaker has high variance in enrollment, be more lenient
        if speaker_std > 0.15:  # High variability
            consistency_adjustment = -0.05  # Lower threshold
        elif speaker_std < 0.08:  # Very consistent
            consistency_adjustment = +0.02  # Can be slightly stricter
        else:
            consistency_adjustment = 0.0
            
        final_threshold = quality_threshold + consistency_adjustment
        
        # Clamp to reasonable range
        final_threshold = np.clip(final_threshold, 0.55, 0.75)
        
        return final_threshold
        
    def _record_rejection(self, reason):
        """Track rejection reasons for analysis"""
        if reason not in self.rejection_stats['rejection_reasons']:
            self.rejection_stats['rejection_reasons'][reason] = 0
        self.rejection_stats['rejection_reasons'][reason] += 1
        
    def get_statistics(self):
        """Get verification statistics"""
        return self.rejection_stats

