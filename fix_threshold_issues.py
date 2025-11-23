"""
Fix threshold issues based on research findings

Key issues identified:
1. Thresholds too high (0.65-0.85) - Resemblyzer typically works with 0.5-0.7
2. Z-norm might be distorting scores if not fitted properly
3. Strict mode (all 5 methods must pass) is too restrictive
4. Score normalization might reduce similarity scores incorrectly

Research findings:
- Resemblyzer cosine similarity typically ranges 0.4-0.9 for same speaker
- Optimal threshold is usually 0.5-0.6 for same speaker
- Thresholds above 0.7 cause high false rejection
- Need to balance false acceptance vs false rejection
"""

# Based on research, typical Resemblyzer thresholds:
# - Same speaker: 0.5-0.7 (usually 0.55-0.65)
# - Different speaker: 0.3-0.5
# - Optimal threshold: 0.55-0.60 for balanced performance

RECOMMENDED_BASE_THRESHOLD = 0.55  # Lower from 0.65
RECOMMENDED_MIN_THRESHOLD = 0.50   # Lower from 0.65
RECOMMENDED_MAX_THRESHOLD = 0.70   # Lower from 0.85

# For per-speaker thresholds:
# - High quality: 0.60-0.70 (not 0.70-0.85)
# - Medium quality: 0.55-0.65
# - Lower quality: 0.50-0.60

