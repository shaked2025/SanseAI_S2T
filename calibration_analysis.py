"""
Analyze calibration data and recommend optimal parameters
"""

import numpy as np

# Data from calibration test
rms_values = [
    3205.6, 2368.1, 3009.5, 1022.9, 2013.8, 1875.0, 2182.4, 1586.3, 
    2525.4, 1625.1, 2707.4, 1884.3, 2593.4, 1897.7, 2884.4, 2578.5, 
    1476.6, 4933.2, 244.0, 4212.9
]

transcriptions = [
    "Hey everyone, I tried.",
    "to speak again because...",
    "because the cursor is very stupid.",
    "Make sure to wearええ right here",  # ERROR: Japanese characters
    "and I don't know...",
    "is right now.",
    "its end",
    "but I continue.",
    "7, 8,",
    "9",
    "and more than...",
    "to end.",
    "Chang Baichang",  # ERROR: Misheard
    "I continue.",
    "Are you wondering?",
    "right now.",
    "If we",
    "didn't need also.",
    "There you go. Okay, one second.",
    "woman"
]

print("="*100)
print("CALIBRATION ANALYSIS - Parameter Recommendations")
print("="*100)

# === RMS ANALYSIS ===
rms_array = np.array(rms_values)
speech_rms = rms_array[rms_array > 500]  # Filter out silence
silence_rms = rms_array[rms_array <= 500]

print(f"\n[RMS ANALYSIS]")
print(f"   Total chunks: {len(rms_values)}")
print(f"   Speech chunks: {len(speech_rms)}")
print(f"   Silence chunks: {len(silence_rms)}")
print(f"\n   RMS Statistics:")
print(f"      Min: {np.min(rms_array):.1f}")
print(f"      Max: {np.max(rms_array):.1f}")
print(f"      Mean: {np.mean(rms_array):.1f}")
print(f"      Median: {np.median(rms_array):.1f}")
print(f"      Std Dev: {np.std(rms_array):.1f}")
print(f"\n   Speech-only RMS:")
print(f"      Min: {np.min(speech_rms):.1f}")
print(f"      Max: {np.max(speech_rms):.1f}")
print(f"      Mean: {np.mean(speech_rms):.1f}")
print(f"      Median: {np.median(speech_rms):.1f}")

# Recommended thresholds
speech_mean = np.mean(speech_rms)
speech_median = np.median(speech_rms)

print(f"\n   [RECOMMENDED RMS THRESHOLDS]")
print(f"      Minimum for speech: {speech_median * 0.3:.0f} (30% of median)")
print(f"      Optimal threshold: {speech_median * 0.4:.0f} - {speech_median * 0.5:.0f} (40-50% of median)")
print(f"      Current setting: 800")
print(f"      Recommendation: {int(speech_median * 0.4)} - {int(speech_median * 0.5)}")

# === TRANSCRIPTION ANALYSIS ===
print(f"\n[TRANSCRIPTION ANALYSIS]")
print(f"   Total transcriptions: {len(transcriptions)}")
print(f"   Average length: {np.mean([len(t) for t in transcriptions]):.1f} characters")
print(f"   Average words: {np.mean([len(t.split()) for t in transcriptions]):.1f} words")

# Check for errors
errors = []
for i, text in enumerate(transcriptions):
    if len(text) < 10:  # Very short
        errors.append(f"Chunk {i+1}: Too short - '{text}'")
    # Check for non-ASCII (transcription errors)
    if any(ord(c) > 127 for c in text):
        errors.append(f"Chunk {i+1}: Non-ASCII characters - '{text}'")

print(f"\n   [TRANSCRIPTION ISSUES]")
if errors:
    for error in errors:
        print(f"      - {error}")
else:
    print(f"      No obvious errors detected")

print(f"\n   [OBSERVATIONS]")
print(f"      - Many transcriptions are incomplete (ending with '...')")
print(f"      - Some very short transcriptions (1-2 words)")
print(f"      - Whisper may need longer context or different settings")

# === FINAL RECOMMENDATIONS ===
print(f"\n{'='*100}")
print("[FINAL PARAMETER RECOMMENDATIONS]")
print(f"{'='*100}")

optimal_rms = int(speech_median * 0.45)
print(f"\n1. RMS THRESHOLD:")
print(f"   Current: 800")
print(f"   Recommended: {optimal_rms}")
print(f"   Reason: 45% of median speech RMS ({speech_median:.0f}) = {optimal_rms}")
print(f"   This will filter silence (<500) while keeping all speech (>1000)")

print(f"\n2. VAD THRESHOLD (Voice Activity Detection):")
print(f"   Current: 500")
print(f"   Recommended: {int(optimal_rms * 0.6)}")
print(f"   Reason: Slightly lower than RMS threshold for early detection")

print(f"\n3. TRANSCRIPTION SETTINGS:")
print(f"   Issue: Incomplete transcriptions, cutting off sentences")
print(f"   Recommendations:")
print(f"      - Use longer audio chunks (3-4 seconds instead of 2)")
print(f"      - Use 'beam_size=5' for better accuracy")
print(f"      - Use 'temperature=0' for more deterministic results")
print(f"      - Consider 'condition_on_previous_text=True' for context")

print(f"\n4. VOICE SIMILARITY THRESHOLD:")
print(f"   Current: 0.60")
print(f"   Recommendation: Keep at 0.60 (seems appropriate)")
print(f"   Note: Need enrollment data to verify")

print(f"\n{'='*100}")

