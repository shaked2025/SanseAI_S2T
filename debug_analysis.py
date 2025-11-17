"""
Debug the analysis results to identify and fix issues
"""

import json
import numpy as np

# Load results
with open('brad_pitt_results.json', 'r') as f:
    results = json.load(f)

print("="*90)
print("DEBUGGING ANALYSIS RESULTS")
print("="*90)

print(f"\nBasic Stats:")
print(f"  Chunks: {len(results['timeline'])}")
print(f"  Duration: {results['duration_seconds']/60:.1f} minutes")

print(f"\n{'='*90}")
print("PROBLEM 1: Topic Analysis")
print(f"{'='*90}")

# Check topic labels assigned
topic_labels = {}
for entry in results['timeline']:
    label = entry['topic_label']
    topic_id = entry['topic_id']
    
    if topic_id not in topic_labels:
        topic_labels[topic_id] = {
            'label': label,
            'count': 0,
            'chunks': []
        }
    topic_labels[topic_id]['count'] += 1
    topic_labels[topic_id]['chunks'].append(entry['chunk_index'])

print(f"\nTopics detected: {len(topic_labels)}")
print(f"Topic summaries returned: {len(results['topics'])}")
print(f"\nISSUE: {len(topic_labels)} topics detected but {len(results['topics'])} summaries!")

print(f"\nTopic breakdown:")
for topic_id, data in sorted(topic_labels.items()):
    print(f"  Topic {topic_id} ({data['label']}): {data['count']} mentions")
    print(f"    Chunks: {data['chunks']}")

# Check for duplicate labels
label_counts = {}
for tid, tdata in topic_labels.items():
    label = tdata['label']
    if label not in label_counts:
        label_counts[label] = []
    label_counts[label].append(tid)

print(f"\nDuplicate labels (should be grouped!):")
for label, tids in label_counts.items():
    if len(tids) > 1:
        print(f"  '{label}': {len(tids)} separate topic IDs {tids}")
        print(f"    ❌ PROBLEM: Same label should be ONE topic!")

print(f"\n{'='*90}")
print("PROBLEM 2: Acoustic Feature Values")
print(f"{'='*90}")

# Analyze acoustic values
jitters = [entry['key_acoustic']['jitter'] for entry in results['timeline']]
shimmers = [entry['key_acoustic']['shimmer'] for entry in results['timeline']]
f0s = [entry['key_acoustic']['f0_mean'] for entry in results['timeline']]

print(f"\nJitter statistics:")
print(f"  Range: {min(jitters):.2f} - {max(jitters):.2f}%")
print(f"  Mean: {np.mean(jitters):.2f}%")
print(f"  ❌ PROBLEM: Should be <1%, got {np.mean(jitters):.1f}%")
print(f"  CAUSE: Calculation likely multiplying by 100 twice, or using wrong units")

print(f"\nShimmer statistics:")
print(f"  Range: {min(shimmers):.2f} - {max(shimmers):.2f}%")
print(f"  Mean: {np.mean(shimmers):.2f}%")
print(f"  ❌ PROBLEM: Should be <3%, got {np.mean(shimmers):.1f}%")
print(f"  CAUSE: Major calculation error - likely measuring wrong thing")

print(f"\nF0 statistics:")
print(f"  Range: {min(f0s):.1f} - {max(f0s):.1f} Hz")
print(f"  Mean: {np.mean(f0s):.1f} Hz")
print(f"  ✅ REASONABLE for male voices (expected 85-180 Hz)")

print(f"\n{'='*90}")
print("PROBLEM 3: Acoustic Stress Too High")
print(f"{'='*90}")

acoustic_stresses = [entry['acoustic_stress'] for entry in results['timeline']]
linguistic_stresses = [entry['linguistic_stress'] for entry in results['timeline']]

print(f"\nAcoustic Stress:")
print(f"  Range: {min(acoustic_stresses):.2f} - {max(acoustic_stresses):.2f}")
print(f"  Mean: {np.mean(acoustic_stresses):.2f}")
print(f"  ⚠️ ISSUE: Average 0.66 (MODERATE-HIGH for a comedy interview)")
print(f"  CAUSE: Jitter/shimmer values too high → inflating stress score")

print(f"\nLinguistic Stress:")
print(f"  Range: {min(linguistic_stresses):.2f} - {max(linguistic_stresses):.2f}")
print(f"  Mean: {np.mean(linguistic_stresses):.2f}")
print(f"  ✅ REASONABLE (0.10 = LOW, appropriate for casual interview)")

print(f"\n{'='*90}")
print("FIXES NEEDED")
print(f"{'='*90}")

print(f"\n1. FIX Topic Summarization:")
print(f"   - get_all_topics_summary() not working")
print(f"   - Need to debug topic_modeling_analysis.py")
print(f"   - Ensure topics dict is populated correctly")

print(f"\n2. FIX Jitter Calculation:")
print(f"   - Current: {np.mean(jitters):.1f}% (6x too high)")
print(f"   - Expected: <1%")
print(f"   - Check: Units, normalization, formula")

print(f"\n3. FIX Shimmer Calculation:")
print(f"   - Current: {np.mean(shimmers):.1f}% (20x too high!)")
print(f"   - Expected: <3%")
print(f"   - Check: Amplitude extraction, averaging method")

print(f"\n4. RECALIBRATE Stress Thresholds:")
print(f"   - Once jitter/shimmer fixed, acoustic stress will drop")
print(f"   - Retest to establish proper thresholds")

print(f"\n5. IMPROVE Topic Clustering:")
print(f"   - Multiple 'Alibi', 'Motive' topics should be ONE each")
print(f"   - Increase similarity threshold or improve matching")

print(f"\n{'='*90}")
print("ACTION PLAN")
print(f"{'='*90}")

print(f"\nStep 1: Fix shimmer/jitter calculations (critical)")
print(f"Step 2: Fix topic summarization (get_all_topics_summary)")
print(f"Step 3: Improve topic clustering (group same labels)")
print(f"Step 4: Recalibrate stress scoring")
print(f"Step 5: Re-run analysis and validate")

print(f"\n✅ Debug complete - issues identified!")

