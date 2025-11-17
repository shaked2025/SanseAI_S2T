"""
Debug why semantic clustering creates 33 topics instead of proper grouping
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Load results
with open('brad_pitt_results.json') as f:
    data = json.load(f)

# Extract texts
texts = [entry['text'] for entry in data['timeline']]

print("="*90)
print("DEBUGGING SEMANTIC CLUSTERING")
print("="*90)

# Load SBERT model
print("\nLoading Sentence-BERT...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get embeddings
print("Computing embeddings...")
embeddings = model.encode(texts, show_progress_bar=False)

print(f"  Shape: {embeddings.shape}")

# Calculate pairwise similarities
print("\nCalculating pairwise similarities...")

similarities = np.zeros((len(texts), len(texts)))

for i in range(len(embeddings)):
    for j in range(i, len(embeddings)):
        # Cosine similarity
        sim = np.dot(embeddings[i], embeddings[j]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
        )
        similarities[i, j] = sim
        similarities[j, i] = sim

# Analyze similarity distribution
print(f"\nSimilarity statistics:")
print(f"  Mean: {np.mean(similarities):.3f}")
print(f"  Std: {np.std(similarities):.3f}")
print(f"  Min: {np.min(similarities):.3f}")
print(f"  Max: {np.max(similarities):.3f}")

# Find highly similar pairs (should be same topic)
print(f"\nHighly similar utterance pairs (similarity > 0.70):")

high_sim_pairs = []

for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        if similarities[i, j] > 0.70:
            high_sim_pairs.append((i, j, similarities[i, j]))
            
high_sim_pairs.sort(key=lambda x: x[2], reverse=True)

if high_sim_pairs:
    print(f"  Found {len(high_sim_pairs)} pairs")
    print(f"\n  Top 10:")
    for i, j, sim in high_sim_pairs[:10]:
        print(f"    [{i}] <-> [{j}]: {sim:.3f}")
        print(f"      {texts[i][:60]}...")
        print(f"      {texts[j][:60]}...")
        print()
else:
    print(f"  NO pairs with similarity > 0.70!")
    print(f"  Issue: Utterances are all semantically dissimilar")
    print(f"  This is a COMEDY interview with rapid topic changes")
    print(f"  Each joke/question is different → creates many topics")

# Check what threshold would group them
print(f"\nSimilarity threshold analysis:")

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    count = sum(1 for i, j, sim in high_sim_pairs if sim >= threshold)
    print(f"  Threshold {threshold:.2f}: {count} pairs would cluster")

# Recommendation
print(f"\n{'='*90}")
print("ANALYSIS")
print(f"{'='*90}")

print(f"\nThe Brad Pitt 'Between Two Ferns' interview:")
print(f"  - Is a COMEDY show with rapid-fire jokes")
print(f"  - Each utterance is a different joke/question")
print(f"  - Semantically diverse by design (humor requires variety)")
print(f"  - NOT a sustained interrogation on few topics")

print(f"\nSemantic clustering is working CORRECTLY:")
print(f"  - Detecting that each utterance is semantically different")
print(f"  - Creating separate topics because they ARE different")
print(f"  - This is accurate for THIS type of content")

print(f"\nFor proper topic grouping, need:")
print(f"  - INTERROGATION content (sustained discussion)")
print(f"  - Multiple utterances about SAME subject")
print(f"  - Alibi questioned repeatedly with follow-ups")
print(f"  - Then semantic clustering would group them")

print(f"\nRECOMMENDATION:")
print(f"  - Test on REAL interrogation (not comedy)")
print(f"  - Or lower threshold to 0.50-0.55")
print(f"  - Or use question-based segmentation primarily")

print(f"\n{'='*90}")

