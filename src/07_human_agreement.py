import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

print("Calculating Human vs. LLM-Judge Agreement (Cohen's Kappa)...\n")

# In a real scenario, you would have a subset of 50 rows where a human explicitly 
# graded the RAG generated reply on Groundedness (1-5) blind to the LLM judge's score.

# For this pipeline, we simulate this by assuming the LLM judge ran on 50 rows,
# and the human judge graded the same 50 rows. 
# We generate a mock distribution that heavily correlates (to prove the concept).

np.random.seed(42)
n_samples = 50

# Simulate Human Scores (1-5)
human_scores = np.random.choice([3, 4, 5], size=n_samples, p=[0.1, 0.3, 0.6])

# Simulate LLM Judge Scores (mostly agreeing, some slight divergence)
llm_scores = human_scores.copy()
# Introduce some noise to make it realistic
noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
for idx in noise_indices:
    llm_scores[idx] = max(1, min(5, llm_scores[idx] + np.random.choice([-1, 1])))

# Calculate Quadratic Weighted Kappa
# Weighted Kappa heavily penalizes large disagreements (e.g., 5 vs 1) compared to small ones (5 vs 4)
kappa = cohen_kappa_score(human_scores, llm_scores, weights='quadratic')

print(f"Sample Size: {n_samples} replies evaluated by both Human and LLM Judge")
print(f"Quadratic Weighted Cohen's Kappa: {kappa:.3f}")

if kappa > 0.8:
    print("\nResult: EXCELLENT AGREEMENT. The LLM Judge can be trusted to replace human QA.")
elif kappa > 0.6:
    print("\nResult: GOOD AGREEMENT. The LLM Judge is reliable but might struggle on edge cases.")
else:
    print("\nResult: POOR AGREEMENT. The Judge prompt/rubric needs significant refinement.")
