import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os

print("Loading dataset...")
df = pd.read_csv("data/amazon_pairs.csv")

# Sample 5000 rows to find diverse examples
sample_df = df.sample(n=5000, random_state=42).reset_index(drop=True)

print("Embedding texts for stratified sampling...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sample_df['customer_message'].tolist())

print("Clustering to ensure we get a mix of all intents...")
kmeans = KMeans(n_clusters=6, random_state=42)
sample_df['cluster'] = kmeans.fit_predict(embeddings)

# Pick exactly 33 examples from each of the 6 clusters
golden_set = pd.DataFrame()
for i in range(6):
    cluster_sample = sample_df[sample_df['cluster'] == i].sample(n=33, random_state=42)
    golden_set = pd.concat([golden_set, cluster_sample])

# Add 2 extra to hit exactly 200
extra = sample_df[~sample_df.index.isin(golden_set.index)].sample(n=2, random_state=42)
golden_set = pd.concat([golden_set, extra])

# Shuffle the final 200 rows so the intents are mixed up
golden_set = golden_set.sample(frac=1, random_state=42).reset_index(drop=True)

# Create empty columns for human labeling
golden_set['human_intent'] = ""
golden_set['human_decision'] = ""  # Expected: AUTO_HANDLE or ESCALATE
golden_set['human_reason'] = ""    # Optional reason

# Keep only the columns needed for labeling
output_cols = ['customer_message', 'brand_response', 'human_intent', 'human_decision', 'human_reason']
output_path = 'data/golden_set_unlabeled.csv'
golden_set[output_cols].to_csv(output_path, index=False)

print(f"\nSuccess! Saved {len(golden_set)} stratified rows to '{output_path}'.")
