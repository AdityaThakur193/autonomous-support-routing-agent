import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading data...")
df = pd.read_csv("data/amazon_pairs.csv")
sample_df = df.sample(n=2000, random_state=42).reset_index(drop=True)
texts = sample_df['customer_message'].tolist()

print("Embedding texts...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

print("Clustering (k=6)...")
k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
sample_df['cluster'] = kmeans.fit_predict(embeddings)

print("Extracting TF-IDF terms...")
cluster_texts = sample_df.groupby('cluster')['customer_message'].apply(lambda x: " ".join(x))
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(cluster_texts)
terms = np.array(tfidf.get_feature_names_out())

for i in range(k):
    cluster_data = sample_df[sample_df['cluster'] == i]
    print(f"\n{'='*60}")
    print(f"CLUSTER {i} (Size: {len(cluster_data)})")
    print(f"{'='*60}")
    
    top_indices = tfidf_matrix[i].toarray()[0].argsort()[-15:][::-1]
    print(f"TOP TERMS: {', '.join(terms[top_indices])}\n")
    
    print("EXAMPLES:")
    examples = cluster_data['customer_message'].sample(min(8, len(cluster_data)), random_state=42)
    for ex in examples:
        print(f"- {ex.strip()}")
