import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

print("Loading Golden Set...")
try:
    golden_df = pd.read_csv('data/golden_set.csv')
except FileNotFoundError:
    print("Golden set not found yet. Waiting for labeler to finish!")
    exit(1)

# Ensure the golden set has human labels
golden_df = golden_df.dropna(subset=['human_intent', 'human_decision'])
if len(golden_df) == 0:
    print("Golden set is empty or not fully labeled yet.")
    exit(1)

# ==========================================
# TRIVIAL BASELINE
# ==========================================
print("\n--- TRIVIAL BASELINE ---")
# Predict majority class, always escalate, single canned reply
trivial_preds_intent = ["DELIVERY_STATUS"] * len(golden_df)
trivial_preds_decision = ["ESCALATE"] * len(golden_df)
canned_reply = "Thanks for reaching out, please DM your order number."

acc_t_intent = accuracy_score(golden_df['human_intent'], trivial_preds_intent)
acc_t_decision = accuracy_score(golden_df['human_decision'], trivial_preds_decision)

print(f"Trivial Intent Accuracy:   {acc_t_intent:.2%}")
print(f"Trivial Decision Accuracy: {acc_t_decision:.2%}")


# ==========================================
# SIMPLE BASELINE
# ==========================================
print("\n--- SIMPLE BASELINE ---")
def simple_intent_classifier(text):
    text = str(text).lower()
    if any(w in text for w in ['gracias', 'merci', 'danke', 'hola', 'bonjour', 'la', 'el', 'que']): return "NON_ENGLISH"
    if any(w in text for w in ['return', 'refund', 'replace', 'broken', 'damaged']): return "RETURN_REFUND"
    if any(w in text for w in ['app', 'kindle', 'prime', 'music', 'stick', 'video']): return "TECH_SUPPORT"
    if any(w in text for w in ['payment', 'charge', 'money', 'card', 'bank', 'charged']): return "ACCOUNT_PAYMENT_ISSUE"
    if any(w in text for w in ['thanks', 'dm', 'sent', 'done']): return "FOLLOW_UP"
    return "DELIVERY_STATUS"

def simple_decision_engine(text):
    text = str(text).lower()
    if any(w in text for w in ['angry', 'sue', 'lawyer', 'worst', 'pathetic', 'cancel', 'refund', 'terrible', 'nightmare']):
        return "ESCALATE"
    return "AUTO_HANDLE"

simple_preds_intent = [simple_intent_classifier(t) for t in golden_df['customer_message']]
simple_preds_decision = [simple_decision_engine(t) for t in golden_df['customer_message']]

acc_s_intent = accuracy_score(golden_df['human_intent'], simple_preds_intent)
f1_s_intent = f1_score(golden_df['human_intent'], simple_preds_intent, average='weighted', zero_division=0)
acc_s_decision = accuracy_score(golden_df['human_decision'], simple_preds_decision)

try:
    f1_s_decision = f1_score(golden_df['human_decision'], simple_preds_decision, pos_label='ESCALATE', average='binary', zero_division=0)
except ValueError:
    f1_s_decision = 0.0

print(f"Simple Intent Accuracy:    {acc_s_intent:.2%} (F1: {f1_s_intent:.2f})")
print(f"Simple Decision Accuracy:  {acc_s_decision:.2%} (F1: {f1_s_decision:.2f})")


# ==========================================
# SIMPLE BASELINE RAG (Nearest Neighbor)
# ==========================================
print("\n--- SIMPLE BASELINE RAG ---")
print("Loading Knowledge Base (10,000 historical pairs)...")
kb_df = pd.read_csv('data/amazon_pairs.csv').sample(n=10000, random_state=42).reset_index(drop=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedding KB & finding nearest neighbor replies for golden set...")
kb_embeddings = model.encode(kb_df['customer_message'].tolist())
query_embeddings = model.encode(golden_df['customer_message'].tolist())

similarities = cosine_similarity(query_embeddings, kb_embeddings)
top_1_idx = np.argmax(similarities, axis=1)

golden_df['simple_reply_pred'] = [kb_df.iloc[i]['brand_response'] for i in top_1_idx]

# Save baseline results
golden_df['trivial_intent'] = trivial_preds_intent
golden_df['trivial_decision'] = trivial_preds_decision
golden_df['trivial_reply'] = canned_reply

golden_df['simple_intent'] = simple_preds_intent
golden_df['simple_decision'] = simple_preds_decision

golden_df.to_csv('data/baseline_results.csv', index=False)
print("\nSaved baseline predictions to 'data/baseline_results.csv'")
