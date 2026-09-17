import pandas as pd
import json
import os
import google.generativeai as genai
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class AgentDecision(BaseModel):
    intent: str = Field(description="One of: DELIVERY_STATUS, RETURN_REFUND, TECH_SUPPORT, ACCOUNT_PAYMENT_ISSUE, NON_ENGLISH, FOLLOW_UP")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    routing_decision: str = Field(description="Must be 'AUTO_HANDLE' or 'ESCALATE'")
    escalation_reason: str = Field(description="Reason for escalation, or 'N/A' if auto-handled")
    draft_reply: str = Field(description="The grounded draft reply to the customer")

def build_agent():
    # 1. Load Knowledge Base for RAG
    print("Loading Agent Knowledge Base (Sampling 2,000 historical cases for speed)...")
    kb_df = pd.read_csv('data/amazon_pairs.csv').sample(n=2000, random_state=42).reset_index(drop=True)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Embedding precedents for RAG (this takes about 5-10 seconds)...")
    kb_embeddings = embedder.encode(kb_df['customer_message'].tolist(), show_progress_bar=True)
    
    # 2. Define the Agent Pipeline
    def process_ticket(customer_message: str, api_key: str):
        # A. Retrieve Top 3 Historical Resolutions
        query_embed = embedder.encode([customer_message])
        sims = cosine_similarity(query_embed, kb_embeddings)[0]
        top_3_idx = sims.argsort()[-3:][::-1]
        
        context = ""
        for i, idx in enumerate(top_3_idx):
            context += f"Historical Case {i+1}:\n"
            context += f"Customer: {kb_df.iloc[idx]['customer_message']}\n"
            context += f"Amazon Reply: {kb_df.iloc[idx]['brand_response']}\n\n"
            
        # B. LLM Decision & Generation
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are an elite customer support routing agent for @AmazonHelp.
        
        Incoming Customer Message:
        "{customer_message}"
        
        Historical Precedents (Use these to ground your reply):
        {context}
        
        Task:
        1. Classify the intent.
        2. Decide to AUTO_HANDLE or ESCALATE (Escalate if angry, complex, legal, or non-English).
        3. Draft a reply grounded ONLY in how Amazon historically responded in the precedents above.
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=AgentDecision,
                temperature=0.0
            )
        )
        
        return AgentDecision(**json.loads(response.text))
        
    return process_ticket

if __name__ == "__main__":
    print("========================================")
    print("🐝 Hiver AI Agent Pipeline (Gemini) 🐝")
    print("========================================")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Please paste your Gemini API Key: ").strip()
        
    print("\nInitializing Agent and Knowledge Base...")
    agent = build_agent()
    print("Agent is ready!\n")
    
    print("Type 'exit' to quit.\n")
    while True:
        customer_msg = input("Customer: ")
        if customer_msg.lower() == 'exit':
            break
            
        print("\nThinking...")
        try:
            decision = agent(customer_msg, api_key)
            print("----------------------------------------")
            print(f"🧠 Intent:      {decision.intent} (Confidence: {decision.confidence})")
            print(f"🚦 Routing:     {decision.routing_decision} (Reason: {decision.escalation_reason})")
            print(f"💬 Draft Reply: {decision.draft_reply}")
            print("----------------------------------------\n")
        except Exception as e:
            print(f"Error: {e}\n")
