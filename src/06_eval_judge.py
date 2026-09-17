import pandas as pd
from pydantic import BaseModel, Field
import google.generativeai as genai
import json
import os
import time

class JudgeRubric(BaseModel):
    groundedness_score: int = Field(description="1-5: Does the reply strictly use historical precedent? (1=Hallucinated, 5=Perfectly grounded)")
    tone_score: int = Field(description="1-5: Is the tone appropriate for @AmazonHelp? (1=Rude, 5=Polite and empathetic)")
    routing_correct: bool = Field(description="Did the agent correctly decide to auto-handle vs escalate compared to the human label?")

def evaluate_agent_results(api_key: str):
    df = pd.read_csv('data/baseline_results.csv')
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        'gemini-3.1-pro-preview', 
        system_instruction="You are a rigorous QA judge for customer support."
    )
    
    print("Running LLM-as-a-Judge Evaluation (Powered by Gemini 2.5 Pro)...")
    sample = df.head(10)
    
    for idx, row in sample.iterrows():
        prompt = f'''
        Customer: {row['customer_message']}
        
        Human Label Routing: {row['human_decision']}
        Agent Routing: {row['simple_decision']}
        Agent Draft Reply: {row['simple_reply_pred']}
        
        Evaluate the Agent's performance based on the rubric.
        '''
        
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=JudgeRubric,
                    temperature=0.0
                )
            )
            result = JudgeRubric(**json.loads(response.text))
            print(f"Row {idx} | Groundedness: {result.groundedness_score}/5 | Tone: {result.tone_score}/5 | Routing Correct: {result.routing_correct}")
        except Exception as e:
            print(f"Row {idx} | Rate limited or error: {e}")
        time.sleep(12)  # Respect the 5 RPM limit!

if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        evaluate_agent_results(api_key)
    else:
        print("Please set GEMINI_API_KEY environment variable.")
