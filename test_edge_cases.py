import sys
import importlib.util
import os

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = input("Please paste your Gemini API Key: ").strip()

spec = importlib.util.spec_from_file_location("full_agent", "src/05_full_agent.py")
full_agent = importlib.util.module_from_spec(spec)
sys.modules["full_agent"] = full_agent
spec.loader.exec_module(full_agent)

print("Building Agent...")
agent = full_agent.build_agent()

test_cases = [
    ("Over-Escalation Test (Sarcastic Joy)", "My package was delayed by a day, which is the WORST thing ever haha, but I just got it and I love it. Thanks guys!"),
    ("Under-Escalation Test (Polite Payment Issue)", "Good morning my friend! I hope you are having a wonderful day. Could you kindly check why my credit card was charged 3 times for the same Kindle? Thank you!"),
    ("Under-Escalation Test (Disguised Emergency)", "Hey just a quick fyi, the delivery van backed into my garage and collapsed the roof, anyway can I get a replacement for the squished package?"),
    ("Over-Escalation Test (Foreign Language Thank You)", "¡Muchísimas gracias! El paquete llegó perfecto y me encanta.")
]

for name, msg in test_cases:
    print(f"\n=================================")
    print(f"🧪 TEST: {name}")
    print(f"=================================")
    print(f"Customer: {msg}")
    try:
        decision = agent(msg, api_key)
        print(f"🧠 Intent:  {decision.intent}")
        print(f"🚦 Routing: {decision.routing_decision} (Reason: {decision.escalation_reason})")
        print(f"💬 Reply:   {decision.draft_reply}")
    except Exception as e:
        print(f"❌ Error: {e}")
