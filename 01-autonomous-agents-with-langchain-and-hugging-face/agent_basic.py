import os
from dotenv import load_dotenv

# Imports related to huggingface and transformers
from huggingface_hub import login
from transformers import pipeline

# 1. Load `.env` file (contains your HF_TOKEN)
load_dotenv()

# 2. Log in to Hugging Face with your token
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
  raise SystemError("❌ No Hugging Face token found. Add HF_TOKEN=... in your `.env` file.")
login(token=HF_TOKEN)

# 3. Choose a small, free text-generation pipeline
MODEL_ID = "google/flan-t5-base"

# 4. Create a local text2text-generation pipeline
pipe = pipeline(
  task="text2text-generation",
  model=MODEL_ID,
  tokenizer=MODEL_ID,
  device=-1                   # Make use of CPU, instead of GPU for consistent results
)

# 5. (Test) Send one short prompt and print 
print("✅ Agent ready! Type a question, or 'quit' or 'exit' to stop.")

# 6. Create an infinite loop, until `exit` or `quit` is typed in to stop the agent from taking input.
while True:
  try:
    user_input = input("> ").strip()
    if user_input.lower() in {"quit", "exit"}:
      print("Goodbye!")
      break
    if not user_input:
      continue

    res = pipe(user_input, max_new_tokens=100)
    # print(res) # debug line
    result = res[0]["generated_text"]
    print(f"🤖 Agent: {result}")

  except KeyboardInterrupt:
    print("\n (Stopped By User) Goodbye!")
    break
