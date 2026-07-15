import os
from dotenv import load_dotenv

from transformers import pipeline
from huggingface_hub import login

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

# 5. Tiny prompt template, 1 line, setup - punchline
TEMPLATE = (
  "Write one clean, original one-line joke about {topic}. "
  "Format: Setup - Punchline. Keep it under 18 words."
)

# 6. Definition for make_joke method
def make_joke(topic: str = "computers") -> str:
  prompt = TEMPLATE.format(topic=topic)
  result = pipe(
    prompt,
    do_sample=True,           # small dose of variety
    top_p=0.92,
    top_k=50,
    max_new_tokens=40
  )[0]["generated_text"]
  return result.splitlines()[0].strip()

# Run the main logic
if __name__ == "__main__":
  print("🤖 Joke-Teller ready! Type a topic, or 'quit' / 'exit' to stop.\n")
  while True:
    try:
      topic = input("Topic: ").strip()
      if topic.lower() in {"quit", "exit"}:
        print("Goodbye!")
        break
      if not topic:
        topic = "computers"
      print(f"\n😂 {make_joke(topic)}\n")
    except KeyboardInterrupt:
      print("(Forceful User Interrupt) Goodbye!")
      break
