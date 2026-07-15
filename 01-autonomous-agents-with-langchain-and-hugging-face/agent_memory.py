import os, re
from dotenv import load_dotenv

from transformers import pipeline
from huggingface_hub import login

# Login to Hugging Face
load_dotenv()
token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
if not token:
  raise SystemError("❌ No Hugging Face token found. Add HF_TOKEN=... to your `.env` file")
login(token=token)

# Select the model, and create a pipeline
MODEL_ID = "google/flan-t5-base"
pipe = pipeline(
  task="text2text-generation",
  model=MODEL_ID,
  tokenizer=MODEL_ID,
  device=-1                     # CPU for stability/consistency
)

# Memory Stores
history: list[str] = []       # chat turns: "You: ...", "Agent: ..."
facts: dict[str, str] = {}    # normalized preferences, e.g., {"sport", "basketball", "food": "ramen"}

# Create in-memory stores using dictionaries for history, and facts
# This will help the LLM in recalling facts
FAV_PAT = re.compile(r"my (?:favorite|favourite|best)\s+(\w+)\s+is\s+([A-Za-z ]+)", re.I)
LIKE_PAT = re.compile(r"\bi (?:like|love)\s+([A-Za-z ]+)", re.I)

# Helper methods
def normalize_key(word: str) -> str:
  # Tiny normalization: pluarl -> singular for a few common categories
  word = word.lower().strip()
  mapping = {"sports": "sport", "foods": "food", "songs": "song", "movies": "movie"}
  return mapping.get(word, word)

def store_fact(user_message: str) -> None:
  """
  Captures simple 'favorite X is Y' and 'I like/love Y' facts.
  """
  m = FAV_PAT.search(user_message)
  if m:
    key = normalize_key(m.group(1))
    val = m.group(2).strip(" .!?,")
    facts[key] = val
    return
  m2 = LIKE_PAT.search(user_message)
  if m2:
    val = m2.group(1).strip()
    facts["general_like"] = val

def answer_from_facts(user_message: str) -> str | None:
  """
  Answer from stored facts if the question references them.
  """
  msg = user_message.lower()

  # If message mentions a specific known category, answer that first
  for key, val in facts.items():
    if key == "general_like":
      continue
    if key in msg:
      return f"You said your favourite {key} is {val}"
  
  # If user asks about favourites in general and we have exactly one favourite
  if ("favourite" in msg or "favorite" in msg) and any(k for k in facts if k != "general_like"):
    # Prefer sport if present; otherwise first favourite we have
    if "sport" in facts:
      return f"You said your favourite sport is {facts['sport']}."
    for k, v in facts.items():
      if k != "general_like":
        return f"You said your favourite {k} is {v}."
      
  # If user asks what they "like/love", try generic like first, else fallback to a favourite
  if "like" in msg or "love" in msg:
    if "general_like" in msg:
      return f"You said you like {facts['general_like']}"
    
    # If message hints the category (e.g., 'What sports do I like?'), use favourite of that category
    for k, v in facts.items():
      if k != "general_like" and k in msg:
        return f"You said you like {v}."
    
    # Otherwise, if we have exactly one favourite, answer with it
    favourites = [(k, v) for k, v in facts.items() if k != "general_like"]
    if len(favourites) == 1:
      k, v = favourites[0]
      return f"You said your favourite {k} is {v}."

STOPWORDS = {
  "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "at", "by", "from", "is", "are",
  "this", "that", "those", "these", "about", "into", "as", "it", "its", "be", "being", "been", "was", "were"
}

# Sanitization Methods (Helpers)

def one_line(text: str) -> str:
  return text.strip().splitlines()[0].strip()

def hard_sanitize(text: str) -> str:
  """
  Strip any leaked meta/instruction lines.
  """
  lines = []
  for ln in text.splitlines():
    s = ln.strip()
    if not s:
      continue

    # Drop lines that look like meta/instructions
    if any(w in s.lower() for w in [
      "rule", "instruction", "do not", "never speak", "context:", "assistant:", "user:", "[", "]", ":", "You are"
    ]):
      continue

    lines.append(s)
  
  cleaned = " ".join(lines)

  # Secondary cleanup if anything slipped through
  cleaned = re.sub(r"(do not|never speak|instruction).*", "", cleaned, flags=re.I).strip()
  return one_line(cleaned) or "I don't know"


# Response Logic
def respond(user_message: str) -> str:
  # 1. Try symbolic memory first (deterministic, no LLM needed)
  recall = answer_from_facts(user_message)
  if recall:
    return recall
  
  # 2. Minimal context window (last 6 lines)
  context = "\n".join(history[-6:]) if history else "(no prior messages)"

  # 3. Neutral Q&A style prompt to minimize parroting
  prompt = f"""Answer the user in one short sentence using only the conversation context.

Conversation:
{context}

Question:
{user_message}

Short answer:"""
  
  raw = pipe(prompt, do_sample=False, max_new_tokens=40)[0]["generated_text"]
  return hard_sanitize(raw)


# Main Loop

if __name__ == "__main__":
  print("🧠 Memory Agent ready! Type anything [OR] type 'quit' / 'exit' to stop.\n")
  while True:
    try:
      user_input = input("You: ").strip()
      if user_input.lower() in {"quit", "exit"}:
        print("Goodbye!")
        break
      if not user_input:
        continue

      # Save simple facts BEFORE we answer (so recall works immediately)
      store_fact(user_input)
      answer = respond(user_input)

      # log both sides AFTER we answer (so context reflects the conversation)
      history.append(f"You: {user_input}")
      history.append(f"Agent: {answer}")

      print(f"\n🤖 Agent: {answer}\n")
    except KeyboardInterrupt:
      print("[Forced User Exit] Goodbye!")
      break
