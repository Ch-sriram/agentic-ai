# agent_tools.py
# Tool-Enable Agent with Robust Search:
# - Safe Calculator (Pure Python)
# - Web Search: ddgs -> Wikipedia REST -> offline facts -> LLM one-liner
# - FLAN-T5 fallback for general text

import os, re, ast, requests, operator as op
from dotenv import load_dotenv

# DuckDuckGoSearch Package
from ddgs import DDGS

# AI related packages
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

# Tool 1: Safe Calculator
SAFE_OPS = {
  ast.Add: op.add,
  ast.Sub: op.sub,
  ast.Mult: op.mul,
  ast.Div: op.truediv,
  ast.FloorDiv: op.floordiv,
  ast.Mod: op.mod,
  ast.Pow: op.pow,
  ast.USub: op.neg,
  ast.UAdd: op.pos
}

def _eval_ast(node):
  """
  Recursive code to evaluate the Abstract Syntax Tree (AST) formed after parsing the math expression
  """

  if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
    return node.value
  
  if isinstance(node, ast.BinOp):
    left = _eval_ast(node.left)
    right = _eval_ast(node.right)
    fn = SAFE_OPS.get(type(node.op))
    if not fn:
      raise ValueError("Unsupported operator")
    return fn(left, right)
  
  if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
    return SAFE_OPS[type(node.op)](_eval_ast(node.operand))
  
  if isinstance(node, ast.Expr):
    return _eval_ast(node.value)

  # Since our calculator support is basic, all other operation types
  # like complex numbers, quadratic equations, etc, are unsupported.
  raise ValueError("Unsupported expression")

def safe_calculate(expr: str) -> str:
  try:
    expr = expr.replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    return f"{_eval_ast(tree.body)}"
  except Exception:
    return "I couldn't compute that safely"


# Tool 2: Web Search (layered fallbacks)
def canon(q: str) -> str:
  q = q.lower().strip()
  q = re.sub(r"[^a-z0-9\s]", " ", q)  # drop puncuation like ?!',.
  q = re.sub(r"\s+", " ", q)          # collapse spaces
  return q

OFFLINE_FACTS = {
  "capital of japan": "Tokyo",
  "capital of france": "Paris",
  "capital of germany": "Berlin",
  "who founded hugging face": "Clément Delangue, Julien Chaumond, and Thomas Wolf.",
  "what is langchain": "LangChain is a framework for building LLM-powered apps using chains, tools, and agents.",
  "define agentic ai": "Agentic AI refers to AI systems that can plan, choose tools, and act autonomously toward goals.",
}


def _ddgs_search(query: str) -> str | None:
  try:
    with DDGS() as ddg:
      results = list(ddg.text(
        query=query,
        max_results=3,
        safesearch="moderate",
        region="us-en"
      ))
    
    if not results:
      return None
    
    top = results[0]
    title = (top.get("title") or "").strip()
    snippet = (top.get("body") or "").strip()
    return f"{title} - {snippet}".strip(" -")

  except Exception as e:
    print(e)
    return None

def _wiki_summary(query: str) -> str | None:
  try:
    # Very light heuristic: if asking for capitals, extract the place
    ql = canon(query)
    term = query
    if ql.startswith('capital of'):
      term = query.lower().replace('capital of', '').strip(' ?!.,')

    url = f"https://en.wikipedia.org/api/rest_v1/page/summary{term.title().replace(' ', '%20')}"
    r = requests.get(url, timeout=5)
    if r.status_code == 200:
      data = r.json()
      if "extract" in data and data["extract"]:
        return data["extract"].split(". ")[0]
    
    return None

  except Exception as e:
    print(e)
    return None


def _offline_answer(query: str) -> str | None:
  key = canon(query)
  return OFFLINE_FACTS.get(key)

def web_search(query: str) -> str:
  # 1. DDGS
  hit = _ddgs_search(query)
  if hit:
    return hit
  
  # 2. Wikipedia summary
  hit = _wiki_summary(query)
  if hit:
    return hit
  
  # 3. Offline facts (normalized)
  hit = _offline_answer(query)
  if hit:
    return hit
  
  # 4. Last resort LLM one-liner definition/explanation
  prompt = f"Answer in one short, factual sentence suitable for a beginner: {query}"
  out = pipe(prompt, do_sample=False, max_new_tokens=40)[0]["generated_text"].strip()
  return out.splitlines()[0].strip() or "No results found."


# Simple Router
def is_math(q: str) -> bool:
  return any(ch.isdigit() for ch in q) and any(sym in q for sym in "+-*%()^")

def should_search(q: str) -> bool:
  ql = q.lower()
  return any(kw in ql for kw in [
    "who", "when", "where", "what is", "what's", "latest", "capital",
    "define", "search", "find", "founder", "founded", "meaning of"
  ])

def agent_reply(user_input: str) -> str:
  if is_math(user_input):
    return f"Calculator: {safe_calculate(user_input)}"
  
  if should_search(user_input):
    return f"Search: {web_search(user_input)}"
  
  # LLM fallback for general chit-chat
  prompt = f"Answer in one short sentence: {user_input}"
  out = pipe(prompt, do_sample=False, max_new_tokens=40)[0]["generated_text"].strip()
  return out.splitlines()[0].strip()


# Main Logic
if __name__ == "__main__":
  print("⚙️  Tool-Enabled Agent ready! Type a question, or type 'quit' / 'exit' to stop.\n")
  while True:
    try:
      q = input("> ").strip()
      if q.lower() in {"quit", "exit"}:
        print("Goodbye!")
        break
      if not q:
        continue

      print("\n🤖", agent_reply(q), "\n")

    except KeyboardInterrupt:
      print("\n(User Forced Exit) Goodbye!")
      break
