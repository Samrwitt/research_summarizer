
from keybert import KeyBERT

# Load model once if possible, or lazy load
# For now, we load inside function to avoid startup cost if unused
# Ideally, cache this.

_kw_model = None

def get_model():
    global _kw_model
    if _kw_model is None:
        print("Loading KeyBERT model...")
        _kw_model = KeyBERT()
    return _kw_model

from .qa import ask_llm

def extract_insights(data):
    """
    Extract semantic keywords and generates a high-level analysis.
    Input: data dict (needs 'focus_text')
    """
    text = data.get('focus_text', '')
    if not text:
        return {"keywords": [], "analysis": "No text to analyze."}
        
    # 1. KeyBERT (Keywords)
    try:
        kw_model = get_model()
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
        tags = [k[0] for k in keywords]
        scores = [k[1] for k in keywords]
    except Exception as e:
        print(f"KeyBERT failed: {e}")
        tags = []
        scores = []
    
    # 2. LLM / QA Analysis (The "Insight")
    # We ask a generative question to getting a summarized insight.
    # Note: If Ollama is offline, ask_llm falls back to DistilBERT which extracts spans.
    # We phrase the question such that a span answer (like a sentence from the abstract) is still useful.
    
    target_question = "What is the main contribution and key findings of this paper?"
    
    # We limit context to first 12k chars to be safe for smaller models/time
    context_window = text[:12000]
    
    try:
        analysis = ask_llm(target_question, context_window)
    except Exception as e:
        analysis = f"Could not generate insights: {e}"
    
    return {
        "keywords": tags,
        "scores": scores,
        "analysis": analysis
    }
