
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

def extract_insights(text, top_n=10):
    """
    Extract semantic keywords and potentially other insights.
    """
    if not text:
        return {"keywords": []}
        
    kw_model = get_model()
    
    # KeyBERT extraction
    # keyphrase_ngram_range=(1, 2) allows bigrams like "Deep Learning"
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    
    # Format: [('keyword', score), ...]
    # We just return the words for UI
    tags = [k[0] for k in keywords]
    
    return {
        "keywords": tags
    }
