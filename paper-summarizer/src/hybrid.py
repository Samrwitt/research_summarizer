
from .extractive import summarize_extractive
from .abstractive import summarize_abstractive
import math

def summarize_hybrid(text, model_name="allenai/led-base-16384", reduction_ratio=0.5):
    """
    Hybrid Summarization Strategy:
    1. Filter: Use Extractive Summarization to keep the top 50% (reduction_ratio) most important sentences.
    2. Refine: Feed this reduced context to the Abstractive model.
    
    Why?
    - Abstractive models (especially Longformer/LED) are slow O(N^2) or O(N*W).
    - Reducing input length N by 50% drastically speeds up inference.
    - Removes noise/fluff that might confuse the model.
    """
    if not text:
        return ""
        
    print(f"Hybrid Mode: condensing text (Ratio: {reduction_ratio})...")
    
    # Step 1: Extractive Filtering
    # Estimate sentence count roughly
    approx_sentences = text.count('.') 
    target_sentences = max(5, int(approx_sentences * reduction_ratio))
    
    # We ignore the summary_text output, we want the list of sentences to reconstruct context
    _, top_sentences = summarize_extractive(text, num_sentences=target_sentences)
    
    # Important: Top sentences must be re-ordered to original flow to make sense!
    # summarize_extractive already does this sorting at the end.
    
    condensed_text = " ".join(top_sentences)
    print(f"Condensed length: {len(text)} -> {len(condensed_text)} chars")
    
    # Step 2: Abstractive Summarization
    # We pass this 'condensed' text as a single chunk (or multiple if still massive)
    # The abstractive module handles chunking if needed.
    
    summary = summarize_abstractive([condensed_text], model_name=model_name)
    
    return summary
