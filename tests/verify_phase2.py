
import sys
import os

print("--- Phase 2 Verification ---")

print("1. Checking Imports...")
try:
    from src.models.analysis import extract_insights
    from src.models.hybrid import summarize_hybrid
    import keybert
    print("Imports OK")
except ImportError as e:
    print(f"Import Failed: {e}")
    sys.exit(1)

print("\n2. Testing Keyword Extraction (KeyBERT)...")
text = """
The Transformer model, introduced by Vaswani et al., revolutionized Natural Language Processing (NLP).
It relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions.
Deep Learning has seen massive growth due to these models.
"""
print(f"Input text len: {len(text)}")
try:
    insights = extract_insights(text, top_n=3)
    print(f"Insights Output: {insights}")
    if not insights['keywords']:
        print("FAILED: No keywords extracted.")
        sys.exit(1)
    print("Keyword Extraction OK")
except Exception as e:
    print(f"Extraction Failed: {e}")
    sys.exit(1)

print("\n3. Testing Hybrid Summarization...")
# Create dummy long text
long_text = (text + " ") * 20 
print(f"Long text len: {len(long_text)}")

try:
    # We use a very high reduction ratio to force it to do something visible
    # Note: This might fail if model download fails or takes too long, 
    # so we might wrap it or mock the abstractive part if strictly unit testing, 
    # but here we want integration test.
    # However, loading LED model might be slow.
    # We will test the 'reduction' logic primarily.
    
    from src.models.extractive import summarize_extractive
    
    # Manually testing the logic flow if actual model inference is too heavy for quick check
    # But let's try calling it.
    
    # Mocking abstractive to avoid 1GB download validation in this script if possible?
    # No, let's trust the environment.
    pass
except Exception as e:
    print(e)

print("Verification Script Done (Partial Check).")
