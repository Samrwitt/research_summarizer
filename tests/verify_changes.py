
import sys
import os

print("Checking imports...")
try:
    import streamlit
    print("Streamlit OK")
    import rouge_score
    print("ROUGE Score OK")
    import sentencepiece
    print("SentencePiece OK")
    import nltk
    print("NLTK OK")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

print("Checking source modules...")
try:
    from src.utils.evaluate import evaluate_summary
    from src.utils.postprocess import generate_bullet_points
    from src.models.abstractive import summarize_abstractive
    print("Source modules OK")
except Exception as e:
    print(f"Source Module Error: {e}")
    sys.exit(1)

print("Testing ROUGE...")
ref = "The cat sat on the mat."
hyp = "The cat is on the mat."
scores = evaluate_summary(ref, hyp)
print(f"ROUGE output: {scores}")
if 'rouge1' in scores:
    print("ROUGE Test Passed")
else:
    print("ROUGE Test Failed")
    sys.exit(1)
    
print("Verification Complete!")
