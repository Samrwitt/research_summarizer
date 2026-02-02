"""
Simple test to verify abstractive summarization works correctly.
"""

from src.models.abstractive import summarize_abstractive

# Test with a simple chunk
test_text = """
The Transformer is a neural network architecture that relies entirely on attention mechanisms
to draw global dependencies between input and output. It dispenses with recurrence and 
convolutions entirely. The architecture is simple and allows for significantly more 
parallelization than previous sequence models.
"""

print("Testing abstractive summarization...")
print(f"Input length: {len(test_text)} chars")
print("\nGenerating summary...")

try:
    summary = summarize_abstractive([test_text], model_name="sshleifer/distilbart-cnn-12-6")
    print("\n✅ SUCCESS!")
    print(f"Summary: {summary}")
    print(f"Summary length: {len(summary)} chars")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
