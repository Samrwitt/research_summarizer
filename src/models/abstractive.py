from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

def summarize_abstractive(chunks, model_name="sshleifer/distilbart-cnn-12-6"):
    """
    Summarize chunks using HuggingFace model directly.
    Combines chunk summaries into a final summary.
    Raises RuntimeError if model cannot be loaded (e.g. offline and not cached).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading summarization model: {model_name} on device {device}")
    
    try:
        # Load model and tokenizer directly (more compatible than pipeline)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.to(device)
        model.eval()  # Set to evaluation mode
    except Exception as e:
        raise RuntimeError(f"Failed to load abstractive model '{model_name}'. Possible offline mode without cache. Error: {e}")

    # Check if model is Longformer/LED based for long context support
    is_long_context = "led" in model_name.lower() or "longformer" in model_name.lower() or "long-t5" in model_name.lower()
    
    print(f"Summarizing {len(chunks)} chunks... Long Context Mode: {is_long_context}")
    
    chunk_summaries = []
    
    for i, chunk in enumerate(chunks):
        try:
            input_len = len(chunk.split())
            
            # Dynamic params
            if is_long_context:
                # LED/Long-T5 can handle longer, so we can be generous
                max_len = 256
                min_len = 64
                max_input = 16384 if "led" in model_name.lower() else 16384  # Both support 16K
            else:
                max_len = min(150, max(30, int(input_len * 0.5)))
                min_len = min(30, max(10, int(input_len * 0.1)))
                max_input = 1024
            
            # Tokenize input
            inputs = tokenizer(chunk, max_length=max_input, truncation=True, return_tensors="pt")
            inputs = inputs.to(device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=max_len,
                    min_length=min_len,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode
            summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summaries.append(summary_text)
            
        except Exception as e:
            print(f"Error summarizing chunk {i}: {e}")
            continue

    if not chunk_summaries:
        return ""

    # If result is still list of summaries, join them.
    # If we had many chunks, the summary might be disjointed.
    # A second pass could be done here if needed, but simple join is stable.
    combined_summary = " ".join(chunk_summaries)
    
    return combined_summary
