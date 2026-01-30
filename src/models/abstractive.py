from transformers import pipeline
import torch
import os

def summarize_abstractive(chunks, model_name="sshleifer/distilbart-cnn-12-6"):
    """
    Summarize chunks using HuggingFace pipeline.
    Combines chunk summaries into a final summary.
    Raises RuntimeError if model cannot be loaded (e.g. offline and not cached).
    """
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading summarization model: {model_name} on device {device}")
    
    try:
        # local_files_only=True if we want to strictly force offline, 
        # but better to let it try download if online, and fail if offline.
        # We can detect offline by catching OSError or ConnectionError from transformers.
        summarizer = pipeline("summarization", model=model_name, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load abstractive model '{model_name}'. Possible offline mode without cache. Error: {e}")

    # Check if model is Longformer/LED based for long context support
    is_long_context = "led" in model_name.lower() or "longformer" in model_name.lower()
    
    print(f"Summarizing {len(chunks)} chunks... Long Context Mode: {is_long_context}")
    
    # If we are using LED, we might not want to chunk if the total text fits.
    # But current ingest pipeline chunks to 3000 chars anyway.
    # For LED, we should ideally concatenating chunks back if they are small, 
    # but for safety/simplicity we will process chunks. 
    # However, if we have a robust PC, we could try to increasing chunk size in preprocess, 
    # but here we deal with what we have.
    
    chunk_summaries = []
    
    for i, chunk in enumerate(chunks):
        try:
            input_len = len(chunk.split())
            
            # Dynamic params
            if is_long_context:
                # LED can handle longer, so we can be generous
                max_len = 256
                min_len = 64
            else:
                max_len = min(150, max(30, int(input_len * 0.5)))
                min_len = min(30, max(10, int(input_len * 0.1)))
            
            # Call pipeline
            res = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
            text = res[0]['summary_text']
            chunk_summaries.append(text)
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
