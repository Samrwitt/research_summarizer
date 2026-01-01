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

    chunk_summaries = []
    print(f"Summarizing {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        try:
            # Max input length for distilbart is 1024. 
            # Our chunks are approx 3000 chars ~ 750 tokens maybe? should be safe.
            # But we must truncate if too long to avoid crash.
            # Pipeline handles truncation usually if truncation=True, but let's be explicit via args if needed.
            # For summarization pipeline, it automatically truncates.
            
            # Generate summary for chunk
            # Calculate max_length dynamically based on input length to avoid "max_length > input_length" issues
            input_len = len(chunk.split())
            max_len = min(150, max(30, int(input_len * 0.5)))
            min_len = min(30, max(10, int(input_len * 0.1)))
            
            res = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
            text = res[0]['summary_text']
            chunk_summaries.append(text)
        except Exception as e:
            print(f"Error summarizing chunk {i}: {e}")
            # Continue with others
            continue

    if not chunk_summaries:
        return ""

    # If we have multiple chunk summaries, we might want to summarize them again if they are too long.
    # For now, just join them.
    combined_summary = " ".join(chunk_summaries)
    
    return combined_summary
