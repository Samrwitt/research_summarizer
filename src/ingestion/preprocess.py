import re

def preprocess(data):
    """
    Preprocess the ingested data.
    Input: dict from ingest()
    Output:
    {
      ...input_data,
      "clean_text": str,
      "sections": {name: text},
      "focus_text": str,
      "chunks": [str],
      "stats": dict
    }
    """
    raw_text = data.get('text', '') or ''
    
    # 1. Clean Text
    clean_text = _clean_text(raw_text)
    
    # 2. Extract Sections
    sections = _extract_sections(clean_text)
    
    # If abstract was not in metadata (e.g. from PDF/Text), try to get from sections
    if not data.get('abstract') and 'abstract' in sections:
        data['abstract'] = sections['abstract']

    # 3. Build Focus Text
    # Priority: Abstract -> Introduction -> Method -> Results -> Conclusion
    # If standard sections not found, use full text
    focus_text = _build_focus_text(sections, data.get('abstract'), clean_text)
    
    # 4. Chunking
    chunks = _chunk_text(focus_text, chunk_size=3000, overlap=200) # Approx tokens (chars / 4)
    
    return {
        **data,
        "clean_text": clean_text,
        "sections": sections,
        "focus_text": focus_text,
        "chunks": chunks,
        "stats": {
            "raw_len": len(raw_text),
            "clean_len": len(clean_text),
            "num_chunks": len(chunks)
        }
    }

def _clean_text(text):
    # Normalize unicode
    # Hyphenated line breaks: "process-\ning" -> "processing"
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Multiple whitespace to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip Ref/Bib
    # Simple heuristic to cut off references if clearly marked
    # Look for "References" or "Bibliography" on a new line (or after new line in original)
    # But since we flattened newlines, we check for " References " with capitalization patterns
    # This is risky, so be conservative.
    # A safe bet is looking for a standalone Header-like "References" at end of doc.
    # For now, we'll strip common tails if found.
    
    match = re.search(r'\s(References|Bibliography|LITERATURE CITED)\s', text[-10000:], re.IGNORECASE)
    if match:
        # Cut if it looks like a header (surrounded by space or uppercase)
        pass # To implement safely needs more context. Skipping aggressive cut for now to avoid losing content.
        
    return text.strip()

def _extract_sections(text):
    """
    Attempt to find Introduction, Methods, Results, Discussion/Conclusion.
    Regex for headers.
    """
    sections = {}
    
    # Heuristic regexes for headers. 
    # Assumes text is somewhat clean but headers might still be identifiable by numbering or keywords.
    # Since we flattened text, headers are hard to find purely by structure.
    # We might need to rely on the sequence.
    
    # Common headers
    headers = {
        'abstract': r'(?:^|\s)(?:ABS\s?TRACT)(?:\s|$)',
        'introduction': r'(?:^|\s)(?:1\.?|I\.?)?\s*INTRODUCTION(?:\s|$)',
        'methods': r'(?:^|\s)(?:2\.?|II\.?)?\s*(?:METHODS?|APPROACH|EXPERIMENTAL)(?:\s|$)',
        'results': r'(?:^|\s)(?:3\.?|III\.?)?\s*(?:RESULTS?)(?:\s|$)',
        'conclusion': r'(?:^|\s)(?:4\.?|IV\.?|5\.?|V\.?)?\s*(?:CONCLUSIONS?|DISCUSSION)(?:\s|$)',
        'references': r'(?:^|\s)(?:REFERENCES|BIBLIOGRAPHY|LITERATURE CITED)(?:\s|$)',
    }
    
    # We will search for indices of these headers
    indices = []
    for key, pattern in headers.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            indices.append((key, match.start()))
            
    # Sort by index
    indices.sort(key=lambda x: x[1])
    
    # Slice text
    for i, (name, start_idx) in enumerate(indices):
        end_idx = indices[i+1][1] if i + 1 < len(indices) else len(text)
        # Skip the header length itself roughly
        # We find exact match to get end of header
        header_match = re.search(headers[name], text[start_idx:start_idx+100], re.IGNORECASE)
        header_end = start_idx + header_match.end() if header_match else start_idx
        
        content = text[header_end:end_idx].strip()
        if content:
            sections[name] = content
            
    return sections

def _build_focus_text(sections, abstract, full_text):
    """
    Construct text optimized for summarization.
    """
    ordered_keys = ['introduction', 'methods', 'results', 'conclusion']
    parts = []
    
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")
        
    for k in ordered_keys:
        if k in sections:
            parts.append(f"{k.upper()}: {sections[k]}")
            
    if not parts:
        # Fallback to full text if parsing failed
        return full_text
        
    return "\n\n".join(parts)

def _chunk_text(text, chunk_size=3000, overlap=200):
    """
    Chunk text by approximating tokens.
    Safe implementation that prevents infinite loops.
    """
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to break at a space
        if end < text_len:
            # We must ensure that picking a space doesn't make us regress or loop.
            # next_start will be (end - overlap).
            # We require next_start > start, so end > start + overlap.
            
            safe_search_start = start + overlap
            if safe_search_start < end:
                # Look for last space in the safe zone
                last_space = text.rfind(' ', safe_search_start, end)
                if last_space != -1:
                    end = last_space
                
        # Append chunk
        sub = text[start:end]
        chunks.append(sub)
        
        # Calculate next start
        next_start = end - overlap
        
        # Safety guarantee: always advance by at least 1 character to avoid infinite loop
        if next_start <= start:
            # This happens if overlap >= chunk size (bad config) or other edge cases.
            # Force advance.
            next_start = start + max(1, chunk_size - overlap)
            
        start = next_start
        
    return chunks
