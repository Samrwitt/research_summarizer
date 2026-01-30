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
    # Also handle "process- ing" which sometimes happens in PDF extraction
    text = re.sub(r'(\w+)-\s*\n?\s*(\w+)', r'\1\2', text)
    
    # Multiple whitespace to single space (collapse newlines too as we want a stream)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip Ref/Bib
    # Simple heuristic to cut off references if clearly marked
    # Look for "References" or "Bibliography" on a new line (or after new line in original)
    
    # We look for the last occurrence of a References header in the last 20% of the text or last 10k chars.
    # This prevents cutting "References" mentioned in the intro.
    
    search_area_start = max(0, len(text) - 15000)
    search_text = text[search_area_start:]
    
    # Matches: newline + optional numbering + References/Bibliography + newline/colon
    # Examples: "\nReferences\n", "\n7. Bibliography\n", "\nLITERATURE CITED"
    # Note: \s includes newlines.
    match = re.search(r'(?:^|\n)\s*(?:[0-9]+\.?\s*)?(?:References|Bibliography|LITERATURE CITED|Reference List)(?:\s*[:\.]?)\s*\n', search_text, re.IGNORECASE)
    
    if match:
        # Calculate absolute cut position
        cut_pos = search_area_start + match.start()
        # Keep text up to that point
        text = text[:cut_pos].strip()
        print(f"Stripped references at char {cut_pos}")
        
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
    ordered_keys = ['abstract', 'conclusion', 'results', 'methods']
    # Introduction is often too generic. We handle it separately.
    
    parts = []
    
    # 1. Abstract (Highest priority)
    if abstract:
        parts.append(f"{abstract}")
        
    # 2. Conclusion / Discussion (High priority for takeaways)
    if 'conclusion' in sections:
        parts.append(f"{sections['conclusion']}")
        
    # 3. Results (Evidence)
    if 'results' in sections:
        parts.append(f"{sections['results']}")
        
    # 4. Methods (Context, but kept concise)
    if 'methods' in sections:
        # Limit methods to first 2000 chars to avoid getting bogged down in equations
        parts.append(f"{sections['methods'][:2000]}...")
        
    # 5. Introduction (Context, but filtered)
    if 'introduction' in sections:
        # Intro often has "Background". We limit it.
        parts.append(f"{sections['introduction'][:1000]}...")
            
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
