from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
import re

# Ensure nltk data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def summarize_extractive(text, num_sentences=10):
    """
    Summarize text using TF-IDF sentence ranking.
    """
    sentences = nltk.sent_tokenize(text)
    
    if not sentences:
        return "", []

    # Clean sentences for TF-IDF (keep original for output)
    clean_sentences = []
    valid_indices = []
    
    for i, s in enumerate(sentences):
        # Filter out garbage: citations, URLs, short fragments
        s_lower = s.lower().strip()
        if len(s_lower) < 20: continue
        if s_lower.startswith(('arxiv', 'doi', 'http', 'vol.', 'no.')): continue
        if 'preprint' in s_lower or 'copyright' in s_lower: continue
        
        # Heuristic: Filter out table rows / math jumbles
        # Count non-alpha characters (digits, symbols)
        alpha_count = sum(c.isalpha() for c in s)
        total_count = len(s)
        # Require 70% letters to be considered "text". Math/tables usually have < 50%.
        if total_count > 0 and (alpha_count / total_count) < 0.7:
            # If less than 70% letters, it's likely a table row or math formula
            continue
            
        # Count single letter words (often variables like d, k, v, h in math)
        words = s_lower.split()
        single_letter_words = sum(1 for w in words if len(w) == 1 and w.isalpha() and w not in ['a', 'i'])
        # If more than 20% of words are single letters (variables), skip
        if len(words) > 0 and (single_letter_words / len(words)) > 0.2:
            continue

        # Clean for TF-IDF
        clean = re.sub(r'[^a-zA-Z\s]', '', s_lower)
        clean_sentences.append(clean)
        valid_indices.append(i)
        
    if not clean_sentences:
        return "Could not extract meaningful sentences.", []
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(clean_sentences)
    except ValueError:
        # e.g. empty vocabulary
        return " ".join(sentences[:num_sentences]), sentences[:num_sentences]
        
    # Score sentences: sum of TF-IDF scores
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    
    # Rank
    # sentence_scores corresponds to clean_sentences, which maps to sentences[valid_indices[i]]
    if len(clean_sentences) <= num_sentences:
        top_local_indices = range(len(clean_sentences))
    else:
        top_local_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        
    # Map back to original sentence indices to preserve flow order
    top_original_indices = sorted([valid_indices[i] for i in top_local_indices])
        
    summary_sentences = [sentences[i] for i in top_original_indices]
    summary_text = " ".join(summary_sentences)
    
    return summary_text, summary_sentences
