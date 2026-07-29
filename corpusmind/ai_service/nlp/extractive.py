import hashlib
import math
import nltk
import numpy as np
import re

# Ensure nltk data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def summarize_extractive(text, num_sentences=10):
    """
    Summarize text using BERT-style sentence embedding centrality.
    """
    sentences = nltk.sent_tokenize(text)
    
    if not sentences:
        return "", []

    embeddings = _encode_sentences(sentences)
    centroid = embeddings.mean(axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-9)
    sentence_scores = embeddings @ centroid
    
    # Rank
    if len(sentences) <= num_sentences:
        top_indices = range(len(sentences))
    else:
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices) # Restore original order
        
    summary_sentences = [sentences[i] for i in top_indices]
    summary_text = " ".join(summary_sentences)
    
    return summary_text, summary_sentences


def _encode_sentences(sentences):
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return np.asarray(model.encode(sentences, normalize_embeddings=True), dtype=float)
    except Exception:
        vectors = np.asarray([_deterministic_embedding(sentence) for sentence in sentences], dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)


def _deterministic_embedding(text, dimensions=384):
    vector = np.zeros(dimensions, dtype=float)
    terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}", text.lower())
    for term in terms:
        digest = hashlib.sha256(term.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1 if digest[4] % 2 == 0 else -1
        vector[index] += sign * (1 + math.log1p(len(term)))
    return vector
