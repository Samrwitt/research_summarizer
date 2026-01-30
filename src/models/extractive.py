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
    clean_sentences = [re.sub(r'[^a-zA-Z\s]', '', s.lower()) for s in sentences]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(clean_sentences)
    except ValueError:
        # e.g. empty vocabulary
        return " ".join(sentences[:num_sentences]), sentences[:num_sentences]
        
    # Score sentences: sum of TF-IDF scores
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    
    # Rank
    if len(sentences) <= num_sentences:
        top_indices = range(len(sentences))
    else:
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices) # Restore original order
        
    summary_sentences = [sentences[i] for i in top_indices]
    summary_text = " ".join(summary_sentences)
    
    return summary_text, summary_sentences
