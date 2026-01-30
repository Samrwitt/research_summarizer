import numpy as np
import nltk
from sentence_transformers import SentenceTransformer, util
import torch

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("Loading SentenceTransformer model...")
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model

def answer_question(question, text, top_k=5):
    """
    Super simple RAG: 
    1. Split text into sentences.
    2. Embed sentences and question.
    3. Find top_k relevant sentences.
    4. Return context (and eventually feed to an LLM).
    """
    if not text:
        return "No text available to answer from."
        
    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:
        return "No sentences found."
        
    model = get_embed_model()
    
    # 1. Embed sentences
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # 2. Embed question
    question_embedding = model.encode(question, convert_to_tensor=True)
    
    # 3. Compute cosine similarity
    hits = util.semantic_search(question_embedding, sentence_embeddings, top_k=top_k)
    hits = hits[0] # Get hits for the first (and only) question
    
    # 4. Filter and build context
    context_sentences = []
    for hit in hits:
        context_sentences.append(sentences[hit['corpus_id']])
    
    context = "\n".join(context_sentences)
    
    # In a full app, we would feed 'question' and 'context' to an LLM.
    # For now, we return the retrieved context as the "Relevant Information".
    # We can also add a simple heuristic or a QA pipeline if desired.
    
    return context

def ask_llm(question, context, model_name=None):
    """
    Placeholder for LLM call (e.g., Ollama or a HF model).
    """
    # If we want to use the summarizer as a generator (not great but works in a pinch)
    # or if we want to integrate Ollama:
    try:
        import requests
        # Try local Ollama if running
        response = requests.post("http://localhost:11434/api/generate", 
                               json={
                                   "model": "llama3.2", # default suggestion
                                   "prompt": f"Context: {context}\n\nQuestion: {question}\n\nAnswer the question based only on the context provided. If not in context, say you don't know.",
                                   "stream": False
                               }, timeout=5)
        if response.status_code == 200:
            return response.json().get('response', 'Error: No response field')
    except Exception:
        # Fallback to HF QA pipeline if Ollama is offline
        # We load a small, fast QA model (DistilBERT)
        from transformers import pipeline
        
        # Load lazily to save startup time
        if not hasattr(ask_llm, '_qa_pipeline'):
            try:
                print("Loading fallback QA pipeline...")
                ask_llm._qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            except Exception as e:
                return f"Could not load QA model: {e}\n\nContext found:\n{context}"
                
        try:
            result = ask_llm._qa_pipeline(question=question, context=context)
            return result.get('answer', "I couldn't find the answer in the text.")
        except Exception as e:
            return f"QA Generation failed: {e}\n\nContext:\n{context}"
    
    return context
