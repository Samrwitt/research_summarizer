from rouge_score import rouge_scorer

def evaluate_summary(reference, hypothesis):
    """
    Calculate ROUGE scores between a reference summary and a generated hypothesis.
    Returns a dict with ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and fmeasure.
    """
    if not reference or not hypothesis:
        return {}
        
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    
    # Format for easier consumption
    results = {}
    for key, score in scores.items():
        results[key] = {
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "fmeasure": round(score.fmeasure, 4)
        }
        
    return results

