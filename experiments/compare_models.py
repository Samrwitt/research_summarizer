"""
Experimental script to compare different summarization models.

Usage:
    python experiments/compare_models.py --arxiv 1706.03762 --models led-base longt5-base distilbart
"""

import argparse
import time
import json
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion.ingest import ingest
from src.ingestion.preprocess import preprocess
from src.models.abstractive import summarize_abstractive
from src.models.model_configs import MODEL_REGISTRY, get_model_config
from src.utils.evaluate import evaluate_summary

def run_experiment(paper_source, models_to_test):
    """
    Run summarization with multiple models and compare results.
    
    Args:
        paper_source: dict with 'arxiv', 'pdf', or 'text' key
        models_to_test: list of model keys from MODEL_REGISTRY
    
    Returns:
        dict with experiment results
    """
    print("\n" + "="*80)
    print("STARTING MODEL COMPARISON EXPERIMENT")
    print("="*80)
    
    # Step 1: Ingest and preprocess
    print("\n[1/3] Ingesting paper...")
    data = ingest(**paper_source)
    data = preprocess(data)
    
    print(f"✓ Paper: {data['title']}")
    print(f"✓ Chunks: {len(data['chunks'])}")
    print(f"✓ Focus text length: {len(data['focus_text'])} chars")
    
    reference_summary = data.get('abstract', '')
    if reference_summary:
        print(f"✓ Reference abstract available ({len(reference_summary)} chars)")
    else:
        print("⚠ No reference abstract (ROUGE scores unavailable)")
    
    # Step 2: Run each model
    print("\n[2/3] Testing models...")
    results = {
        "paper_id": data.get('paper_id'),
        "title": data.get('title'),
        "timestamp": datetime.now().isoformat(),
        "reference_abstract": reference_summary,
        "models": {}
    }
    
    for model_key in models_to_test:
        if model_key not in MODEL_REGISTRY:
            print(f"⚠ Skipping unknown model: {model_key}")
            continue
        
        config = get_model_config(model_key)
        print(f"\n{'─'*80}")
        print(f"🔹 Testing: {model_key}")
        print(f"   Model ID: {config['model_id']}")
        print(f"   Max context: {config['max_input_length']} tokens")
        
        try:
            start_time = time.time()
            
            # Generate summary
            summary = summarize_abstractive(
                data['chunks'], 
                model_name=config['model_id']
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"✓ Summary generated in {elapsed_time:.2f}s")
            print(f"✓ Summary length: {len(summary)} chars")
            
            # Evaluate if reference available
            rouge_scores = None
            if reference_summary:
                print("   Calculating ROUGE scores...")
                rouge_scores = evaluate_summary(reference_summary, summary)
                print(f"   ROUGE-1: {rouge_scores['rouge1']['fmeasure']:.4f}")
                print(f"   ROUGE-2: {rouge_scores['rouge2']['fmeasure']:.4f}")
                print(f"   ROUGE-L: {rouge_scores['rougeL']['fmeasure']:.4f}")
            
            # Store results
            results["models"][model_key] = {
                "config": config,
                "summary": summary,
                "time_seconds": elapsed_time,
                "rouge_scores": rouge_scores,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results["models"][model_key] = {
                "config": config,
                "summary": None,
                "time_seconds": None,
                "rouge_scores": None,
                "success": False,
                "error": str(e)
            }
    
    # Step 3: Summary comparison
    print("\n[3/3] Comparison Summary")
    print("="*80)
    
    successful_models = [k for k, v in results["models"].items() if v["success"]]
    
    if not successful_models:
        print("❌ All models failed!")
        return results
    
    print(f"\n✓ {len(successful_models)}/{len(models_to_test)} models succeeded\n")
    
    # Speed comparison
    print("⏱️  SPEED RANKING (fastest to slowest):")
    speed_ranking = sorted(
        [(k, v["time_seconds"]) for k, v in results["models"].items() if v["success"]],
        key=lambda x: x[1]
    )
    for i, (model, time_s) in enumerate(speed_ranking, 1):
        print(f"   {i}. {model}: {time_s:.2f}s")
    
    # Quality comparison (if ROUGE available)
    if reference_summary:
        print("\n🏆 QUALITY RANKING (by ROUGE-2 F1):")
        quality_ranking = sorted(
            [(k, v["rouge_scores"]["rouge2"]["fmeasure"]) 
             for k, v in results["models"].items() if v["success"]],
            key=lambda x: x[1],
            reverse=True
        )
        for i, (model, score) in enumerate(quality_ranking, 1):
            print(f"   {i}. {model}: {score:.4f}")
    
    print("\n" + "="*80)
    
    return results

def save_results(results, output_file):
    """Save experiment results to JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare different summarization models")
    
    # Input source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--arxiv', type=str, help='arXiv ID (e.g., 1706.03762)')
    source_group.add_argument('--pdf', type=str, help='Path to PDF file')
    source_group.add_argument('--text', type=str, help='Path to text file')
    
    # Models to test
    parser.add_argument(
        '--models', 
        nargs='+', 
        default=['led-base', 'longt5-base', 'distilbart'],
        choices=list(MODEL_REGISTRY.keys()),
        help='Models to compare (default: led-base longt5-base distilbart)'
    )
    
    # Output
    parser.add_argument(
        '--output', 
        type=str, 
        default='experiment_results.json',
        help='Output JSON file (default: experiment_results.json)'
    )
    
    args = parser.parse_args()
    
    # Prepare source
    if args.arxiv:
        paper_source = {'arxiv': args.arxiv}
    elif args.pdf:
        paper_source = {'pdf': args.pdf}
    else:
        paper_source = {'text': args.text}
    
    # Run experiment
    results = run_experiment(paper_source, args.models)
    
    # Save results
    save_results(results, args.output)
    
    print("\n✅ Experiment complete!")

if __name__ == "__main__":
    main()
