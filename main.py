import argparse
import os
import sys
from src.ingestion.ingest import ingest
from src.ingestion.preprocess import preprocess
from src.models.extractive import summarize_extractive
from src.models.abstractive import summarize_abstractive
from src.utils.postprocess import generate_bullet_points, create_markdown_report
from src.utils.export import export_markdown, export_docx

def main():
    parser = argparse.ArgumentParser(description="NLP Research Paper Summarizer")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--arxiv', help="arXiv ID or URL")
    group.add_argument('--pdf', help="Path to PDF file")
    group.add_argument('--text', help="Path to text file")
    
    parser.add_argument('--method', choices=['extractive', 'abstractive'], default='abstractive', help="Summarization method")
    parser.add_argument('--model', default='sshleifer/distilbart-cnn-12-6', help="HuggingFace model for abstractive")
    parser.add_argument('--outdir', default='outputs', help="Output directory")
    
    args = parser.parse_args()
    
    # 1. Ingest
    print("Step 1: Ingesting...")
    try:
        data = ingest(arxiv=args.arxiv, pdf=args.pdf, text=args.text)
    except Exception as e:
        print(f"Error during ingestion: {e}")
        sys.exit(1)
        
    # 2. Preprocess
    print("Step 2: Preprocessing...")
    data = preprocess(data)
    print(f"Stats: {data['stats']}")
    
    # 3. Summarize
    print(f"Step 3: Summarizing ({args.method})...")
    summary = ""
    used_method = args.method
    
    if args.method == 'abstractive':
        try:
            # We summarize the chunks
            summary = summarize_abstractive(data['chunks'], model_name=args.model)
        except Exception as e:
            print(f"Abstractive summarization failed: {e}")
            print("Fallback to extractive...")
            used_method = 'extractive (fallback)'
            summary, _ = summarize_extractive(data['focus_text'], num_sentences=10)
    else:
        summary, _ = summarize_extractive(data['focus_text'], num_sentences=10)
        
    if not summary:
        print("Error: Could not generate summary.")
        sys.exit(1)
        
    # 4. Postprocess
    print("Step 4: Postprocessing...")
    bullets = generate_bullet_points(summary, num_bullets=5)
    report_md = create_markdown_report(data, summary, bullets, used_method)
    
    # 5. Export
    print("Step 5: Exporting...")
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    filename = data.get('paper_id') or "paper"
    if filename == "paper" and data.get('title'):
        # sanitize title
        filename = "".join(c for c in data['title'] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')[:50]
        
    out_md = os.path.join(args.outdir, f"{filename}.md")
    out_docx = os.path.join(args.outdir, f"{filename}.docx")
    
    export_markdown(report_md, out_md)
    export_docx(data, summary, bullets, out_docx)
    
    print(f"Done! Outputs saved to:\n  {out_md}\n  {out_docx}")

if __name__ == "__main__":
    main()
