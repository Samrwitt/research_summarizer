import nltk

def generate_bullet_points(summary_text, num_bullets=5):
    """
    Generate bullet points from the summary text using extractive ranking.
    Uses NLTK for better sentence splitting.
    """
    if not summary_text:
        return []

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
    # We re-use extractive logic regarding the summary as the source text
    # This ranks sentences within the summary itself (or if summary is short, just returns them)
    # detailed splitting
    from .extractive import summarize_extractive
    
    # We pass the summary as the text to "summarize" again (extract top sentences)
    _, bullets = summarize_extractive(summary_text, num_sentences=num_bullets)
    return bullets

def create_markdown_report(data, summary_text, bullets, method):
    """
    Create a Markdown formatted report.
    """
    title = data.get('title', 'Unknown Title')
    url = data.get('meta', {}).get('url', 'N/A')
    paper_id = data.get('paper_id', 'N/A')
    
    md = f"# Summary: {title}\n\n"
    md += f"**Source:** {data['source'].upper()}\n"
    md += f"**ID:** {paper_id}\n"
    md += f"**Link:** {url}\n"
    md += f"**Method:** {method}\n\n"
    
    md += "## TL;DR\n"
    md += f"{summary_text}\n\n"
    
    md += "## Key Points\n"
    for b in bullets:
        md += f"- {b}\n"
        md += "\n"
        
    md += "## Metadata\n"
    md += "### Sections Found\n"
    md += ", ".join(data.get('sections', {}).keys()) + "\n\n"
    
    md += "### Stats\n"
    stats = data.get('stats', {})
    for k, v in stats.items():
        md += f"- **{k}:** {v}\n"
        
    return md
