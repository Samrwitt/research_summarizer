import os
import re
import requests
import tempfile
import shutil
import subprocess
from urllib.parse import urlparse
from bs4 import BeautifulSoup

def ingest(arxiv=None, pdf=None, text=None):
    """
    Ingest content from one of the sources: arXiv ID, PDF file path, or Text file path.
    Returns:
    {
      "source": "arxiv" | "pdf" | "text",
      "paper_id": str | None,
      "title": str | None,
      "abstract": str | None,
      "text": str,
      "meta": dict
    }
    """
    if arxiv:
        return _ingest_arxiv(arxiv)
    elif pdf:
        return _ingest_pdf(pdf)
    elif text:
        return _ingest_text(text)
    else:
        raise ValueError("Must provide one of arxiv, pdf, or text")

def _ingest_arxiv(arxiv_input):
    """
    Ingest from arXiv.
    Steps:
    1. Parse ID.
    2. Get metadata (title, abstract) from /abs/.
    3. Try /html/ for text (preserve math).
    4. Fallback to /pdf/ and extract text.
    """
    paper_id = _parse_arxiv_id(arxiv_input)
    if not paper_id:
        raise ValueError(f"Invalid arXiv input: {arxiv_input}")
    
    print(f"Processing arXiv ID: {paper_id}")
    
    # Get metadata
    abs_url = f"https://arxiv.org/abs/{paper_id}"
    try:
        r_abs = requests.get(abs_url)
        r_abs.raise_for_status()
        soup_abs = BeautifulSoup(r_abs.text, 'html.parser')
        
        title_tag = soup_abs.select_one('h1.title')
        title = title_tag.text.replace('Title:', '').strip() if title_tag else "Unknown Title"
        
        abs_tag = soup_abs.select_one('blockquote.abstract')
        abstract = abs_tag.text.replace('Abstract:', '').strip() if abs_tag else ""
    except Exception as e:
        print(f"Warning: Failed to fetch metadata for {paper_id}: {e}")
        title = f"arXiv:{paper_id}"
        abstract = ""

    # Try HTML full text
    html_url = f"https://arxiv.org/html/{paper_id}"
    full_text = None
    source_type = "arxiv_html"
    
    try:
        print(f"Attempting HTML fetch: {html_url}")
        r_html = requests.get(html_url)
        if r_html.status_code == 200:
            full_text = _extract_arxiv_html_text(r_html.text)
        else:
            print(f"HTML version not available (status {r_html.status_code}).")
    except Exception as e:
        print(f"HTML fetch failed: {e}")

    # Fallback to PDF
    if not full_text:
        print("Falling back to PDF download and extraction...")
        source_type = "arxiv_pdf"
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        try:
            r_pdf = requests.get(pdf_url, stream=True)
            r_pdf.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                for chunk in r_pdf.iter_content(chunk_size=8192):
                    tmp_pdf.write(chunk)
                tmp_pdf_path = tmp_pdf.name
            
            try:
                # Reuse PDF ingestion logic, but capture text
                pdf_data = _ingest_pdf(tmp_pdf_path)
                full_text = pdf_data['text']
            finally:
                if os.path.exists(tmp_pdf_path):
                    os.remove(tmp_pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download/extract PDF for {paper_id}: {e}")

    return {
        "source": source_type,
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "text": full_text,
        "meta": {"url": abs_url}
    }

def _parse_arxiv_id(input_str):
    # Matches typical IDs like 1706.03762 or 1706.03762v1
    # Also handles URLs like https://arxiv.org/abs/1706.03762
    # Simple regex for the ID part
    match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', input_str)
    if match:
        return match.group(1)
    # Handle older IDs if necessary (e.g. math/0501231), but keeping simple for now
    if '/' in input_str and not input_str.startswith('http'):
         return input_str # Assume raw ID like math/...
    return None

def _extract_arxiv_html_text(html_content):
    """
    Extract text from arXiv HTML, preserving math as LaTeX.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # arXiv HTML often puts math in special spans or mathml.
    # We want to convert them to LaTeX tokens $$...$$ if possible.
    # This is heuristic-based as arXiv HTML format varies.
    
    # Look for MathJax script tags or similar structures if rendered
    # But often raw HTML from /html/ might use other structures.
    # Common pattern: <span class="ltx_Math" ...>...</span> or <math>
    
    # Strategy: Iterate text nodes. If we hit a math node, convert to text.
    # Simplification: Use soup.get_text() but try to handle specific math classes first.
    
    # 1. Replace MathML/MathJax elements with placeholders or TeX
    # Note: Implementing a full MathML->LaTeX converter is complex.
    # We will try to grab the 'alttext' or similar attributes if available, 
    # or just use a placeholder [MATH] if complex.
    
    # Modern arXiv HTML (LaTeXML generated) uses class 'ltx_Math'.
    for math_span in soup.find_all(class_='ltx_Math'):
        # Usually contains content. Try to find the TeX representation.
        # Sometimes it's in an attribute or a child script.
        # Fallback: [MATH]
        tex = None
        # Check for MathJax script inside
        script = math_span.find('script', type='math/tex')
        if script:
            tex = script.get_text()
        
        if not tex:
            # Check for generic 'alt' or 'data-tex' if purely static
            pass

        if tex:
            math_span.replace_with(f" $${tex}$$ ")
        else:
             # Just keep text content or placeholder
             # Often the text content of the span is the symbols themselves
             pass 

    # 2. Extract text from main content
    # arXiv HTML usually has a main container
    content = soup.select_one('.ltx_page_main') or soup.body
    
    if content:
        text = content.get_text(separator=' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)
        
    return text

def _ingest_pdf(pdf_path):
    print(f"Ingesting PDF: {pdf_path}")
    text = ""
    method = "unknown"
    
    # 1. Try pdfplumber
    try:
        import pdfplumber
        print("Trying pdfplumber...")
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    pages.append(extracted)
            text = "\n".join(pages)
        if text.strip():
            method = "pdfplumber"
    except ImportError:
        print("pdfplumber not installed.")
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    # 2. Try pypdf
    if not text.strip():
        try:
            import pypdf
            print("Trying pypdf...")
            reader = pypdf.PdfReader(pdf_path)
            pages = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pages.append(extracted)
            text = "\n".join(pages)
            if text.strip():
                method = "pypdf"
        except ImportError:
            print("pypdf not installed.")
        except Exception as e:
            print(f"pypdf failed: {e}")

    # 3. Try pdftotext (system tool)
    if not text.strip():
        print("Trying pdftotext (system)...")
        if shutil.which("pdftotext"):
            try:
                result = subprocess.run(
                    ["pdftotext", "-layout", pdf_path, "-"],
                    capture_output=True, text=True, check=True
                )
                text = result.stdout
                if text.strip():
                    method = "pdftotext"
            except Exception as e:
                print(f"pdftotext failed: {e}")
        else:
            print("pdftotext not found in PATH.")

    if not text.strip():
        raise RuntimeError("Failed to extract text from PDF using any available method.")

    return {
        "source": "pdf",
        "paper_id": None,
        "title": os.path.basename(pdf_path),
        "abstract": None, # Will be extracted in preprocess
        "text": text,
        "meta": {"path": pdf_path, "extraction_method": method}
    }

def _ingest_text(text_path):
    print(f"Ingesting text file: {text_path}")
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read text file: {e}")
        
    return {
        "source": "text",
        "paper_id": None,
        "title": os.path.basename(text_path),
        "abstract": None,
        "text": text,
        "meta": {"path": text_path}
    }
