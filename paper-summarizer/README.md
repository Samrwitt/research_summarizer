# Research Paper Summarizer

A local-first NLP application to ingest, preprocess, and summarize research papers from arXiv, PDF, or text files.

## Features

- **Multi-source Ingestion**: 
    - arXiv (URL or ID)
    - PDF (Local upload/path)
    - Text files
- **Smart Parsing**: Preserves math formulas from arXiv HTML where possible.
- **Hybrid Summarization**: 
    - **Extractive**: TF-IDF based sentence ranking.
    - **Abstractive**: HuggingFace Transformers (BART/T5) with fallback to extractive if model fails or is offline.
- **Export**: Generates Markdown reports and DOCX summaries.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install `poppler-utils` for PDF processing if falling back to system tools.*

## Usage

### CLI

Run the summarizer from the command line:

```bash
python summarize.py --arxiv 1706.03762 --outdir outputs --method abstractive
```

Or with a local PDF:

```bash
python summarize.py --pdf /path/to/paper.pdf --outdir outputs
```

### Web UI

Run the Streamlit app:

```bash
streamlit run app.py
```
