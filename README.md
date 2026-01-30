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

## Project Structure

```text
research_summarizer/
├── data/               # SQLite database & local storage
├── docs/               # Scientific reports & presentations
├── src/                # Core Package
│   ├── ingestion/     # PDF/arXiv parsing & cleaning
│   ├── models/        # Summarizers, RAG, & Analysis
│   └── utils/         # DB, Export, & Evaluation
├── tests/              # Verification scripts & test data
├── app.py              # Main Web UI (Streamlit)
└── main.py             # CLI Tool
```

## Usage

### Web UI (Recommended)

Experience the premium interface with Library management and Chat-with-Paper features:

```bash
streamlit run app.py
```

### CLI

Run the summarizer directly for batch processing:

```bash
python main.py --arxiv 1706.03762 --outdir outputs --method abstractive
```

Or with a local PDF:

```bash
python main.py --pdf /path/to/paper.pdf --outdir outputs
```
