# 📄 Research Summarizer AI

A premium, local-first NLP application designed to ingest, analyze, and summarize research papers. Built with a focus on aesthetics and deep semantic understanding, it helps researchers and students transform complex documents into clear, actionable insights.

## ✨ Key Features

-   **Multi-Source Ingestion**: Seamlessly process papers from **arXiv IDs**, uploaded **PDFs**, or raw text files.
-   **Hybrid Summarization Intelligence**:
    -   **Abstractive**: Uses state-of-the-art Transformers (e.g., Longformer, BART) to generate human-like summaries.
    -   **Extractive**: Identifies and ranks key sentences using TF-IDF and graph-based algorithms.
    -   **Hybrid Mode**: Combines both approaches for optimal speed and coherence on long documents.
-   **Chat with Paper (RAG)**: An interactive Q&A system that uses Retrieval-Augmented Generation to answer specific questions based *strictly* on the paper's content.
-   **Semantic Insights**: Automatically extracts key themes, research gaps, methodology highlights, and contributions using KeyBERT and LLM analysis.
-   **Quality Metrics**: Built-in ROUGE evaluation tools to compare generated summaries against author abstracts or ground truths.
-   **Library Management**: Automatically saves your processed papers to a local SQLite database for easy retrieval and management.
-   **Export Ready**: Download comprehensive reports in **Markdown** or **DOCX** formats.

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   `pip` package manager

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/samrwitt/research_summarizer.git
    cd research_summarizer
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For PDF processing, ensure you have necessary system libraries if using `pdf2image` (though `pypdf` is the default).*

## 🖥️ Usage

### Web Interface (Recommended)

Launch the premium Streamlit dashboard:

```bash
streamlit run app.py
```

**Using the App:**
1.  **Load a Paper**: Enter an arXiv ID or upload a PDF in the "Ingest" tab.
2.  **Configure Models**: Click the **Settings (⚙️)** icon in the top header to toggle between Abstractive/Extractive models or enable Hybrid mode.
3.  **Analyze**: View the summary, chat with the document, or check the extracted insights in their respective tabs.
4.  **Save/Export**: Your work is automatically saved to the "Library" tab.

### CLI Tool

For batch processing or quick terminal-based summaries:

```bash
# Summarize from arXiv
python main.py --arxiv 1706.03762 --outdir outputs --method abstractive

# Summarize a local PDF
python main.py --pdf documents/paper.pdf --outdir outputs
```

## 📂 Project Structure

```text
research_summarizer/
├── data/               # SQLite database (metadata.db) & local storage
├── docs/               # Documentation & assets
├── src/                # Core Application Logic
│   ├── ingestion/      # Parsers for arXiv, PDF, and text cleaning
│   ├── models/         # Summarization engines, RAG pipelines, & Analysis
│   └── utils/          # Database ORM, Exporters, & Evaluation metrics
├── tests/              # Unit tests
├── app.py              # Main Streamlit Dashboard entry point
├── main.py             # CLI entry point
└── requirements.txt    # Python dependencies
```

## 🛠️ Technology Stack

-   **Frontend**: Streamlit
-   **NLP Models**: HuggingFace Transformers (Longformer, LED, BART), KeyBERT
-   **Vector Search**: FAISS / Sentence-Transformers (for RAG)
-   **Database**: SQLite
-   **Processing**: PyPDF, NLTK, BeautifulSoup
