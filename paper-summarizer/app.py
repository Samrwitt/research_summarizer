import streamlit as st
import os
import tempfile
from src.ingest import ingest
from src.preprocess import preprocess
from src.extractive import summarize_extractive
from src.abstractive import summarize_abstractive
from src.postprocess import generate_bullet_points, create_markdown_report
from src.export import export_markdown, export_docx

st.set_page_config(page_title="Paper Summarizer", layout="wide")

st.title("NLP Research Paper Summarizer")
st.markdown("Summarize research papers from arXiv or PDF.")

# Sidebar
st.sidebar.header("Configuration")
method = st.sidebar.selectbox("Method", ["Abstractive", "Extractive"])
model_name = st.sidebar.text_input("Model (for Abstractive)", "sshleifer/distilbart-cnn-12-6")

# Input
st.subheader("Input Source")
input_type = st.radio("Choose input:", ["arXiv ID/URL", "Upload PDF"])

data = None
process = False

if input_type == "arXiv ID/URL":
    arxiv_input = st.text_input("Enter arXiv ID or URL (e.g., 1706.03762)")
    if st.button("Summarize arXiv"):
        if arxiv_input:
            input_args = {"arxiv": arxiv_input}
            process = True
        else:
            st.warning("Please enter an ID.")
else:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if st.button("Summarize PDF") and uploaded_file:
        # Save temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        input_args = {"pdf": tmp_path}
        process = True

if process:
    with st.spinner("Ingesting and Processing..."):
        try:
            data = ingest(**input_args)
            data = preprocess(data)
            st.success("Ingestion complete!")
            st.json(data['stats'])
        except Exception as e:
            st.error(f"Error: {e}")
            
    if data:
        with st.spinner("Summarizing..."):
            summary_text = ""
            used_method = method.lower()
            
            try:
                if method == "Abstractive":
                    try:
                        summary_text = summarize_abstractive(data['chunks'], model_name=model_name)
                    except Exception as e:
                        st.warning(f"Abstractive failed ({e}), falling back to Extractive.")
                        used_method = "extractive (fallback)"
                        summary_text, _ = summarize_extractive(data['focus_text'])
                else:
                    summary_text, _ = summarize_extractive(data['focus_text'])
            except Exception as e:
                st.error(f"Summarization failed: {e}")
                summary_text = None

            if summary_text:
                bullets = generate_bullet_points(summary_text)
                report_md = create_markdown_report(data, summary_text, bullets, used_method)
                
                st.subheader("Summary")
                st.markdown(f"**Method Used:** {used_method}")
                st.write(summary_text)
                
                st.subheader("Key Points")
                for b in bullets:
                    st.markdown(f"- {b}")
                    
                # Downloads
                st.download_button("Download Report (Markdown)", report_md, file_name="summary.md")
