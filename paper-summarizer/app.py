import streamlit as st
import os
import tempfile
from src.ingest import ingest
from src.preprocess import preprocess
from src.extractive import summarize_extractive
from src.abstractive import summarize_abstractive
from src.hybrid import summarize_hybrid
from src.analysis import extract_insights
from src.postprocess import generate_bullet_points, create_markdown_report
from src.evaluate import evaluate_summary
from src.export import export_markdown, export_docx

st.set_page_config(page_title="Research Summarizer Premium", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Premium Feel ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #34495e;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    .tag {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.9em;
        font-weight: 500;
        border: 1px solid #c8e6c9;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📄 Research Summarizer AI")
    st.markdown("#### *Transforming complexity into clarity.*")
with col2:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80) 

# --- Sidebar ---
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")
method = st.sidebar.selectbox("Summarization Model", ["Abstractive (Longformer/BART)", "Extractive (Key Sentences)"])

use_hybrid = False
model_name = None

if "Abstractive" in method:
    model_name_default = "allenai/led-base-16384" 
    model_name = st.sidebar.text_input("HuggingFace Model ID", model_name_default, help="Use 'allenai/led-base-16384' for long papers.")
    use_hybrid = st.sidebar.checkbox("🚀 Enable Hybrid Mode", value=True, help="Speeds up processing by 50% using extractive filtering first.")
else:
    model_name = None

st.sidebar.markdown("---")
st.sidebar.info("Tip: Hybrid Mode combines TF-IDF filtering with Longformer to process long papers faster without losing key context.")

# --- Main Interface ---
tabs = st.tabs(["📥 Ingest & Summarize", "🧠 Insights", "📊 Evaluation", "📤 Export"])

# State variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'summary_text' not in st.session_state:
    st.session_state.summary_text = ""
if 'bullets' not in st.session_state:
    st.session_state.bullets = []
if 'stats' not in st.session_state:
    st.session_state.stats = {}
if 'insights' not in st.session_state:
    st.session_state.insights = {}

with tabs[0]:
    st.subheader("1. Load Paper")
    input_type = st.radio("", ["arXiv ID/URL", "Upload PDF"], horizontal=True)
    
    input_args = None
    start_process = False

    if input_type == "arXiv ID/URL":
        col_in1, col_in2 = st.columns([3, 1])
        with col_in1:
            arxiv_input = st.text_input("arXiv ID (e.g., 1706.03762)", placeholder="1706.03762")
        with col_in2:
            st.write("") # Spacer
            st.write("")
            if st.button("🚀 Fetch & Summarize"):
                if arxiv_input:
                    input_args = {"arxiv": arxiv_input}
                    start_process = True
                else:
                    st.error("Please provide an ID.")

    else:
        uploaded_file = st.file_uploader("Drop your PDF here", type="pdf")
        if st.button("🚀 Upload & Summarize") and uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            input_args = {"pdf": tmp_path}
            start_process = True

    # Processing Logic
    if start_process:
        with st.status("Processing Paper...", expanded=True) as status:
            st.write("Reading document...")
            try:
                data = ingest(**input_args)
                data = preprocess(data)
                st.session_state.data = data
                st.session_state.stats = data['stats']
                st.write("Preprocessing complete.")
                
                # Insights
                st.write("Extracting Semantic Insights (KeyBERT)...")
                st.session_state.insights = extract_insights(data['clean_text'])
                
                st.write(f"Generating Summary ({method})... Hybrid={use_hybrid}")
                
                summary_text = ""
                used_method = method.split()[0].lower()
                
                if "Abstractive" in method:
                    try:
                        if use_hybrid:
                            summary_text = summarize_hybrid(data['focus_text'], model_name=model_name)
                            used_method = "hybrid (extractive+abstractive)"
                        else:
                            summary_text = summarize_abstractive(data['chunks'], model_name=model_name)
                    except Exception as e:
                        st.warning(f"Abstractive failed: {e}. Falling back to Extractive.")
                        used_method = "extractive (fallback)"
                        summary_text, _ = summarize_extractive(data['focus_text'])
                else:
                     summary_text, _ = summarize_extractive(data['focus_text'])
                
                st.session_state.summary_text = summary_text
                st.session_state.bullets = generate_bullet_points(summary_text)
                st.session_state.used_method = used_method
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                status.update(label="Failed", state="error")
                import traceback
                st.error(traceback.format_exc())

    # Display Results
    if st.session_state.summary_text:
        st.divider()
        st.subheader(f"📝 Summary ({st.session_state.get('used_method', 'Unknown')})")
        
        st.markdown(f"""
        <div style="background-color: #ffffff;
    color: #1f2937;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
    box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    line-height: 1.6;
    font-size: 1rem;">
            {st.session_state.summary_text}
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📌 Key Takeaways")
        for b in st.session_state.bullets:
            st.info(b)

with tabs[1]:
    st.header("🧠 Paper Insights")
    
    if st.session_state.insights:
        st.subheader("Semantic Keywords")
        keywords = st.session_state.insights.get('keywords', [])
        
        # HTML for tags
        tags_html = "".join([f'<span class="tag">{k}</span>' for k in keywords])
        st.markdown(f'<div class="tag-container">{tags_html}</div>', unsafe_allow_html=True)
        
        st.markdown("### Structure Analysis")
        if st.session_state.data and 'sections' in st.session_state.data:
            sections = st.session_state.data['sections'].keys()
            st.write("Identified Sections:")
            st.json(list(sections))
    else:
        st.info("Process a paper to view insights.")

with tabs[2]:
    st.header("Quality Metrics")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### Document Stats")
        if st.session_state.stats:
            s = st.session_state.stats
            st.metric("Raw Characters", s.get('raw_len', 0))
            st.metric("Clean Characters", s.get('clean_len', 0))
            st.metric("Chunks Processed", s.get('num_chunks', 0))
        else:
            st.info("No document processed yet.")

    with col_m2:
        st.markdown("### 🏆 ROUGE Evaluation")
        st.caption("Provide a reference summary (gold standard) to calculate accuracy scores.")
        reference = st.text_area("Reference Summary", height=150, placeholder="Paste a human-written summary here to compare...")
        
        if st.button("Calculate ROUGE Score"):
            if reference and st.session_state.summary_text:
                scores = evaluate_summary(reference, st.session_state.summary_text)
                
                # Display scores nicely
                r1, r2, rl = st.columns(3)
                with r1:
                    st.metric("ROUGE-1 (F1)", scores['rouge1']['fmeasure'])
                with r2:
                    st.metric("ROUGE-2 (F1)", scores['rouge2']['fmeasure'])
                with rl:
                    st.metric("ROUGE-L (F1)", scores['rougeL']['fmeasure'])
                    
                st.json(scores)
            else:
                st.warning("Needs both a generated summary and a reference summary.")

with tabs[3]:
    st.header("Export Report")
    if st.session_state.summary_text:
        col_ex1, col_ex2 = st.columns(2)
        
        report_md = create_markdown_report(st.session_state.data, st.session_state.summary_text, st.session_state.bullets, st.session_state.get('used_method', 'Unknown'))
        
        with col_ex1:
            st.download_button(
                label="📄 Download Markdown",
                data=report_md,
                file_name="research_summary.md",
                mime="text/markdown"
            )
        
        with col_ex2:
            st.info("DOCX Export available via script.")
    else:
        st.info("Generate a summary first to enable export.")

