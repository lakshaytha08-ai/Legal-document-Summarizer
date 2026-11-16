import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import sys
import json
import html
from importlib import import_module

current_dir = os.path.dirname(os.path.abspath(__file__))
RAG_UPLOAD_DIR = os.path.join(current_dir, "uploads")

os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)

# Set page config at the very beginning
st.set_page_config(
    page_title="Legal AI Hub - Document Summarization & Q&A", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Resolve import collision with app.py by temporarily removing current directory from sys.path
sys_path_backup = list(sys.path)
# Add RAG backend path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rag_backend_dir = os.path.join(workspace_dir, "legal-rag", "backend")

RAG_MODULES = {}
RAG_AVAILABLE = False
RAG_ERROR = None
if os.path.isdir(rag_backend_dir):
    sys.path = [p for p in sys.path if p not in (current_dir, "", ".")]
    sys.path.insert(0, rag_backend_dir)
    try:
        embedding_module = import_module("app.embeddings.embedding_model")
        vector_store_module = import_module("app.retrieval.vector_store")
        loaders_module = import_module("app.ingestion.loaders")
        text_cleaner_module = import_module("app.ingestion.text_cleaner")
        legal_chunker_module = import_module("app.chunking.legal_chunker")
        RAG_MODULES = {
            "EmbeddingModel": embedding_module.EmbeddingModel,
            "LocalVectorStore": vector_store_module.LocalVectorStore,
            "loaders": loaders_module,
            "text_cleaner": text_cleaner_module,
            "legal_chunker": legal_chunker_module,
        }
        RAG_AVAILABLE = True
    except (ImportError, AttributeError) as e:
        RAG_ERROR = str(e)
    finally:
        sys.path = sys_path_backup
else:
    RAG_ERROR = f"Optional RAG backend was not found at {rag_backend_dir}"

# Local application imports
from document_processor import extract_text_from_pdf, preprocess_text, calculate_basic_stats, extract_potential_clauses
from nlp_models import load_summarizer, load_classifier, chunk_document, summarize_chunks, summarize_document_hybrid, classify_clauses, calculate_classification_stats
from database import init_db, save_document, save_clauses, get_documents_df, get_clauses_df, delete_document_by_id
import evaluation

# Initialize relational database
init_db()

# Initialize RAG storage directories
if RAG_AVAILABLE:
    RAG_DATA_DIR = os.path.join(rag_backend_dir, "data")
    RAG_STORE_DIR = os.path.join(RAG_DATA_DIR, "vector_store")
    RAG_UPLOAD_DIR = os.path.join(RAG_DATA_DIR, "documents")
    os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)
    os.makedirs(RAG_STORE_DIR, exist_ok=True)
    
    @st.cache_resource
    def get_vector_db():
        return RAG_MODULES["LocalVectorStore"](store_dir=RAG_STORE_DIR)
        
    vector_db = get_vector_db()

# Inject style.css
css_file = os.path.join(current_dir, "style.css")
if os.path.exists(css_file):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ----------------- SIDEBAR SETTINGS -----------------
st.sidebar.markdown('<div style="text-align: center;"><span style="font-size: 3rem;">⚖️</span></div>', unsafe_allow_html=True)
st.sidebar.title("Configuration Portal")
st.sidebar.write("Customize your legal extraction pipelines and API parameters.")

# 1. NLP Execution Mode
use_mock = st.sidebar.checkbox(
    "Use Mock NLP Models (Fast Demo)", 
    value=True, 
    help="Runs heuristic classifiers and extractors instantly on CPU without downloading large neural networks."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Deep Learning Engine")
st.sidebar.markdown(f"**Hardware Device:** `{'CUDA GPU' if torch.cuda.is_available() else 'CPU (No GPU)'}`")

# 2. RAG Configurations
st.sidebar.markdown("---")
st.sidebar.subheader("RAG Model Settings")
llm_provider = st.sidebar.selectbox(
    "LLM Provider", 
    options=["mock", "gemini", "openai", "ollama"], 
    index=0
)
embedding_provider = st.sidebar.selectbox(
    "Embedding Provider", 
    options=["mock", "gemini", "local"], 
    index=0
)

# 3. API Keys
st.sidebar.markdown("---")
st.sidebar.subheader("API Keys (If Online)")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Sync settings to environment variables for RAG backend use
os.environ["LLM_PROVIDER"] = llm_provider
os.environ["EMBEDDING_PROVIDER"] = embedding_provider
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key
else:
    os.environ.pop("GEMINI_API_KEY", None)
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
else:
    os.environ.pop("OPENAI_API_KEY", None)

# Model caching to prevent reloading
@st.cache_resource
def get_neural_models():
    # If a local Hermes/Ollama endpoint is configured, skip the heavy LED model download.
    if os.getenv("HERMES_API_URL") or os.getenv("OLLAMA_BASE_URL") or os.getenv("HERMES_MODEL") or os.getenv("OLLAMA_MODEL"):
        return None, None, load_classifier()
    tokenizer, model = load_summarizer()
    classifier = load_classifier()
    return tokenizer, model, classifier

# ----------------- MAIN TITLE -----------------
st.markdown('<h1 class="title-header">⚖️ Legal Summarization & Clause Classification System</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9CA3AF;'>Long-context transformer summarization, zero-shot clause classification, and hybrid vector RAG analysis.</p>", unsafe_allow_html=True)

# 3-Tab Navigation Layout
tab1, tab2, tab3 = st.tabs([
    "📥 Document Analyzer",
    "📊 CUAD Benchmarking & Evaluation",
    "🗄️ Database & Logs"
])

# ----------------- TAB 1: DOCUMENT ANALYZER -----------------
with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Document Upload")
        uploaded_file = st.file_uploader("Drop a legal contract PDF here", type=["pdf"])
        
        with st.expander("Ingestion Tuning Parameters"):
            chunk_size = st.slider("Window Size (words)", min_value=500, max_value=3000, value=1500, step=100)
            overlap = st.slider("Window Overlap (words)", min_value=50, max_value=500, value=150, step=50)
            
        if uploaded_file is not None:
            if st.button("🚀 Analyze Legal Document", use_container_width=True):
                # Save uploaded file to RAG upload dir
                document_name = os.path.basename(uploaded_file.name)
                temp_pdf_path = os.path.join(RAG_UPLOAD_DIR, document_name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                # 1. Relational Extraction & Analysis
                with st.spinner("Extracting contract text..."):
                    raw_text = extract_text_from_pdf(temp_pdf_path)
                    clean_text = preprocess_text(raw_text)

                # Do not report a successful analysis when the PDF yielded no text.
                if not clean_text:
                    st.error(
                        "No selectable text was found in this PDF. "
                        "If it is a scanned/image-only PDF, OCR is required. "
                        "Try a text-based PDF or provide an OCR-processed PDF."
                    )
                    st.stop()

                with st.spinner("Dividing long document into sliding window chunks..."):
                    chunks = chunk_document(clean_text, max_tokens=chunk_size, overlap=overlap)
                    stats = calculate_basic_stats(clean_text, chunks)
                    
                with st.spinner("Extracting contract structure and clause signals..."):
                    potential_clauses = extract_potential_clauses(clean_text)

                if not use_mock:
                    with st.spinner("Loading deep learning networks (first download takes a moment)..."):
                        tokenizer, summarizer_model, classifier = get_neural_models()
                    with st.spinner("Running hybrid 3:7 legal summary (grounded extraction + LLM synthesis)..."):
                        summary = summarize_document_hybrid(clean_text, chunks, potential_clauses, tokenizer=tokenizer, model=summarizer_model)
                    with st.spinner("Extracting and classifying clauses via BART MNLI..."):
                        classifications = classify_clauses(potential_clauses, classifier, use_mock=False)
                else:
                    with st.spinner("Running Offline Mock summarizer..."):
                        summary = summarize_chunks(chunks, None, None, use_mock=True)
                    with st.spinner("Classifying clauses via keyword heuristic..."):
                        classifications = classify_clauses(potential_clauses, None, use_mock=True)
                        
                # Compute statistical aggregates
                mean_conf, med_conf, std_conf, mean_unc, med_unc, std_unc = calculate_classification_stats(classifications)
                obj_stats = {
                    "mean_conf": float(mean_conf),
                    "median_conf": float(med_conf),
                    "std_conf": float(std_conf),
                    "mean_uncertainty": float(mean_unc),
                    "median_uncertainty": float(med_unc),
                    "std_uncertainty": float(std_unc)
                }
                
                # Save relational information
                with st.spinner("Registering results in SQLite logs..."):
                    doc_id = save_document(document_name, stats, obj_stats, summary)
                    if classifications:
                        save_clauses(doc_id, classifications)
                        
                # 2. RAG Ingestion & Vector indexing
                if RAG_AVAILABLE:
                    with st.spinner("Ingesting pages for RAG database..."):
                        raw_pages = RAG_MODULES["loaders"].extract_document_text(temp_pdf_path)
                        cleaned_pages = RAG_MODULES["text_cleaner"].preprocess_document(raw_pages)
                        rag_chunks = RAG_MODULES["legal_chunker"].chunk_document(document_name, cleaned_pages)
                        
                    with st.spinner("Generating vector embeddings and saving to Local Store..."):
                        emb_model = RAG_MODULES["EmbeddingModel"](provider=embedding_provider)
                        chunk_texts = [c["text"] for c in rag_chunks]
                        embeddings = emb_model.encode(chunk_texts)
                        vector_db.add_chunks(rag_chunks, embeddings)
                        
                st.success("🎉 Ingestion & Analysis completed successfully!")
                
                # Save execution info in session state to show results
                st.session_state["last_doc_stats"] = stats
                st.session_state["last_doc_obj_stats"] = obj_stats
                st.session_state["last_doc_summary"] = summary
                st.session_state["last_doc_classifications"] = classifications
                st.session_state["last_doc_name"] = document_name

    with col_right:
        st.subheader("Extraction Preview")
        if "last_doc_name" in st.session_state:
            st.info(f"Loaded: **{st.session_state['last_doc_name']}**")
            
            # Display Premium Cards
            stats = st.session_state["last_doc_stats"]
            obj_stats = st.session_state["last_doc_obj_stats"]
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-label">Word Count</div>
                    <div class="metric-val">{stats['word_count']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sentences</div>
                    <div class="metric-val">{stats['sentence_count']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Chunks</div>
                    <div class="metric-val">{stats['chunk_count']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Confidence</div>
                    <div class="metric-val">{obj_stats['mean_conf']:.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No document analyzed in this session yet. Upload a PDF and click 'Analyze' to begin.")

    # Results Section
    if "last_doc_name" in st.session_state:
        st.markdown("---")
        st.subheader("📄 Generated Document Summary")
        st.info(st.session_state["last_doc_summary"])
        
        # Download summary button
        st.download_button(
            "💾 Download Summary",
            data=st.session_state["last_doc_summary"],
            file_name=f"{st.session_state['last_doc_name']}_summary.txt",
            mime="text/plain"
        )
        
        st.markdown("---")
        col_c_left, col_c_right = st.columns([1, 1])
        
        with col_c_left:
            st.subheader("🔍 Classified Legal Clauses")
            classifications = st.session_state["last_doc_classifications"]
            
            if classifications:
                for idx, c in enumerate(classifications[:10]):  # Show top 10
                    # Determine Badge Class
                    badge_class = f"badge-{c['category'].lower().replace(' ', '-')}"
                    if c['category'] == "Intellectual Property":
                        badge_class = "badge-ip"
                    elif c['category'] == "Dispute Resolution":
                        badge_class = "badge-dispute"
                        
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.25); padding: 1.25rem; border-radius: 12px; border-left: 5px solid rgba(99, 102, 241, 0.6); margin-bottom: 1rem; border-top: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span class="badge {html.escape(badge_class, quote=True)}">{html.escape(str(c['category']))}</span>
                            <span style="font-size: 0.82rem; color: #9CA3AF;">Confidence: <b>{c['confidence']:.1%}</b> | Uncertainty: <b>{c['uncertainty']:.1%}</b></span>
                        </div>
                        <div style="color: #E5E7EB; font-style: italic; line-height: 1.5; font-size: 0.95rem;">"{html.escape(str(c['clause']))}"</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with st.expander("View Full Raw Classification Table"):
                    st.dataframe(pd.DataFrame(classifications))
            else:
                st.warning("No active legal clauses detected by heuristics.")
                
        with col_c_right:
            st.subheader("📊 Statistical Analysis & Visualizations")
            
            # Sub-table of Detailed Statistics
            stats = st.session_state["last_doc_stats"]
            obj_stats = st.session_state["last_doc_obj_stats"]
            
            detail_data = {
                "Metric Characteristic": [
                    "Mean Chunk Length (words)", "Median Chunk Length (words)", "Std Dev Chunk Length (words)",
                    "Mean Prediction Confidence", "Median Prediction Confidence", "Std Dev Prediction Confidence",
                    "Mean Prediction Uncertainty", "Median Prediction Uncertainty", "Std Dev Prediction Uncertainty"
                ],
                "Value": [
                    f"{stats['mean_chunk_len']:.2f}", f"{stats['median_chunk_len']:.2f}", f"{stats['std_chunk_len']:.2f}",
                    f"{obj_stats['mean_conf']:.3f}", f"{obj_stats['median_conf']:.3f}", f"{obj_stats['std_conf']:.3f}",
                    f"{obj_stats['mean_uncertainty']:.3f}", f"{obj_stats['median_uncertainty']:.3f}", f"{obj_stats['std_uncertainty']:.3f}"
                ]
            }
            st.table(pd.DataFrame(detail_data))
            
            # Matplotlib Visualizations
            if classifications:
                df_cls = pd.DataFrame(classifications)
                
                # Plot 1: Histogram of Confidence
                fig, ax = plt.subplots(figsize=(6, 3))
                plt.style.use('dark_background')
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                ax.hist(df_cls['confidence'], bins=8, color='#6366F1', edgecolor='#312E81', alpha=0.85)
                ax.set_title("Distribution of Classification Confidence", color='#FFFFFF', fontsize=10, fontweight='bold')
                ax.set_xlabel("Confidence Probability", color='#9CA3AF', fontsize=8)
                ax.set_ylabel("Frequency", color='#9CA3AF', fontsize=8)
                ax.tick_params(colors='#9CA3AF', labelsize=8)
                ax.grid(axis='y', linestyle='--', alpha=0.2)
                st.pyplot(fig)
                
                # Plot 2: Donut of Categories
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                fig2.patch.set_facecolor('none')
                ax2.set_facecolor('none')
                
                cat_counts = df_cls['category'].value_counts()
                colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#3B82F6', '#EC4899']
                
                wedges, texts, autotexts = ax2.pie(
                    cat_counts, 
                    labels=cat_counts.index, 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=colors[:len(cat_counts)],
                    wedgeprops=dict(width=0.4, edgecolor='#111827'),
                    textprops=dict(color='#9CA3AF', fontsize=8)
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(7)
                    
                ax2.set_title("Clause Category Breakdown", color='#FFFFFF', fontsize=10, fontweight='bold')
                st.pyplot(fig2)

# ----------------- TAB 2: CUAD BENCHMARKING -----------------
with tab2:
    st.subheader("📈 CUAD Dataset Evaluation & Testing")
    st.write("Evaluate summarization and classification metrics using curated Contract Understanding Atticus Dataset (CUAD) agreements.")
    
    # Load local JSON contracts
    try:
        with open(os.path.join(current_dir, "cuad_samples.json"), "r", encoding="utf-8") as f:
            cuad_contracts = json.load(f)
        names = [c["doc_name"] for c in cuad_contracts]
    except Exception as e:
        names = []
        st.error(f"Failed to load CUAD samples JSON: {str(e)}")
        
    if names:
        selected_cuad = st.selectbox("Select Target CUAD Contract", options=names)
        
        # Get selected contract details
        contract = next(c for c in cuad_contracts if c["doc_name"] == selected_cuad)
        
        st.markdown("### Contract Raw Text Preview")
        st.text_area("Original Text Snippet", contract["text"][:1000] + "\n\n[Truncated...]", height=200, disabled=True)
        
        if st.button("📊 Run Evaluation Suite", use_container_width=True):
            with st.spinner("Processing evaluation dataset..."):
                # Run NLP Summarization on CUAD text
                cuad_chunks = chunk_document(contract["text"], max_tokens=1500, overlap=150)
                stats = calculate_basic_stats(contract["text"], cuad_chunks)
                
                # Summarize
                if not use_mock:
                    tokenizer, summarizer_model, classifier = get_neural_models()
                    gen_summary = summarize_chunks(cuad_chunks, tokenizer, summarizer_model, use_mock=False)
                else:
                    gen_summary = summarize_chunks(cuad_chunks, None, None, use_mock=True)
                    
                # Evaluate summary against reference summary
                ref_summary = contract["reference_summary"]
                summary_metrics = evaluation.evaluate_summary(gen_summary, ref_summary)
                
                # Evaluate zero-shot clause classification
                # Feed ground truth clauses to classifier and evaluate prediction accuracy
                ref_clauses = contract["reference_clauses"]
                clause_texts = [c["text"] for c in ref_clauses]
                true_categories = [c["category"] for c in ref_clauses]
                
                if not use_mock:
                    # Run classifier
                    pred_results = classify_clauses(clause_texts, classifier, use_mock=False)
                else:
                    pred_results = classify_clauses(clause_texts, None, use_mock=True)
                    
                predicted_categories = [p["category"] for p in pred_results]
                
                classification_metrics = evaluation.evaluate_classification(true_categories, predicted_categories)
                
            st.success("Analysis Complete!")
            
            # Display metrics columns
            col_e1, col_e2 = st.columns([1, 1])
            
            with col_e1:
                st.markdown("### 📝 Summarization Metrics")
                st.markdown(f"**ROUGE-1 F1-Score:** `{summary_metrics['rouge1']:.4f}`")
                st.markdown(f"**ROUGE-2 F1-Score:** `{summary_metrics['rouge2']:.4f}`")
                st.markdown(f"**ROUGE-L F1-Score:** `{summary_metrics['rougeL']:.4f}`")
                
                bs_glow = "🟢 (Real BERTScore)" if summary_metrics["is_bertscore_real"] else "🟡 (Offline Fallback Similarity)"
                st.markdown(f"**BERTScore F1:** `{summary_metrics['bertscore_f1']:.4f}` {bs_glow}")
                
                with st.expander("Compare Generated vs Reference Summary"):
                    st.write("**Reference Summary:**")
                    st.caption(ref_summary)
                    st.write("**Generated Summary:**")
                    st.info(gen_summary)
                    
            with col_e2:
                st.markdown("### 🔍 Zero-Shot Classification Accuracy")
                st.markdown(f"**Precision (Weighted):** `{classification_metrics['precision']:.4f}`")
                st.markdown(f"**Recall (Weighted):** `{classification_metrics['recall']:.4f}`")
                st.markdown(f"**F1-Score (Weighted):** `{classification_metrics['f1']:.4f}`")
                
                # Render breakdown per category
                report_data = []
                for cat, v in classification_metrics["class_report"].items():
                    report_data.append({
                        "Category": cat,
                        "Precision": f"{v['precision']:.3f}",
                        "Recall": f"{v['recall']:.3f}",
                        "F1-Score": f"{v['f1-score']:.3f}",
                        "Support": v['support']
                    })
                if report_data:
                    st.table(pd.DataFrame(report_data))
                else:
                    st.warning("Insufficient predictions to generate per-class metrics report.")
                    
                with st.expander("View Ground Truth vs Prediction Comparison"):
                    comp_data = []
                    for idx, c_text in enumerate(clause_texts):
                        comp_data.append({
                            "Clause Text": c_text[:80] + "...",
                            "True Category": true_categories[idx],
                            "Predicted Category": predicted_categories[idx],
                            "Confidence": f"{pred_results[idx]['confidence']:.2%}"
                        })
                    st.dataframe(pd.DataFrame(comp_data))

# ----------------- TAB 3: DATABASE & LOGS -----------------
with tab3:
    st.subheader("🗄️ SQLite Database Logs Explorer")
    
    docs_df = get_documents_df()
    
    if docs_df.empty:
        st.info("No documents stored in the database yet. Go to 'Document Analyzer' and analyze a contract.")
    else:
        st.dataframe(docs_df)
        
        st.write("---")
        st.subheader("Inspect and Delete Records")
        
        # Load select box for deletion / inspection
        doc_map = {row['doc_name']: row['id'] for _, row in docs_df.iterrows()}
        selected_inspect_name = st.selectbox("Select Record to Inspect / Delete", options=list(doc_map.keys()))
        
        if selected_inspect_name:
            doc_db_id = doc_map[selected_inspect_name]
            
            col_act1, col_act2 = st.columns([1, 1])
            
            with col_act1:
                # View summary and clauses from SQLite
                st.markdown(f"**Document Title:** `{selected_inspect_name}` (SQLite ID: `{doc_db_id}`)")
                
                # Fetch clauses
                c_df = get_clauses_df(doc_db_id)
                st.write(f"Associated Clauses Extracted: **{len(c_df)}**")
                if not c_df.empty:
                    st.dataframe(c_df[['clause_text', 'category', 'confidence', 'uncertainty']])
                else:
                    st.warning("No clauses stored for this document.")
                    
            with col_act2:
                # Expander for summary
                doc_record = docs_df[docs_df['id'] == doc_db_id].iloc[0]
                with st.expander("Inspect SQLite Summary"):
                    st.write(doc_record['summary'])
                    
                # Delete record button
                if st.button("❌ Permanent Delete Record", use_container_width=True):
                    with st.spinner("Deleting record from databases..."):
                        # Delete from SQLite
                        delete_document_by_id(doc_db_id)
                        # Delete from RAG vector store
                        if RAG_AVAILABLE:
                            vector_db.delete_document(selected_inspect_name)
                    st.success("Record deleted successfully! Refreshing logs...")
                    st.rerun()
                    
        st.write("---")
        st.subheader("Database Utilities")
        
        # SQLite download button
        db_path = 'legal_docs.db'
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                st.download_button(
                    label="💾 Download SQLite Database File",
                    data=f,
                    file_name="legal_docs.db",
                    mime="application/x-sqlite3",
                    use_container_width=True
                )
