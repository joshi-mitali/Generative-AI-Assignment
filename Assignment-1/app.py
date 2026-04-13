"""
app.py — Streamlit entry point for AskLM.

Renders the sidebar (config + file upload) and the main chat panel.
Manages all session state with proper initialization guards.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load env vars BEFORE any other imports that might need them
load_dotenv()

# Set CrewAI storage to project-local directory
os.environ["CREWAI_STORAGE_DIR"] = os.path.join(os.path.dirname(__file__), "crewai_storage")

# LiteLLM expects GEMINI_API_KEY for Gemini models, but our .env uses GOOGLE_API_KEY
if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Prevent CrewAI's default KnowledgeStorage from throwing OpenAI missing key errors
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-dummy-key"

# Suppress noisy pypdf "CropBox missing" warnings (harmless — defaults to MediaBox)
import logging
import warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="CropBox missing")

from src.asklm.knowledge import build_knowledge_sources, clear_knowledge_folder as reset_knowledge_base
from src.asklm.crew import AskLMCrew
from crewai import LLM
from src.ui.sidebar import render_sidebar
from src.ui.chat import render_chat

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AskLM — Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark-mode custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Global ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

/* ---- Sidebar glass effect ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
    border-right: 1px solid rgba(99, 126, 234, 0.15);
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c9d1d9 !important;
}

/* ---- Buttons ---- */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.45);
}
.stButton > button:not([kind="primary"]) {
    border: 1px solid rgba(99, 126, 234, 0.4) !important;
    color: #c9d1d9 !important;
    transition: all 0.3s ease;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #667eea !important;
    background: rgba(102, 126, 234, 0.1) !important;
}

/* ---- Chat messages ---- */
.stChatMessage {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    animation: fadeIn 0.3s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---- Chat input ---- */
.stChatInput > div {
    border: 1px solid rgba(99, 126, 234, 0.3) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease;
}
.stChatInput > div:focus-within {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.15) !important;
}

/* ---- File uploader ---- */
section[data-testid="stFileUploader"] > div {
    border: 2px dashed rgba(99, 126, 234, 0.3) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease;
}
section[data-testid="stFileUploader"] > div:hover {
    border-color: #667eea !important;
}

/* ---- Expanders ---- */
.streamlit-expanderHeader {
    background: rgba(22, 27, 34, 0.6) !important;
    border-radius: 8px !important;
}

/* ---- Selectbox / Slider ---- */
.stSelectbox > div > div,
.stSlider > div {
    transition: all 0.2s ease;
}

/* ---- Dividers ---- */
hr {
    border-color: rgba(99, 126, 234, 0.12) !important;
}

/* ---- Toast ---- */
div[data-testid="stToast"] {
    background: #1a1a2e !important;
    border: 1px solid rgba(99, 126, 234, 0.3) !important;
    color: #c9d1d9 !important;
    border-radius: 10px !important;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99, 126, 234, 0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(99, 126, 234, 0.5); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization (guarded — only runs once)
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "knowledge_sources" not in st.session_state:
    st.session_state.knowledge_sources = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
config, uploaded_files, ingest_clicked, reset_clicked = render_sidebar()

# ---------------------------------------------------------------------------
# Handle Reset
# ---------------------------------------------------------------------------
if reset_clicked:
    reset_knowledge_base()
    st.session_state.knowledge_sources = []
    st.session_state.ingested_files = set()
    st.session_state.kb_ready = False
    st.session_state.chat_history = []
    st.toast("Knowledge base cleared!", icon="🗑️")
    st.rerun()

# ---------------------------------------------------------------------------
# Handle Ingest
# ---------------------------------------------------------------------------
if ingest_clicked:
    # Validate API key first
    api_error = config.validate_api_key()
    if api_error:
        st.error(f"⚠️ {api_error}")
    elif not uploaded_files:
        st.warning("Please upload at least one document first.")
    else:
        with st.spinner("Saving documents to knowledge base..."):
            import os
            knowledge_dir = os.path.join(os.getcwd(), "knowledge")
            os.makedirs(knowledge_dir, exist_ok=True)
            
            saved_count = 0
            for f in uploaded_files:
                try:
                    filepath = os.path.join(knowledge_dir, f.name)
                    with open(filepath, "wb") as out_f:
                        out_f.write(f.getbuffer())
                    st.session_state.ingested_files.add(f.name)
                    saved_count += 1
                except Exception as e:
                    st.error(f"Failed to save {f.name}: {e}")

            if saved_count > 0:
                # Build CrewAI native knowledge sources wrapper
                sources = build_knowledge_sources(config)
                st.session_state.knowledge_sources = sources
                st.session_state.kb_ready = True
                
                st.toast(
                    f"Saved {saved_count} documents to knowledge base!",
                    icon="✅",
                )
                st.rerun()

# ---------------------------------------------------------------------------
# Main panel header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <h1 style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        margin-bottom: 0;
    ">AskLM</h1>
    <p style="color: #888; margin-top: 4px;">
        Intelligent Document Q&A — powered by CrewAI
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
query = render_chat(config)

if query:
    if not st.session_state.kb_ready:
        st.warning("Please upload and ingest documents before asking questions.")
    else:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": query,
        })

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(query)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    llm = LLM(
                        model=config.get_llm_model_string(),
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    )

                    import os
                    knowledge_dir = os.path.join(os.getcwd(), "knowledge")
                    filenames = []
                    if os.path.exists(knowledge_dir):
                        filenames = [
                            f for f in os.listdir(knowledge_dir)
                            if os.path.isfile(os.path.join(knowledge_dir, f))
                        ]
                    available_files = "\n".join(f"- {f}" for f in filenames) if filenames else "- (none)"

                    asklm_crew = AskLMCrew(
                        llm=llm,
                        knowledge_sources=st.session_state.knowledge_sources,
                        embedder=config.get_embedder_config()
                    )

                    output = asklm_crew.crew().kickoff(inputs={
                        "query": query,
                        "available_files": available_files
                    })
                    answer = str(output.raw)

                    st.markdown(answer)

                    # Save to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                    })
                except Exception as e:
                    error_msg = f"Error generating answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
