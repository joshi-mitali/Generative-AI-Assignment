"""
ui/sidebar.py — Streamlit sidebar: configuration widgets + file upload.

Returns (AppConfig, uploaded_files, ingest_clicked, reset_clicked).
"""

from __future__ import annotations
import streamlit as st
from src.config import (
    AppConfig,
    GEMINI_MODELS,
    GROQ_MODELS,
    HF_EMBEDDING_MODELS,
    CHUNK_SIZE_OPTIONS,
    CHUNK_OVERLAP_OPTIONS,
    MAX_TOKEN_OPTIONS,
    TOP_K_OPTIONS,
)


def render_sidebar() -> tuple:
    """
    Render the full sidebar and return the current configuration.

    Returns
    -------
    tuple of (AppConfig, list[UploadedFile], bool, bool)
        (config, uploaded_files, ingest_clicked, reset_clicked)
    """
    with st.sidebar:
        st.markdown("## 📄 AskLM")
        st.caption("Intelligent Document Q&A")

        # ---- File Upload ----
        st.markdown("---")
        st.markdown("### 📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "csv", "json", "xlsx"],
            accept_multiple_files=True,
            key="file_uploader",
        )

        # Show file status
        if uploaded_files:
            for f in uploaded_files:
                ingested = f.name in st.session_state.get("ingested_files", set())
                icon = "✅" if ingested else "⏳"
                st.caption(f"{icon} {f.name}")

        # ---- LLM Configuration ----
        st.markdown("---")
        st.markdown("### 🤖 LLM Settings")

        llm_provider = st.selectbox(
            "LLM Provider",
            options=["gemini", "groq"],
            index=0,
            key="llm_provider",
        )

        if llm_provider == "gemini":
            model_options = list(GEMINI_MODELS.keys())
            model_labels = list(GEMINI_MODELS.values())
            default_idx = 1  # gemini-2.5-flash
        else:
            model_options = list(GROQ_MODELS.keys())
            model_labels = list(GROQ_MODELS.values())
            default_idx = 0

        llm_model = st.selectbox(
            "Model",
            options=model_options,
            format_func=lambda x: (
                GEMINI_MODELS.get(x, x)
                if llm_provider == "gemini"
                else GROQ_MODELS.get(x, x)
            ),
            index=default_idx,
            key="llm_model",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            key="temperature",
        )

        max_tokens = st.selectbox(
            "Max Output Tokens",
            options=MAX_TOKEN_OPTIONS,
            index=2,  # 1024
            key="max_tokens",
        )

        # ---- Embedding Configuration ----
        st.markdown("---")
        st.markdown("### 🔗 Embedding Settings")

        embedding_provider = st.selectbox(
            "Embedding Provider",
            options=["google", "huggingface"],
            format_func=lambda x: "Gemini API" if x == "google" else "HuggingFace (Local)",
            index=0,
            key="embedding_provider",
        )

        hf_model = HF_EMBEDDING_MODELS[0]
        if embedding_provider == "huggingface":
            hf_model = st.selectbox(
                "HuggingFace Model",
                options=HF_EMBEDDING_MODELS,
                index=0,
                key="hf_model",
            )

        # ---- Chunking Configuration ----
        st.markdown("---")
        st.markdown("### 📐 Chunking Settings")

        chunk_size = st.selectbox(
            "Chunk Size (tokens)",
            options=CHUNK_SIZE_OPTIONS,
            index=1,  # 512
            key="chunk_size",
        )

        chunk_overlap = st.selectbox(
            "Chunk Overlap (tokens)",
            options=CHUNK_OVERLAP_OPTIONS,
            index=1,  # 64
            key="chunk_overlap",
        )

        # ---- Retrieval Configuration ----
        st.markdown("---")
        st.markdown("### 🔍 Retrieval Settings")

        top_k = st.selectbox(
            "Top-K Results",
            options=TOP_K_OPTIONS,
            index=1,  # 5
            key="top_k",
        )

        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            key="similarity_threshold",
        )

        # ---- Action Buttons ----
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            ingest_clicked = st.button(
                "🚀 Ingest",
                use_container_width=True,
                type="primary",
            )
        with col2:
            reset_clicked = st.button(
                "🗑️ Reset KB",
                use_container_width=True,
            )

    # ---- Build config ----
    config = AppConfig(
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        embedding_provider=embedding_provider,
        hf_embedding_model=hf_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    # ---- API key validation ----
    api_error = config.validate_api_key()
    if api_error and (ingest_clicked or st.session_state.get("kb_ready")):
        st.sidebar.error(f"⚠️ {api_error}")

    return config, uploaded_files, ingest_clicked, reset_clicked
