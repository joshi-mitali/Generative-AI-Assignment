"""
ui/chat.py — Chat history rendering, active config bar.
"""

from __future__ import annotations
import streamlit as st
from src.config import AppConfig


def render_config_bar(config: AppConfig):
    """Show active configuration summary at the top of the main panel."""
    llm_display = config.llm_model.split("/")[-1]
    if config.embedding_provider == "google":
        embed_display = "Gemini embedding-001"
    else:
        embed_display = config.hf_embedding_model.split("/")[-1]

    st.markdown(
        f"""<div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #30475e;
            border-radius: 10px;
            padding: 12px 20px;
            margin-bottom: 20px;
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            font-size: 0.85em;
        ">
            <span>🤖 <strong>{llm_display}</strong></span>
            <span>🔗 <strong>{embed_display}</strong></span>
            <span>🌡️ <strong>Temp {config.temperature}</strong></span>
        </div>""",
        unsafe_allow_html=True,
    )


def render_chat(config: AppConfig):
    """
    Render the main chat panel.
    - Active config summary bar
    - Chat history natively cited by LLM
    - Input box
    """
    render_config_bar(config)

    # ---- Chat history ----
    messages = st.session_state.get("chat_history", [])

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- Chat input ----
    query = st.chat_input("Ask a question about your documents...")
    return query
