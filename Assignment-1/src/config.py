"""
config.py — Pydantic settings schema for all AskLM configurable parameters.

Every value is passed directly into CrewAI's Knowledge, LLM, or Agent
constructors at runtime.
"""

import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model catalogs — single source of truth for dropdown options
# ---------------------------------------------------------------------------

GEMINI_MODELS = {
    "gemini/gemini-2.5-pro": "Gemini 2.5 Pro (Stable)",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash (Stable) — Default",
    "gemini/gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite (Stable)",
    "gemini/gemini-3.1-pro-preview": "Gemini 3.1 Pro (Preview)",
    "gemini/gemini-3-flash-preview": "Gemini 3 Flash (Preview)",
    "gemini/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite (Preview)",
}

GROQ_MODELS = {
    "groq/llama-3.1-8b-instant": "LLaMA 3.1 8B Instant",
    "groq/llama-3.3-70b-versatile": "LLaMA 3.3 70B Versatile",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": "LLaMA 4 Scout (Preview)",
    "groq/meta-llama/llama-4-maverick-17b-128e-instruct": "LLaMA 4 Maverick (Preview)",
    "groq/qwen/qwen-3-32b": "Qwen3 32B (Preview)",
}

HF_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-large-v2",
]

CHUNK_SIZE_OPTIONS = [256, 512, 1024, 2048]
CHUNK_OVERLAP_OPTIONS = [0, 64, 128]
MAX_TOKEN_OPTIONS = [256, 512, 1024, 2048]
TOP_K_OPTIONS = [3, 5, 10]


class AppConfig(BaseModel):
    """All user-configurable RAG pipeline settings."""

    # LLM settings
    llm_provider: str = Field(default="gemini", description="gemini or groq")
    llm_model: str = Field(
        default="gemini/gemini-2.5-flash",
        description="Full model string with provider prefix",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024)

    # Embedding settings
    embedding_provider: str = Field(
        default="google", description="google or huggingface"
    )
    hf_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Chunking settings
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # Retrieval settings
    top_k: int = Field(default=5)
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    # ----- derived helpers ------------------------------------------------

    def get_llm_model_string(self) -> str:
        """Return the full provider/model string for CrewAI LLM class."""
        return self.llm_model  # already stored with prefix

    def get_embedder_config(self) -> dict:
        """Return the embedder dict expected by CrewAI Agent / Crew."""
        if self.embedding_provider == "google":
            return {
                "provider": "google",
                "config": {
                    "model": "models/gemini-embedding-001",
                    "api_key": os.getenv("GOOGLE_API_KEY"),
                },
            }
        else:
            return {
                "provider": "huggingface",
                "config": {
                    "model": self.hf_embedding_model,
                },
            }

    def validate_api_key(self) -> str | None:
        """Return an error message if the required API key is missing."""
        if self.llm_provider == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                return "GOOGLE_API_KEY is not set. Add it to your .env file."
        elif self.llm_provider == "groq":
            if not os.getenv("GROQ_API_KEY"):
                return "GROQ_API_KEY is not set. Add it to your .env file."
        # Embedding key check
        if self.embedding_provider == "google" and not os.getenv("GOOGLE_API_KEY"):
            return "GOOGLE_API_KEY is required for Gemini embeddings."
        return None
