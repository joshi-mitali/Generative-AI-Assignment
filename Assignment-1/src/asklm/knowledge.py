"""
crew/knowledge.py — Build CrewAI Knowledge sources from files in ./knowledge.

Uses native CrewAI file-based knowledge sources. chunk_size and chunk_overlap
are passed through from AppConfig so retrieval granularity is user-configurable.
"""

from __future__ import annotations
import os
from src.config import AppConfig

# --- Monkey-patch: CrewAI's KnowledgeStorage.save passes empty metadata ---
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage

_orig_save = KnowledgeStorage.save


def _fixed_save(self, documents, metadata=None):
    """Ensure metadata is valid and deduplicate documents to avoid ChromaDB errors."""
    import hashlib

    n = len(documents)
    if n == 0:
        return _orig_save(self, documents, metadata)

    default_meta = {"source": "file"}

    # Case 1: metadata is None or falsy → generate defaults
    if not metadata:
        metadata = [default_meta.copy() for _ in range(n)]
    # Case 2: metadata is a single dict → replicate it
    elif isinstance(metadata, dict):
        meta = metadata if metadata else default_meta
        metadata = [meta.copy() for _ in range(n)]
    # Case 3: metadata is a list → ensure each element is a non-empty dict
    elif isinstance(metadata, list):
        # Pad if shorter than documents
        while len(metadata) < n:
            metadata.append(default_meta.copy())
        # Replace any empty dicts with the default
        for i in range(len(metadata)):
            if not isinstance(metadata[i], dict) or not metadata[i]:
                metadata[i] = default_meta.copy()

    # Deduplicate documents by content hash to prevent ChromaDB
    # "Expected IDs to be unique" errors on re-ingestion
    seen_hashes = set()
    deduped_docs = []
    deduped_meta = []
    for i, doc in enumerate(documents):
        content_hash = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            deduped_docs.append(doc)
            deduped_meta.append(metadata[i])

    if len(deduped_docs) == 0:
        return

    return _orig_save(self, deduped_docs, deduped_meta)


KnowledgeStorage.save = _fixed_save
# --------------------------------------------------

from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource

try:
    from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
except ImportError:
    ExcelKnowledgeSource = None
try:
    from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
except ImportError:
    JSONKnowledgeSource = None

# Extension → CrewAI source class
SOURCE_MAPPINGS = {
    ".txt": TextFileKnowledgeSource,
    ".md": TextFileKnowledgeSource,
    ".pdf": PDFKnowledgeSource,
    ".csv": CSVKnowledgeSource,
    ".xlsx": ExcelKnowledgeSource,
    ".json": JSONKnowledgeSource,
}


def build_knowledge_sources(config: AppConfig) -> list:
    """
    Scan ./knowledge for files and return native CrewAI knowledge sources.

    chunk_size and chunk_overlap from AppConfig are forwarded to each source
    so the user can tune retrieval granularity from the sidebar.
    """
    knowledge_dir = os.path.join(os.getcwd(), "knowledge")
    if not os.path.exists(knowledge_dir):
        return []

    # Group filenames by extension
    files_by_ext: dict[str, list[str]] = {}
    for filename in os.listdir(knowledge_dir):
        if os.path.isfile(os.path.join(knowledge_dir, filename)):
            ext = os.path.splitext(filename)[1].lower()
            files_by_ext.setdefault(ext, []).append(filename)

    sources = []
    for ext, filenames in files_by_ext.items():
        SourceClass = SOURCE_MAPPINGS.get(ext)
        if SourceClass is None:
            print(f"Warning: extension '{ext}' not supported — skipping.")
            continue

        # Common kwargs: chunk_size / chunk_overlap are accepted by every source
        common = {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        }

        # Try batch (file_paths=[...]) first, then individual (file_path=...)
        try:
            source = SourceClass(file_paths=filenames, **common)
            sources.append(source)
        except Exception:
            for fname in filenames:
                try:
                    source = SourceClass(file_path=fname, **common)
                    sources.append(source)
                except Exception as e:
                    print(f"Failed to load {fname}: {e}")

    return sources


def clear_knowledge_folder():
    """Delete every file inside ./knowledge (auto-cleanup)."""
    knowledge_dir = os.path.join(os.getcwd(), "knowledge")
    if not os.path.exists(knowledge_dir):
        return
    for filename in os.listdir(knowledge_dir):
        filepath = os.path.join(knowledge_dir, filename)
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to delete {filepath}: {e}")
