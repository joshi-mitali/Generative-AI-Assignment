# AskLM — Intelligent Document Q&A Assistant

> Retrieval-Augmented Generation (RAG) for Domain-Specific Question Answering  
> Powered entirely by **CrewAI**

AskLM is a configurable, document-centric question-answering system. Upload documents from any domain, tune the full RAG pipeline via a clean settings panel, and receive accurate answers grounded directly in your uploaded content.

## Architecture

```
User Upload → File Parsing → Text Cleaning → Chunking → Embedding → ChromaDB
                                                                         │
User Query → Query Embedding → Semantic Search (Top-K) → Prompt Assembly → LLM → Answer + Citations
```

**Single-Agent Design:** One CrewAI agent (_Document QA Specialist_) owns the entire pipeline — no multi-agent hand-offs.

## Features

| Feature | Details |
|---------|---------|
| **Document Formats** | PDF, TXT, DOCX, Markdown (.md) |
| **LLM Providers** | Gemini (2.5 Pro/Flash/Flash-Lite, 3.x Preview) · Groq (LLaMA, Qwen) |
| **Embedding Models** | Gemini embedding-001 · HuggingFace (MiniLM, BGE, E5) |
| **Vector DB** | ChromaDB (persistent, via CrewAI Knowledge backend) |
| **Configurable** | Chunk size/overlap, temperature, max tokens, top-K, similarity threshold |
| **Source Citations** | Filename + page number + relevance score for every answer |

## Project Structure

```
├── app.py                    # Streamlit entry point
├── config.py                 # Pydantic settings schema
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
├── crew/
│   ├── __init__.py
│   ├── agent.py              # Document QA Agent
│   ├── knowledge.py          # CrewAI native Knowledge source builder
│   ├── pipeline.py           # Crew assembly + kickoff
│   └── task.py               # QA Task definition
├── prompts/
│   ├── __init__.py
│   └── templates.py          # Prompt templates
├── ui/
│   ├── __init__.py
│   ├── chat.py               # Chat history + config bar
│   └── sidebar.py            # Config widgets + file upload
└── tests/
    ├── e2e_test.py            # End-to-end pipeline test
    ├── test_pipeline.py       # Feature-level test suite
    └── test_docs/             # Sample documents for tests
```

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/asklm.git
cd asklm
```

### 2. Create virtual environment
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and add your API keys:
#   GOOGLE_API_KEY=your_gemini_key
#   GROQ_API_KEY=your_groq_key
```

### 5. Run the application
```bash
streamlit run app.py
```

## Configuration Options

| Setting | Options | Default |
|---------|---------|---------|
| LLM Provider | Gemini / Groq | Gemini |
| Temperature | 0.0 – 1.0 | 0.3 |
| Max Output Tokens | 256 / 512 / 1024 / 2048 | 1024 |
| Embedding Provider | Gemini API / HuggingFace | Gemini |
| Chunk Size | 256 / 512 / 1024 / 2048 | 512 |
| Chunk Overlap | 0 / 64 / 128 | 64 |
| Top-K Retrieval | 3 / 5 / 10 | 5 |
| Similarity Threshold | 0.0 – 1.0 | 0.35 |

## Technology Stack

| Layer | Library | Purpose |
|-------|---------|---------|
| AI Framework | CrewAI ≥ 0.80 | Knowledge + Agent + Task + Crew |
| Frontend | Streamlit ≥ 1.35 | Config panel, file upload, chat |
| Vector DB | ChromaDB ≥ 0.5 | Persistent vector storage |
| File Parsers | pypdf, python-docx | Raw text extraction |

> **Note:** LangChain is explicitly excluded. CrewAI is the sole AI framework.

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GOOGLE_API_KEY` | Yes (for Gemini) | Gemini LLM + embedding API |
| `GROQ_API_KEY` | For Groq models | Groq LLM API |
| `HF_TOKEN` | Optional | HuggingFace gated models |
