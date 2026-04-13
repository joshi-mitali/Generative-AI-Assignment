# Generative AI Assignments

This repository contains two major Generative AI projects demonstrating advanced prompt engineering, RAG architecture, and Large Language Model (LLM) fine-tuning.

## 📂 [Assignment 1: AskLM — Intelligent Document Q&A Assistant](./Assignment-1/)

**AskLM** is a configurable Retrieval-Augmented Generation (RAG) system built entirely on the **CrewAI** framework and hosted on a **Streamlit** user interface. 

**Key Features:**
- **Multi-Format Support:** Easily upload PDFs, DOCX, TXT, or Markdown documents to build the context engine.
- **Agentic Workflow:** Employs a dedicated CrewAI "Document QA Specialist" agent to orchestrate the pipeline from searching the vector embedding store to generating grounded responses.
- **Dynamic Configuration:** End-users can swap out LLMs (Gemini/Groq), embedders (Gemini/HuggingFace), vector chunk sizes, and retrieval top-K strategies via the UI.
- **Persistent Knowledge:** Uses ChromaDB beneath the engine for semantic and precise knowledge retrieval.

## 📂 [Assignment 2: MediFirst Medical Assistant](./Assignment-2/)

**MediFirst** is a fine-tuning project that builds a domain-specific emergency and medical Q&A assistant utilizing the **Google Gemma 3** open-weights base model.

**Key Features:**
- **Synthetic Data Generation:** Includes an automated pipeline using Python, asyncio, and Gemini 2.0 Flash to synthesize ~3,500 highly-detailed, medically-structured emergency-response examples.
- **QLoRA Fine-Tuning:** Features a memory-efficient training notebook optimized for Google Colab/T4 environments utilizing Parameter-Efficient Fine-Tuning (PEFT) alongside 4-bit Quantization (bitsandbytes).
- **Hugging Face Stack:** Deploys the TRL (`SFTTrainer`), Accelerate, and Transformers libraries to tune Gemma 3 on custom systemic prompts and instructions.

---
*Note: Refer to the README files located inside each assignment directory for extensive setup instructions and commands.*


