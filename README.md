# Local LLM (RAG)

A fully local RAG (Retrieval Augmented Generation) pipeline for chatting with your documents. No data leaves your machine.

Built with [Ollama](https://ollama.com), [LlamaIndex](https://www.llamaindex.ai), and [ChromaDB](https://www.trychroma.com).

## Demo Output

```
========================================
Local LLM Ready
Type 'q' to quit

To index a new directory, run:
  python preprocessing.py ~/path/to/folder
========================================

Q: what files are there in here
A: There is one file located at the specified path.

Q: what is the file about
A: It appears to be a survey or questionnaire submitted by a club or society
   to a potential sponsor. The purpose of the document seems to be for the
   club or society to share their information and requirements with the
   sponsor, in order to establish a mutually beneficial relationship.
```

## Features

- Fully offline — your documents never leave your machine
- Index any directory on your computer
- Supports `.pdf`, `.docx`, `.html`, `.txt`, `.md`
- Incremental indexing — only re-indexes changed or new files
- Automatically removes deleted files from the vector store

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with `llama3.1:8b` pulled (`ollama pull llama3.1:8b`)

## Installation

```bash
pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface
pip install llama-index-vector-stores-chroma chromadb
pip install llama-index-readers-file unstructured docx2txt
```

## Usage

**1. Index a directory:**
```bash
python preprocessing.py /path/to/your/documents
```

**2. Start chatting:**
```bash
python chat.py
```

Type `q` to quit.

## Project Structure

```
local-rag/
├── chat.py              # Interactive chat interface
├── preprocessing.py     # Document indexing pipeline
├── chroma_db/           # Vector store (auto-generated)
├── models/              # Cached embedding models (auto-generated)
├── indexed_files.json   # Index tracking log (auto-generated)
└── data/                # Default document directory (optional)
```

## Models

| Component | Model |
|-----------|-------|
| LLM | `llama3.1:8b` via Ollama |
| Embeddings | `nomic-ai/nomic-embed-text-v1.5` via HuggingFace |
