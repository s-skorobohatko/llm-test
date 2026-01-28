# llm-infra (RAG without WebUI)

This repo provides a simple local RAG pipeline for Ollama:
- Ingest docs/repos -> chunk -> embed -> store in SQLite
- Ask questions -> retrieve top chunks -> inject into prompt -> chat with your Ollama model

## Setup

```bash
cd rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# pull embedding model
ollama pull nomic-embed-text 
