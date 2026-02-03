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




++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Puppet Refactor Pipeline (v1) — Local RAG + Two-Pass Plan→Diff (CPU-first)

This project builds a **local, production-shaped** Puppet refactoring pipeline that:

- indexes Puppet 8 “good practice” knowledge into **Qdrant** (RAG)
- scans a target Puppet module from the **filesystem**
- generates a **two-pass** LLM output:
  1) **Plan** (what to change, which files)
  2) **Diff** (minimal edits, strict file path enforcement)

**v1 intentionally does NOT run `puppet-lint` or `puppet parser validate`.**
Those are added in later iterations once the pipeline is stable and fast.

---

## Architecture Overview

### Two separate data sources

1) **Knowledge Base (Qdrant / `puppet_governance`)**
   - Puppet style guides
   - Wikimedia Puppet guidelines
   - your internal “gold standard” modules
   - ingested once (or periodically)

2) **Target Module (filesystem)**
   - the module you want to refactor
   - scanned at runtime using `module_path`
   - becomes the authoritative `MODULE_CONTEXT` for the LLM

The LLM is forced to:
- only reference files listed in `MODULE_FILES`
- only propose diffs for those files  
If it outputs a diff for a file not in `MODULE_FILES`, the server rejects it.

---

## Requirements

### Hardware / runtime
- Host VM: **64GB RAM**
- CPU-first inference via **Ollama**
- Qdrant running locally (Docker)

### Software
- Python 3.10+
- Docker + docker-compose
- Ollama installed and running

---

## Models

### Embeddings (RAG)
- `nomic-embed-text`

### Refactoring model (CPU-friendly)
- `qwen2.5-coder:7b-instruct-q4_K_M`

Pull models:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
