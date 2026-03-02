# Minimal RAG Demo

A clean, production-safe Retrieval-Augmented Generation (RAG) system built in Python — no LangChain, no LlamaIndex, no GPU required.

## Project Structure

```
rag-mini/
├── docs/
│   ├── sample1.txt        # Sample document: RAG concepts
│   └── sample2.txt        # Sample document: NLP, LLMs, vector databases
├── ingest.py              # Ingestion pipeline (chunking → embeddings → FAISS)
├── query.py               # Query interface (retrieve → generate answer)
├── test_rag.py            # Self-test suite
├── requirements.txt       # Pinned dependencies
└── README.md              # This file
```

---

## 🚀 Use Cases

Demonstrating core RAG architecture without frameworks

Learning how vector search works under the hood

Technical assessment / interview demo project

Foundation for building document Q&A systems

Lightweight internal knowledge search tool

Base template for scaling into production RAG systems

## Setup

### 1. Prerequisites

- Python 3.10 or higher
- Internet connection (for downloading the embedding model on first run)

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `faiss-cpu` requires a recent version of pip. If installation fails, upgrade pip first:
> ```bash
> pip install --upgrade pip
> ```

### 4. Set your OpenAI API key

**Windows (PowerShell — current session):**
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**Windows (persistent, for all sessions):**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-...", "User")
```

**macOS / Linux:**
```bash
export OPENAI_API_KEY="sk-..."
```

> Get your API key from: https://platform.openai.com/api-keys

---

## Running the Project

### Step 1 — Ingest documents

```bash
python ingest.py
```

This will:
- Load all `.txt` files from `docs/`
- Chunk each document (500-char chunks, 50-char overlap)
- Generate embeddings using `all-MiniLM-L6-v2`
- Save `index.faiss` and `metadata.json` to the project root

**Expected output:**
```
============================================================
RAG Ingestion Pipeline
============================================================

[1/4] Loading documents...
  Loaded: sample1.txt (4821 chars)
  Loaded: sample2.txt (5012 chars)
  Loaded 2 document(s).

[2/4] Chunking documents...
  Chunked 'sample1.txt' into 11 chunks.
  Chunked 'sample2.txt' into 12 chunks.
  Total unique chunks: 23

[3/4] Loading embedding model 'all-MiniLM-L6-v2'...

Generating embeddings for 23 chunks...
100%|████████████████████| 23/23 [00:01<00:00, ...]

[4/4] Building FAISS index and saving artifacts...
  FAISS index built: 23 vectors, dimension=384
  Saved FAISS index to 'index.faiss'
  Saved metadata to 'metadata.json'

============================================================
Ingestion complete.
  Documents : 2
  Chunks    : 23
  Index     : index.faiss
  Metadata  : metadata.json
============================================================
```

---

### Step 2 — Query the system

```bash
python query.py
```

**Example session:**

```
Loading FAISS index and metadata...
  Index loaded: 23 vectors
  Metadata loaded: 23 chunks

Loading embedding model 'all-MiniLM-L6-v2'...

RAG Query Interface (type 'exit' or 'quit' to stop)

Your question: What is Retrieval-Augmented Generation?

============================================================
Retrieved 3 chunk(s):
============================================================

[Chunk 1]
  Source   : sample1.txt
  Distance : 0.4812
  Text     : Retrieval-Augmented Generation (RAG) is a technique in natural
             language processing that combines the power of large language ...

[Chunk 2]
  Source   : sample1.txt
  Distance : 0.6104
  Text     : The core idea behind RAG is simple: when a user asks a question,
             the system first converts that question into an embedding vector ...

[Chunk 3]
  Source   : sample2.txt
  Distance : 0.7233
  Text     : Hallucination in language models refers to the phenomenon where a
             model generates plausible-sounding but factually incorrect ...

============================================================

Generating answer...

============================================================
ANSWER:
============================================================
Retrieval-Augmented Generation (RAG) is a technique in natural language
processing that combines large language models (LLMs) with external knowledge
retrieval. Instead of relying solely on the model's parametric knowledge, RAG
systems first retrieve relevant documents from a knowledge base, then use those
documents as context to generate accurate, grounded answers — reducing
hallucinations and improving factual accuracy.
============================================================
```

---

### Step 3 — Run self-tests

```bash
python test_rag.py
```

This will automatically:
1. Run the full ingestion pipeline
2. Verify `index.faiss` and `metadata.json` exist
3. Validate the FAISS index is loadable and non-empty
4. Check metadata consistency
5. Retrieve top-3 chunks for the test question
6. Verify the LLM answer is non-empty (skipped gracefully if no API key)

**Expected output (with API key set):**
```
============================================================
RAG Self-Test Suite
============================================================

[Test 1] Running ingestion pipeline (python ingest.py)...
  [PASS] Ingestion completed successfully (exit code 0).

[Test 2] Verifying artifacts exist on disk...
  [PASS] 'index.faiss' exists (12,412 bytes).
  [PASS] 'metadata.json' exists (28,944 bytes).

[Test 3] Loading and validating FAISS index...
  [PASS] FAISS index loaded: 23 vectors, dimension=384.

[Test 4] Loading and validating metadata...
  [PASS] Metadata loaded: 23 chunks, all keys valid.

[Test 5] Retrieving top-3 chunks for: 'What is Retrieval-Augmented Generation?'
  [PASS] 3 chunks retrieved as expected.
    Chunk 1: source='sample1.txt', text_preview='Retrieval-Augmented Generation (RAG) is a technique...'
    Chunk 2: source='sample1.txt', text_preview='The core idea behind RAG is simple...'
    Chunk 3: source='sample2.txt', text_preview='Hallucination in language models...'

[Test 6] Verifying answer generation...
  [PASS] Non-empty answer received (312 chars).

============================================================
ALL TESTS PASSED
============================================================
```

---

## Adding Your Own Documents

1. Place any `.txt` files into the `docs/` folder.
2. Re-run `python ingest.py` to rebuild the index.
3. Query as normal with `python query.py`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `FileNotFoundError: FAISS index not found` | Run `python ingest.py` before querying |
| `OPENAI_API_KEY not set` | Set the environment variable as shown in Setup §4 |
| `AuthenticationError` | Your API key is invalid or has expired — generate a new one |
| `APIConnectionError` | Check your internet connection and VPN settings |
| `No .txt files found in 'docs/'` | Add at least one `.txt` document to the `docs/` folder |
| `UnicodeDecodeError` | `ingest.py` automatically retries with `latin-1` fallback |
| Slow embedding on first run | The model (~90 MB) is downloaded once and cached locally |
| `pip install faiss-cpu` fails | Upgrade pip: `pip install --upgrade pip`, then retry |

---

## Architecture Overview

```
docs/*.txt
    │
    ▼
[ingest.py]
    ├── Load & decode .txt files
    ├── Chunk text (500 chars, 50 overlap)
    ├── Deduplicate chunks (MD5 hash)
    ├── Embed with all-MiniLM-L6-v2 (384-dim)
    ├── Build FAISS IndexFlatL2
    └── Save index.faiss + metadata.json
             │
             ▼
[query.py]
    ├── Load index + metadata
    ├── Embed user question (same model)
    ├── Search FAISS → top-3 chunks
    ├── Build grounded prompt
    └── Call OpenAI (temperature=0) → answer
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `sentence-transformers` | 3.4.1 | Text embeddings |
| `faiss-cpu` | 1.10.0 | Vector similarity search |
| `numpy` | 1.26.4 | Array operations |
| `openai` | 1.65.4 | LLM API client |
| `tqdm` | 4.67.1 | Progress bars |

---

## Design Principles

- **No hallucination**: The LLM is strictly instructed to use only retrieved context and respond with `"I don't know."` if the answer is absent.
- **Deterministic output**: `temperature=0` ensures reproducible answers.
- **No GPU required**: All components run on CPU.
- **No heavy frameworks**: Pure Python with five focused libraries.
- **Production-safe error handling**: Every failure mode is caught and reported clearly.

## 🎯 Purpose

This project demonstrates a clean implementation of Retrieval-Augmented Generation (RAG) from scratch without high-level frameworks. It focuses on clarity, reproducibility, and deterministic grounded responses.

It runs fully on CPU and is suitable for small-scale document Q&A tasks.
