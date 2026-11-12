# AI Document Intelligence System - Production RAG Pipeline

> Enterprise-grade Retrieval-Augmented Generation for document querying and knowledge retrieval.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![CI](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat&logo=github-actions&logoColor=white)](https://github.com/features/actions)

**Live Demo:** [https://pratham243-ai-document-intelligence.streamlit.app/](https://pratham243-ai-document-intelligence.streamlit.app/)

---

## What This Is

A production-ready RAG system that lets you query a large document corpus with natural-language questions and get accurate, grounded answers — not hallucinations.

**Key problem solved:** Standard RAG with dense embeddings alone misses exact-match queries (product codes, names, IDs). This system uses **hybrid search** — BM25 for keyword precision + semantic embeddings for conceptual recall — combined with cross-encoder re-ranking, which is how production search at companies like Elastic and Vespa works.

---

## How It Works

```
User Query (text or voice)
    │
    ▼
┌──────────────────────────────────┐
│  Guardrails Validation           │
│  (prompt injection detection)    │
└─────────┬────────────────────────┘
          │
    ┌─────┴──────┐
    │            │
    ▼            ▼
BM25 Search  Semantic Search
(keyword,   (all-MiniLM-L6-v2
 top-10)     + ChromaDB, top-10)
    │            │
    └─────┬──────┘
          │
          ▼
    Reciprocal Rank
    Fusion (k=60)
          │
          ▼
    Cross-Encoder Rerank
    (ms-marco-MiniLM-L-6-v2)
    top-20 → top-3
          │
          ▼
    LLM Generation
    (Groq → Ollama → raw chunks)
    with source citations
          │
          ▼
    FastAPI Response + LangFuse Trace
```

---

## Results (measured)

See [`tests/evaluation/results.json`](tests/evaluation/results.json) — generated from `pytest tests/evaluation/`.

| Metric | Source |
|---|---|
| Retrieval accuracy (top-3) | `results.json → summary.retrieval_accuracy_top3` |
| Average BLEU-4 | `results.json → summary.avg_bleu4` |
| Average ROUGE-L | `results.json → summary.avg_rouge_l` |
| p50 latency | `results.json → summary.p50_latency_ms` |
| p95 latency | `results.json → summary.p95_latency_ms` |
| Prompt injection block rate | `results.json → summary.prompt_injection_block_rate` |

> **Note:** BLEU-4 and ROUGE-L scores reflect no-LLM mode (raw retrieved chunks returned verbatim). With Groq or Ollama enabled, expect significantly higher answer-quality scores. Retrieval accuracy and safety metrics are LLM-independent.

Re-run the evaluation yourself:

```bash
pytest tests/evaluation/ -v -s
```

---

## Stack

| Component | Technology |
|---|---|
| Vector Store | ChromaDB (persistent, local) |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2, 384-dim) |
| Sparse Retrieval | BM25 (rank-bm25) |
| Re-ranking | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| LLM (primary) | Groq Llama 3.3 70B |
| LLM (GDPR fallback) | Ollama (llama3.1:8b, local) |
| Voice Input | faster-whisper (base.en, local) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Observability | LangFuse |
| Safety | Regex-based prompt injection detection |
| Data Versioning | DVC |
| Evaluation | BLEU-4, ROUGE-L, retrieval accuracy, latency |
| Testing | pytest |
| CI/CD | GitHub Actions |

---

## Project Structure

```
ai-document-intelligence/
├── app/
│   ├── api/              # FastAPI routes (/documents, /query, /health, etc.)
│   ├── ingestion/        # PDF/DOCX/TXT extraction + semantic chunking
│   ├── retrieval/        # BM25 + semantic hybrid search, RRF, cross-encoder
│   ├── generation/       # Triple-fallback LLM (Groq → Ollama → raw chunks)
│   ├── observability/    # LangFuse tracing
│   ├── safety/           # Prompt injection detection
│   ├── voice/            # faster-whisper transcription
│   ├── config.py         # Pydantic settings
│   └── main.py           # FastAPI app entry point
├── streamlit_app.py      # Streamlit frontend (spawns FastAPI as subprocess)
├── tests/
│   ├── test_basic.py     # Unit tests
│   └── evaluation/       # Evaluation harness
│       ├── qa_pairs.json # 30 Q&A pairs
│       ├── run_evaluation.py
│       └── results.json  # Real evaluation results
├── data/
│   └── sample_docs/      # Research paper corpus (DVC-tracked)
├── .github/workflows/    # CI pipeline
├── requirements.txt
├── packages.txt          # Streamlit Cloud apt dependencies
├── .python-version       # Pinned to 3.11
└── .env.example
```

---

## How to Run

### Local development

```bash
git clone https://github.com/Pratham243/ai-document-intelligence
cd ai-document-intelligence
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

**Backend only:**
```bash
uvicorn app.main:app --reload
```

**Full app (Streamlit + FastAPI):**
```bash
streamlit run streamlit_app.py
```

API docs at `http://localhost:8000/docs`

### Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
GROQ_API_KEY=your_groq_key     # Required for Groq mode
OLLAMA_BASE_URL=http://localhost:11434  # For GDPR/local mode
LANGFUSE_PUBLIC_KEY=...         # Optional observability
LANGFUSE_SECRET_KEY=...
```

---

## GDPR / Local-LLM Mode

For GDPR compliance or air-gapped environments, the system supports a triple-fallback LLM strategy:

1. **Groq (cloud):** Llama 3.3 70B via Groq's API — fast, high quality. Requires `GROQ_API_KEY`.
2. **Ollama (local):** Runs llama3.1:8b entirely on your machine. No data leaves your network. Set `OLLAMA_BASE_URL`.
3. **No-LLM (raw retrieval):** If both are unavailable, returns retrieved chunks verbatim with a "LLM unavailable" notice.

Toggle between modes in the Streamlit sidebar.

---

## Voice Input

Ask questions by speaking instead of typing:

- Uses **faster-whisper** (CTranslate2-based) with the `base.en` model
- Transcription runs entirely locally — no data sent to external APIs
- Model auto-downloads on first use (~150MB)
- Click the mic button in the Streamlit UI to record

---

## Observability

All retrieval and generation calls are traced via **LangFuse**:

- Each query creates a trace with retrieval span + LLM generation span
- Guardrail checks are logged (blocked count only, no PII)
- Trace URLs are shown in the Streamlit UI when LangFuse is configured
- **Graceful degradation:** If `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are not set, tracing is silently disabled

---

## Safety

Queries are validated for prompt injection before processing:

- Regex-based detection of role-override patterns, system-prompt leak attempts, jailbreak keywords
- Blocked queries return HTTP 400 with a rejection message
- Blocked attempt count (no PII) is logged to LangFuse
- The adversarial test suite checks 10 injection patterns; see evaluation results for the block rate

---

## Data Versioning

The sample document corpus is tracked with **DVC**:

```bash
dvc init          # Already done
dvc pull          # Fetch sample_docs from configured remote
```

- `.dvc/config` is set up for a local remote (no cloud needed for demo)
- `data/sample_docs.dvc` tracks the corpus
- Add new documents and run `dvc push` to version them

---

## API Endpoints

All under `/api/v1/`:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/documents` | Upload PDF/DOCX/TXT |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{id}` | Remove a document |
| POST | `/query` | Text question |
| POST | `/query/voice` | Audio question (Whisper) |
| GET | `/health` | Health check |
| GET | `/evaluation/results` | Evaluation metrics |

---

## What Makes This Production-Ready

- **Hybrid retrieval** — BM25 + semantic + cross-encoder reranking
- **Quantified evaluation** — retrieval accuracy, BLEU, ROUGE, latency, safety
- **Triple-fallback LLM** — cloud, local, or raw retrieval
- **Voice input** — local Whisper transcription
- **Observability** — LangFuse tracing on every query
- **Safety** — prompt injection detection
- **CI/CD** — tests run on every push via GitHub Actions
- **Streamlit Cloud deployable** — single-container pattern

---

*M.Sc. AI project · by [Prathamesh Shimpi](https://linkedin.com/in/prathamesh-shimpi0501)*
