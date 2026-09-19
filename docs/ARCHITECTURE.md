# NEXUS System Architecture

**Version**: 1.0.0  
**Platform**: Snapdragon X Elite · Windows 11 ARM64  
**Classification**: Offline Multimodal Work Agent

---

## 1. System Overview

NEXUS is a locally-executed, air-gapped, multimodal AI work agent. It accepts documents, images, voice, and structured data as inputs; orchestrates a deterministic multi-step cognitive pipeline; and produces grounded, evidence-backed work products — entirely without network access.

### 1.1 Design Principles

1. **Offline-First**: Every component operates without internet connectivity. The `LocalNetworkGuard` enforces this at the socket layer.
2. **NPU-Primary**: All heavy matrix operations (embedding, vision, ASR, LLM) are dispatched to the Qualcomm Hexagon NPU via ONNX Runtime QNN Execution Provider.
3. **Defense-in-Depth**: Documents are untrusted data. The system enforces strict sandboxing at ingestion, planning, tool execution, and network layers.
4. **Grounded Reasoning**: All agent conclusions are traceable to specific document chunks with citation IDs. The system never generates unsupported claims.
5. **Transparent Reporting**: OCR, retrieval, and verification results are tagged `OBSERVED` vs. `INFERRED`. No hallucination laundering.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                                │
│  Web App (HTML/CSS/JS)  ─────  REST API (FastAPI, port 8000)        │
│  Voice Input  ──────────────── /api/speech/transcribe               │
│  Document Upload  ──────────── /api/documents/                      │
│  Task Goal  ────────────────── /api/agent/run                       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                     INGESTION SUBSYSTEM                              │
│  DocumentValidator  ──  Magic byte check, path traversal block      │
│  FileExtractors     ──  PDF (PyPDF2), DOCX (python-docx), TXT, JSON │
│  ChunkProcessor     ──  Sentence-aware chunking (512 token max)     │
│  VectorIndexer      ──  all-MiniLM-L6-v2 → SQLite vector store      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                     AGENT ORCHESTRATOR                               │
│  AgentPlanner       ──  Llama-3.2-1B → DAG task plan (Kahn check)   │
│  AgentRunner        ──  Topological task execution loop              │
│  ToolRegistry       ──  7 whitelisted deterministic tools only       │
│  StateManager       ──  SQLite task/evidence/result persistence      │
└──────────┬───────────────────────────────────────┬───────────────────┘
           │                                       │
┌──────────▼───────────────┐         ┌─────────────▼──────────────────┐
│    NPU INFERENCE LAYER   │         │    VERIFICATION SUBSYSTEM      │
│  [Hexagon HTP via QNN]   │         │  EvidenceBuilder  ─  Citations │
│                          │         │  ClaimVerifier    ─  Grounding │
│  ASR:  Whisper-Small     │         │  ReportGenerator  ─  Export    │
│  OCR:  EasyOCR (CRAFT)   │         └────────────────────────────────┘
│  Vision: CLIP ViT-B/32   │
│  Embed: all-MiniLM-L6-v2 │
│  LLM:  Llama-3.2-1B-Inst │
└──────────────────────────┘
```

---

## 3. Component Descriptions

### 3.1 Frontend (Single-Page Application)

- **Technology**: Vanilla HTML5, CSS3, JavaScript (no framework dependencies)
- **Screens**: Home, New Task, Agent Workspace, Evidence, Documents, Task History, Performance, Privacy, Settings
- **Communication**: REST API calls to `http://127.0.0.1:8000`
- **Voice Integration**: Web Speech API for microphone capture; transcription via `/api/speech/transcribe`
- **Design**: Dark-mode glassmorphism UI; accessible typography; keyboard navigation

### 3.2 Backend (FastAPI Application)

**Entry Point**: `backend/main.py`  
**Server**: Uvicorn ASGI with lifespan management

| Router Module | Prefix | Responsibility |
|---|---|---|
| `routes_documents.py` | `/api/documents` | Upload, list, retrieve, delete |
| `routes_agent.py` | `/api/agent` | Submit goals, query task status |
| `routes_llm.py` | `/api/llm` | Direct LLM completions |
| `routes_knowledge.py` | `/api/knowledge` | Semantic search, chunk retrieval |
| `routes_speech.py` | `/api/speech` | ASR transcription |
| `routes_vision.py` | `/api/vision` | Image analysis, CLIP embeddings |
| `routes_tools.py` | `/api/tools` | List registered tools |
| `routes_verification.py` | `/api/verification` | Claim verification |
| `routes_benchmarks.py` | `/api/benchmarks` | Run performance benchmarks |
| `routes_network.py` | `/api/network` | Network guard status |
| `routes_demo.py` | `/api/demo` | Competition demo workflow |
| `health.py` | `/health` | System health check |

### 3.3 Agent Subsystem (`backend/agent/`)

**Planner** (`planner.py`):
- Receives user goal as natural language text
- Constructs structured system + user prompt with security fencing
- Calls Llama-3.2-1B-Instruct (local, via `backend/interfaces/llm.py`)
- Parses JSON `StructuredPlan` with `tasks[]` and `dependencies[]`
- Validates DAG using Kahn's topological sort — rejects cycles immediately
- Returns `StructuredPlan` (Pydantic model with `created_at` ISO timestamp)

**Runner** (`runner.py`):
- Executes tasks in dependency order (topological sort)
- Dispatches each task to its registered tool
- Writes task results to SQLite (`TaskResult` table)
- Accumulates evidence entries per task
- Returns `AgentResult` with full evidence chain

### 3.4 Ingestion Subsystem (`backend/ingestion/`)

**Validator** (`validator.py`):
- `validate_filename()` — Null byte, path traversal, reserved names, length limit
- `validate_file_content()` — Magic byte signature verification per MIME type
- `validate_file_size()` — Hard 50MB cap
- Supported types: `.pdf`, `.docx`, `.txt`, `.png`, `.jpg`, `.jpeg`, `.json`

**Extractors** (`extractors.py`):
- `PDFExtractor` — PyPDF2 page-by-page text extraction with corrupt-stream guard
- `DOCXExtractor` — python-docx paragraph iteration
- `TXTExtractor` — UTF-8 / Latin-1 fallback plain text
- `ImageExtractor` — Pillow metadata (actual OCR via OCR interface)

**Chunker** (`chunker.py`):
- Sentence-boundary-aware splitting
- Maximum 512 tokens per chunk with 50-token overlap

### 3.5 Knowledge Subsystem (`backend/knowledge/`)

**VectorStore** (`store.py`):
- Embeddings stored as binary blobs in SQLite (`chunks` table)
- Retrieval: dot-product cosine similarity over NumPy arrays
- Returns top-k chunks with similarity scores and chunk IDs

**Retriever** (`retriever.py`):
- Accepts a query string
- Embeds via `all-MiniLM-L6-v2`
- Returns ranked citation list with source document, page, and text

### 3.6 NPU Inference Layer (`backend/interfaces/`)

All AI interfaces implement a provider adapter pattern:

```python
class InferenceAdapter:
    def __init__(self):
        self.provider = detect_execution_provider()
        # Returns "QNNExecutionProvider" on Snapdragon ARM64
        # Returns "CPUExecutionProvider" on x64 development host
```

- **`llm.py`**: `onnxruntime-genai` GenAI session; `QNNExecutionProvider` on ARM64
- **`embedding.py`**: ORT InferenceSession with MiniLM ONNX; QNN on ARM64
- **`ocr.py`**: EasyOCR CRAFT+CRNN; QNN on ARM64 (pending DLC compilation)
- **`vision.py`**: CLIP ViT-B/32 via ORT; QNN on ARM64
- **`asr.py`**: Whisper-Small via ORT; QNN on ARM64

### 3.7 Verification Subsystem (`backend/verification/`)

**EvidenceBuilder**:
- Links each retrieved chunk to the specific task that retrieved it
- Records `(chunk_id, document_id, similarity_score, text_snippet)` per evidence item

**ClaimVerifier**:
- Receives a list of claims and a list of supporting evidence chunks
- For each claim: checks if any evidence chunk contradicts or supports it
- Returns `VerificationResult` with `supported`, `contradicted`, `confidence`

**ReportGenerator**:
- Produces structured Markdown action report
- Sections: Executive Summary, Findings, Evidence, Verification, Action Items
- All findings include citation IDs traceable to source documents

### 3.8 Security Layer

See [`docs/SECURITY.md`](SECURITY.md) for full threat model.

**Summary**:
- `DocumentValidator` — Input validation & magic byte checks
- `LocalNetworkGuard` — Socket-layer air-gap enforcement
- Prompt fencing — `<UNTRUSTED_DOCUMENT_CONTEXT>` tags + Rule 8
- Tool allowlist — Exactly 7 deterministic tools (no shell, no eval)
- Path traversal guards — Strict `data/` directory confinement
- DAG cycle detection — Kahn's algorithm on every plan

---

## 4. Data Flow: End-to-End Agent Workflow

```
1. USER submits goal text via UI or CLI
2. AgentPlanner prompts Llama-3.2-1B → StructuredPlan (JSON DAG)
3. AgentRunner iterates tasks in topological order:
   a. list_documents → Discovers ingested files
   b. read_document  → Loads document text (path-sandboxed)
   c. search_knowledge → Semantic retrieval (top-3 chunks)
   d. analyze_image  → CLIP vision analysis
   e. run_ocr        → Text extraction from images
   f. create_report  → Assembles evidence + findings
   g. export_report  → Writes Markdown to data/reports/ (approval-gated)
4. VerificationLayer checks all claims against retrieved evidence
5. ReportGenerator produces final action report
6. UI renders plan, evidence, verification, and report
```

---

## 5. Database Schema

SQLite file: `data/nexus.db`

| Table | Columns | Purpose |
|---|---|---|
| `documents` | id, filename, file_type, size_bytes, ingested_at, chunk_count | Document registry |
| `chunks` | id, document_id, chunk_index, text, embedding (BLOB), token_count | Vector store |
| `task_runs` | id, goal, status, created_at, completed_at | Agent run registry |
| `task_results` | id, run_id, task_id, tool_name, status, result_json, error | Task execution log |
| `evidence_items` | id, run_id, task_id, chunk_id, similarity, snippet | Evidence citations |

---

## 6. Execution Provider Strategy

```python
def detect_execution_provider() -> str:
    """
    Returns the best available ONNX Runtime execution provider.
    Priority: QNNExecutionProvider > CPUExecutionProvider
    """
    import platform
    providers = ort.get_available_providers()
    if "QNNExecutionProvider" in providers:
        return "QNNExecutionProvider"   # Snapdragon X Elite Hexagon NPU
    return "CPUExecutionProvider"       # Development host fallback
```

This single function governs all model loading across the entire pipeline, ensuring transparent NPU acceleration on target hardware with no code changes.

---

## 7. Configuration

All configuration is managed through `backend/config.py` (Pydantic Settings):

| Setting | Default | Description |
|---|---|---|
| `APP_NAME` | `NEXUS` | Application name |
| `VERSION` | `1.0.0` | Version string |
| `OFFLINE_MODE` | `True` | Enforce air-gap |
| `TARGET_PLATFORM` | `snapdragon-x-elite` | Target hardware |
| `MAX_FILE_SIZE_MB` | `50` | Document size ceiling |
| `LLM_TIMEOUT_SECONDS` | `60` | Local LLM response timeout |
| `VECTOR_STORE_TOP_K` | `5` | Default retrieval depth |
| `DATA_DIR` | `data/` | Runtime data root |
| `DEMO_MODE` | `False` | Enable demo dataset |
