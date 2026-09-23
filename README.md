# Medical Records RAG — Clinical Workstation

A multi-modal Retrieval-Augmented Generation (RAG) clinical workstation designed for oncology electronic health records (EHR). Features dual retrieval backends (GraphRAG on Neo4j + PageIndex hierarchical reasoning trees), strict evidence grounding, dual access roles, and an ephemeral upload sandbox.

---

## Key Features

- **Dual Clinical Access Roles**:
  - **Individual Patient View**: Streamlined patient self-lookup, focused clinical summaries, and timeline browsing.
  - **Group / Hospital Staff View**: Multi-patient cohort review, cross-patient comparisons, and multi-disciplinary tumor board workflows.
- **Dual Retrieval Backends**:
  - **GraphRAG (Neo4j)**: Deterministic, high-precision Cypher queries for laboratory trends, TNM staging, receptor biomarkers (ER/PR/HER2/Ki-67), and surgical/chemotherapy histories.
  - **PageIndex (Hierarchical Tree Traversal)**: LLM reasoning traversal over document summaries and page-level chunks for unstructured clinical narratives.
- **Strict Evidence Grounding & Confidence Badges**:
  - Every clinical claim is backed by explicit citations linking to source document page numbers and chunk IDs.
  - Interactive Evidence Inspector with color-coded confidence indicators (High $\ge 80\%$, Medium $50\text{--}79\%$, Low $< 50\%$).
- **Ephemeral Sandbox**:
  - Drag-and-drop clinical PDF upload allowing instant, isolated ingestion and querying without polluting permanent database records.
- **Hardware-Adaptive Inference**:
  - Automatic GPU detection (CUDA / llama.cpp GPU offloading) with crash-proof fallback to CPU.
  - Supports cloud LLMs (Google Gemini, Anthropic Claude, OpenAI) and local weights (MedGemma GGUF).

---

## Repository Structure

```text
├── app.py                      # Primary Streamlit web application & clinical dashboard
├── orchestrate.py              # Query classification, routing, and answer synthesis
├── patients.py                 # Registry and metadata for enrolled patients
├── evidence_store.py           # Structured evidence records and citation schemas
├── ingest.py                   # PDF rendering and page image extraction
├── ocr.py                      # Bilingual (English + Devanagari) OCR engine
├── split_reports.py            # Clinical report segmentation and anchor detection
├── chunk.py                    # Semantic chunking and evidence record packaging
├── normalize.py                # Oncology terminology normalization & regex extractors
├── llm_client.py               # Unified LLM wrapper (Gemini, Claude, OpenAI, Local GGUF)
├── device_utils.py             # Hardware auto-detection (CUDA/CPU) and runtime synchronization
├── temp_session.py             # Ephemeral single-report sandbox session manager
├── download_models.py          # Automated downloader for OCR and GGUF model weights
├── rebuild_backends.py         # Utility script to wipe and rebuild GraphRAG and PageIndex
├── run_pipeline_phase1.py      # Core data preparation pipeline runner
├── magic-pdf.json              # MinerU OCR configuration
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── .env.example                # Environment variables template
├── docs/                       # Specifications and architecture guides
│   ├── Medical_RAG_System_SRS_v2.md
│   └── BUILD_GUIDE_v2.md
├── scripts/                    # Maintenance and validation utilities
│   └── smoketests/             # Component smoke tests (Neo4j, LLM, OCR, etc.)
├── graph_backend/              # GraphRAG implementation
│   ├── build.py                # Neo4j knowledge graph construction
│   ├── retrieve.py             # Cypher query templates and fact extraction
│   └── schema.cypher           # Graph constraints and node schemas
├── pageindex_backend/          # PageIndex implementation
│   ├── build.py                # Hierarchical tree index builder
│   └── retrieve.py             # LLM reasoning tree traversal
├── data/                       # Data directory
│   ├── raw/                    # Source patient PDFs
│   └── normalization/          # Curated medical & oncology term dictionaries
└── tests/                      # Automated test suite (200 test cases)
```

---

## Prerequisites

- **Python**: 3.10 to 3.13
- **Neo4j**: Neo4j AuraDB (cloud) or a local Neo4j 5.x instance with Bolt enabled
- **Poppler** (for PDF rasterization via `pdf2image`):
  - **Ubuntu/Debian**: `sudo apt-get install -y poppler-utils`
  - **macOS**: `brew install poppler`
  - **Windows**: Install via `winget install -e --id osdn.poppler` or download binary releases and add to `PATH`.

---

## Quickstart

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/Param45/Medical-RAG.git
cd Medical-RAG

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the template file to `.env` and fill in your connection details:

```bash
cp .env.example .env
```

Key variables to configure:
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: Your Neo4j instance credentials.
- `LLM_PROVIDER`: Set to `gemini`, `anthropic`, `openai`, or `local`.
- `LLM_API_KEY`: Required for cloud LLM providers.

### 4. Launch the Application

```bash
streamlit run app.py
```

The workstation will be accessible at `http://localhost:8501`.

---

## Pipeline Administration

The application contains an integrated **Admin & Ingestion Control Panel** in the Streamlit UI. Alternatively, data operations can be triggered via CLI:

- **Run Full Phase 1 Ingestion Pipeline**:
  ```bash
  python run_pipeline_phase1.py
  ```
- **Wipe and Rebuild GraphRAG & PageIndex**:
  ```bash
  python rebuild_backends.py
  ```
- **Download Local Model Weights (Optional)**:
  ```bash
  python download_models.py --all
  ```

---

## Verification & Testing

The repository includes a comprehensive test suite covering all modules:

```bash
pytest tests/
```

To run a specific test suite:
```bash
pytest tests/test_orchestrate.py
pytest tests/test_graph_retrieve.py
pytest tests/test_pageindex_retrieve.py
```