# Software Requirements Specification (SRS) — Demo Scope
# Medical Records RAG System — Individual & Group, GraphRAG(Neo4j) & PageIndex

| Field | Value |
|---|---|
| Document Type | Software Requirements Specification (Demo/Prototype scope) |
| Domain | Medical Records Retrieval-Augmented Generation (RAG) |
| Intended Audience | Developers building a demoable, single-machine prototype |
| Source Data | Two sample hospital patient PDF bundles (scanned/handwritten + printed reports) |
| Status | Draft v2.0 — supersedes v1.0 (full production SRS); this version is intentionally descoped for a demo |

---

## 1. Purpose & Scope

### 1.1 Purpose
This document specifies a **demo-scale** Medical Records RAG system: a single Streamlit application, run locally, that lets a user simulate either (a) a **patient** looking up their own records, or (b) **hospital staff** looking up/comparing records across patients. It supports two interchangeable retrieval backends — **GraphRAG on Neo4j** and **PageIndex** — built from a shared ingestion/normalization pipeline.

This is **not** a production SRS. Everything that exists purely for production concerns (auth, encryption, containerization, async workers, audit trails, REST API surface, multi-tenant access control) has been deliberately removed per explicit client direction. Where a production concern was removed, it is called out so the reasoning is traceable ($3).

### 1.2 Scope

**In scope:**
- Ingesting the **two provided sample PDF bundles** (Patient A, Patient B) — small, known, fixed dataset.
- OCR of printed + handwritten pages (bilingual- English and Hindi).
- Lightweight medical terminology normalization of noisy OCR text.
- Building a knowledge graph in an **online Neo4j instance** (URI/credentials supplied via `.env`).
- Building a **PageIndex** tree per patient, with pure LLM reasoning-based traversal (no vector store).
- A single **Streamlit app** as the only user interface: role dropdown → patient dropdown → backend toggle → chat.
- Evidence (source page/snippet) shown for every answer, stored as **local JSON files**.

**Out of scope (explicitly removed for demo, see $3 for the "why"):**
- Any REST/API layer, OpenAPI schema, HTTP auth headers.
- Docker/containerization, docker-compose, multi-service orchestration.
- Async task queues (Celery/Redis) — ingestion is a simple sequential script.
- PostgreSQL or any relational DB — Neo4j (online) + local JSON files are the only stores.
- S3/object storage — everything lives on local disk under one project folder.
- Encrypted identifier storage, opaque patient IDs, role-based access-control policies — patients are picked from a **hardcoded dropdown** ("Patient A" / "Patient B"), and "staff" mode simply unlocks the ability to pick more than one patient or "All patients" in the same dropdown UI.
- Audit logging.
- Vector embeddings/vector DB for PageIndex.
- Clinical decision support — the system only reports what is documented, never advice/diagnosis, and disclaims accordingly.

### 1.3 Definitions & Acronyms

| Term | Meaning |
|---|---|
| RAG | Retrieval-Augmented Generation |
| OCR | Optical Character Recognition |
| GraphRAG | Retrieval pattern that queries a Neo4j knowledge graph (entities/relationships) instead of/in addition to plain text search |
| PageIndex | A hierarchical, tree-structured index of a document set; retrieval = LLM reasons over node summaries and chooses which branches to open, top-down (no embedding similarity involved) |
| Evidence | The exact source (file, page, raw OCR text) backing a generated statement, stored as a JSON record |
| Chunk | The smallest unit of extracted text used to build graph facts / PageIndex leaves |

### 1.4 Reference Dataset
Two patient bundles (multi-report PDFs), each containing: radiology reports (CECT/USG/mammogram/PET-CT/bone scan), histopathology/cytopathology reports, oncology flowsheets (hand-filled lab-trend tables), chemo drug-administration sheets, discharge summaries, surgical/anesthesia records, echocardiography reports, consent forms, and free-hand clinician notes. OCR quality will vary widely (typed text = high confidence, hand-filled tables = medium, cursive notes = low) — the pipeline must carry a confidence value through to the UI rather than hide it, because this is a medical-evidence system.

Because the dataset is fixed and small (2 patients), **no scalability, pagination, or concurrency handling is required**. Design for correctness and clarity, not throughput.

---

## 2. System Overview

### 2.1 Simplified Pipeline

```
[1] Local PDF folder (data/raw/patient_a.pdf, data/raw/patient_b.pdf)
        │
        ▼
[2] OCR (Chandra OCR 2 or MinerU — pick one via config, run sequentially per page)
        │  → per-page text + simple table detection + handwriting/confidence flag
        ▼
[3] Simple Report Splitting
        │  → break each bundle into known report sections (fixed heuristics tuned to the two sample PDFs)
        ▼
[4] Terminology Normalization (dictionary + light NER, LLM fallback only when needed)
        │  → normalized entities: diagnosis, drug, procedure, lab, date, staging, biomarker
        ▼
[5] Chunking (report/section-aware, simple rules)
        │  → each chunk also becomes one Evidence JSON record
        │
        ├───────────────────────────┬───────────────────────────┐
        ▼                           ▼
[6a] Build Neo4j graph        [6b] Build PageIndex tree (JSON file per patient,
     (online instance)              node summaries via LLM, no embeddings)
        │                           │
        └─────────────┬─────────────┘
                       ▼
[7] Streamlit query flow:
    role dropdown → patient dropdown (single/multi/all) → backend toggle (Graph/PageIndex)
    → retrieval (Cypher for Graph / LLM tree-walk for PageIndex)
    → answer generation with inline citations → evidence viewer
```

### 2.2 Shared ("Common") Components
Built once, used by both backends and both role modes, kept as plain Python modules (no service boundaries needed since there is no API layer):

1. `ingest.py` — PDF → page images → OCR
2. `normalize.py` — terminology normalization + light NER
3. `chunk.py` — chunking + evidence-record creation
4. `evidence_store.py` — read/write local JSON evidence files
5. `llm_client.py` — single wrapper around whichever LLM API is used (chat + optional embeddings, though embeddings are not required per $1.2)
6. `patients.py` — the fixed, hardcoded patient registry (2 entries) used to populate Streamlit dropdowns

---

## 3. Design Decisions (why each production concern was dropped)

| # | Decision | Reason |
|---|---|---|
| D1 | No REST API layer; Streamlit imports and calls the pipeline's Python functions directly (in-process). | Client instruction (10, 6): no need for API endpoints/auth/OpenAPI for a demo. Removes an entire service boundary and its testing/versioning overhead. |
| D2 | No Docker/compose. Developers run `pip install -r requirements.txt` and `streamlit run app.py` locally. | Client instruction (1). |
| D3 | No Celery/Redis. Ingestion is one sequential Python script/function that processes both PDFs one page at a time, in order. | Client instruction (3, 10): dataset is 2 files; async is unnecessary complexity. |
| D4 | Neo4j is a single **online/hosted instance**; credentials (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`) come from a `.env` file. No local Neo4j container, no schema-migration tooling beyond a single init script. | Client instruction (4). |
| D5 | No PostgreSQL. Anything that would have lived in Postgres (patient registry, evidence records, normalization dictionary, PageIndex tree) now lives as **local JSON/YAML files** under `data/`. | Client instruction (9, 5, 11). |
| D6 | No S3. All raw PDFs, page images, OCR output, evidence JSON, and PageIndex trees live under one local `data/` folder tree ($12). | Client instruction (5). |
| D7 | No opaque patient IDs, no encryption, no access-control policy engine. Patients are two fixed, named entries (`"Patient A"`, `"Patient B"`) selected via a Streamlit `st.selectbox`. "Individual" vs "Group" is itself a `st.selectbox` at the top of the app, not a login/session system. | Client instruction (7, 8). |
| D8 | Evidence is one JSON file per patient (`data/evidence/{patient_id}.json`), a flat list of evidence records keyed by `evidence_id`. No relational Evidence Store service. | Client instruction (9). |
| D9 | PageIndex retrieval uses **pure LLM reasoning traversal** over node summaries — no embeddings, no vector index, at either build time or query time. | Client instruction (12). |
| D10 | Report/section boundary detection is tuned specifically to the two sample PDFs' known structure (each embedded report starts with a recognizable hospital-letterhead/report-title line and the two documents' internal report list is enumerable up front — see $5.3) rather than a generic, robust document-segmentation model. | Client instruction (13): dataset is fixed and small; a general-purpose segmenter is over-engineering for a demo. |
| D11 | Configuration is a single `.env` file with only the credentials/settings actually needed to run the demo ($11), not a layered config-management system. | Client instruction (11). |
| D12 | No audit logging. The system may print/log to console for developer debugging, but there is no structured, queryable audit log store. | Client instruction (2). |
| D13 | Evidence-first behavior (every answer cites a source snippet, low-confidence OCR is visibly flagged) is **kept** even though everything else is simplified, because this was called out as a hard requirement independent of the "demo-only" framing. | Explicit client requirement carried over: "Evidence is extremely important for medical RAG." |
| D14 | The system still avoids giving medical advice/diagnosis and only reports what is documented, with a short disclaimer where relevant. | Patient-safety guardrail; low cost to keep even in a demo. |

---

## 4. Actors & Use Cases (unchanged in spirit, simplified selection mechanism)

### 4.1 Actors (selected via dropdown, not login)
- **Individual (Patient)** — after selecting "Individual" in the mode dropdown, a second dropdown lets them pick **one** patient (simulating "this is me"); all queries are scoped to that one patient only.
- **Group (Hospital Staff)** — after selecting "Group" in the mode dropdown, a second control (multi-select or "All patients" option) lets the user pick one, several, or all patients.

### 4.2 Use Cases

| ID | Mode | Example Query | Backend(s) to Support | Answer Must Include |
|---|---|---|---|---|
| UC-1 | Individual | "What is my health progress after multiple reports?" | Graph + PageIndex | Timeline of key metrics/impressions across dated reports, cited |
| UC-2 | Individual | "What diseases was I suffering from in the last year?" | Graph + PageIndex | List of diagnoses/findings with dates, cited |
| UC-3 | Individual | "What chemotherapy/medications have I received and when?" | Graph + PageIndex | Drug, dose, cycle, date, cited |
| UC-4 | Group (single patient selected) | "Give me the treatment details for Patient B" | Graph + PageIndex | Structured summary, cited |
| UC-5 | Group (two patients selected) | "Compare health recovery of Patient A vs Patient B" | Graph + PageIndex | Per-patient comparison, each claim cited and tagged with its patient |
| UC-6 | Group ("All patients" selected) | "Which patients have HER2-positive status documented?" | Graph + PageIndex | List of matching patients with cited evidence |

---

## 5. Functional Requirements — Common Core

### 5.1 Ingestion (`ingest.py`)
**FR-5.1.1** Read PDFs from `data/raw/` (two known files for the demo: `patient_a.pdf`, `patient_b.pdf`; the folder may hold more later, but nothing in the code should hardcode "exactly 2").
**FR-5.1.2** Render each page to an image (`pdf2image`/Poppler) at a fixed DPI (e.g., 200–300) and save under `data/pages/{patient_id}/page_{n}.png` — used for OCR input.
**FR-5.1.3** Assign `patient_id` from the filename (e.g., `patient_a.pdf` → `patient_id="patient_a"`), and a human-readable `display_label` (e.g., `"Patient A"`) from a small static mapping in `patients.py` — no identity extraction/parsing logic is required for the demo.
**FR-5.1.4** Ingestion runs **sequentially**, one page at a time, one patient at a time; log progress to console (e.g., `print`/`logging` at INFO level) — no worker/queue.

### 5.2 OCR (`ocr.py`)
**FR-5.2.1** Provide a small `OCRProvider` interface with two implementations — `ChandraOCRProvider` and `MinerUProvider` — selected by a single config value `OCR_ENGINE` ($11). Only **one** needs to be wired up and working for the demo to run end-to-end; the interface exists so the unused one can be swapped in later without touching any other module.
**FR-5.2.2** For each page image, produce a simple structure:
```json
{
  "page_number": 3,
  "raw_text": "...",
  "is_table": false,
  "is_handwritten": false,
  "confidence": 0.87
}
```
No bounding boxes are required unless the chosen OCR engine returns them for free — if it does, store them, but a page-level image reference is sufficient to satisfy the "evidence" requirement (the UI can show the full page image, not necessarily a cropped region).
**FR-5.2.3** For pages that are clearly a **tabular flowsheet** (visually dense grid of dates/numbers, as in the sample "Flowsheet Medical Oncology" pages), attempt a simple table read: split by column headers known from the sample layout (`HB/PCV`, `Platelates`, `WBC`, `ANC`, `ESR`, etc. — a fixed list matches the two sample PDFs) and rows by date. If table parsing fails or is unreliable, fall back to storing the page as unstructured `raw_text` with `is_table=true` and a lower confidence — do not build a general table-detection model.
**FR-5.2.4** Persist OCR output per page as JSON under `data/ocr/{patient_id}/page_{n}.json`. Re-running ingestion overwrites these files (idempotent by simple overwrite — no versioning needed).
**FR-5.2.5** Simple duplicate-block handling: if a page's `raw_text` is near-identical (e.g., normalized string equality or a basic similarity check) to another already-processed page for the same patient (the sample discharge summary appears OCR'd twice), keep the one with higher `confidence` and skip the other from downstream chunking — a simple in-memory set/check during the sequential loop is sufficient, no separate dedup service.

### 5.3 Report Splitting (`split_reports.py`)
**FR-5.3.1** Because the dataset is fixed and small, maintain a short, explicit list per patient of expected report types **in the order they appear**, driven by simple text-match anchors (e.g., a page containing "RADIOLOGY UNIT" starts a new `RADIOLOGY_*` report; a page containing "Histopathology Report" starts a `HISTOPATHOLOGY` report; a page containing "Flowsheet Medical Oncology" starts an `ONCOLOGY_FLOWSHEET` report; a page containing "DAYCARE DRUGS ADMINISTERED" starts a `CHEMO_DRUG_ADMIN_RECORD`; a page containing "DISCHARGE SUMMARY" starts a `DISCHARGE_SUMMARY`; a page containing "ANAESTHESIA RECORD"/"PAC" starts an `ANESTHESIA_RECORD`; a page containing "ECHOCARDIOGRAPHY REPORT" starts an `ECHOCARDIOGRAPHY` report). This anchor list is a short constant in code, not a trained classifier.
**FR-5.3.2** A new report starts on the first page matching an anchor and continues until the next anchor match (or end of document). Store the resulting `report_id`, `report_type`, `page_range` per detected report in `data/reports/{patient_id}.json`.
**FR-5.3.3** If a page matches no anchor, attach it to the currently open report as a continuation page (simplest reasonable default — no ML classification needed).
**FR-5.3.4** Report dates, where present in the OCR'd text near the top of a report, are extracted with a simple regex for common date formats seen in the corpus (`DD-MM-YYYY`, `DD/MM/YYYY`, `DD.MM.YY`, `DD-MON-YYYY`); if not confidently found, `report_date` is left `null` and the report is still usable (just not date-sortable).

### 5.4 Terminology Normalization (`normalize.py`)
**FR-5.4.1** Maintain one flat dictionary file `data/normalization/oncology_terms.json` mapping known abbreviations/shorthand seen in the corpus to a canonical term + entity type, e.g.:
```json
{
  "MRM": {"canonical": "Modified Radical Mastectomy", "type": "Procedure"},
  "EC": {"canonical": "Epirubicin + Cyclophosphamide", "type": "Regimen"},
  "5FU": {"canonical": "Fluorouracil", "type": "Medication"},
  "IDC": {"canonical": "Invasive Ductal Carcinoma", "type": "Diagnosis"},
  "NST": {"canonical": "No Special Type", "type": "Modifier"},
  "ER": {"canonical": "Estrogen Receptor", "type": "Biomarker"},
  "PR": {"canonical": "Progesterone Receptor", "type": "Biomarker"},
  "HER2": {"canonical": "Human Epidermal Growth Factor Receptor 2", "type": "Biomarker"}
}
```
This dictionary only needs to cover terms that actually appear in the two sample PDFs plus a small margin — it is not meant to be a general medical vocabulary.
**FR-5.4.2** Normalization order: (1) exact/fuzzy dictionary match → (2) a small set of regex parsers for structured values that recur in the corpus: TNM staging (`T\dN\dM\d` patterns), hormone-receptor scores (`ER\s*[+\-]?\s*\d/\d`, `Her-?2\s*neu\s*[:\-]?\s*\d\+`), Nottingham score triplets → (3) LLM fallback (via `llm_client.py`) **only** when neither of the above produces a confident match, explicitly prompting the LLM with the fixed dictionary as allowed options plus "or return null."
**FR-5.4.3** Each normalized entity keeps: `raw_text`, `normalized_term`, `entity_type`, `method` (`dictionary`|`regex`|`llm`|`none`), and inherits the OCR `confidence` of its source chunk (no separate normalization-confidence model is required for the demo — reuse OCR confidence, optionally reduced by a fixed penalty if `method="llm"`).
**FR-5.4.4** Dates are normalized to `YYYY-MM-DD` using the same regex approach as $5.3.4.

### 5.5 Chunking & Evidence (`chunk.py`, `evidence_store.py`)
**FR-5.5.1** Chunk per report ($5.3), then per page within a report (one chunk per page is an acceptable default for prose reports); for flowsheet/table reports, one chunk per row (per date) plus one chunk for the whole table.
**FR-5.5.2** Each chunk produces exactly one Evidence record:
```json
{
  "evidence_id": "patient_a__report_0007__page_3__chunk_0",
  "patient_id": "patient_a",
  "report_id": "report_0007",
  "report_type": "RADIOLOGY_PET_CT",
  "report_date": "2016-12-02",
  "page_number": 3,
  "raw_text": "...",
  "source_type": "typed | tabular_handwritten | cursive_handwritten",
  "confidence": 0.81,
  "page_image_path": "data/pages/patient_a/page_3.png"
}
```
**FR-5.5.3** All evidence records for a patient are appended to a single list and written to `data/evidence/{patient_id}.json`. This file is the **only** evidence store (D8) — both the Graph backend and PageIndex backend reference records in it by `evidence_id`; neither backend duplicates evidence text elsewhere except where noted in $6.1.3/$7.1.3 (a copy of `raw_text` is kept inline on the PageIndex leaf node purely for convenience/readability of the tree file, but `evidence_id` remains the source of truth).
**FR-5.5.4** `source_type` classification is a simple heuristic (from the OCR flags in $5.2.2: `is_table` → `tabular_handwritten` if also flagged handwritten, else the page's `is_handwritten` flag directly maps to `cursive_handwritten` vs `typed`) — no separate handwriting-vs-print ML classifier needs to be trained; rely on whatever the chosen OCR engine reports, defaulting to `typed` if the engine gives no signal.

### 5.6 LLM Client (`llm_client.py`)
**FR-5.6.1** One thin wrapper function `chat(messages, system=None) -> str` used by: normalization fallback ($5.4.2), GraphRAG relation extraction ($6.1), PageIndex summarization/traversal ($7.1/$7.2), and answer generation ($8.2). Provider/model/API key read from `.env` ($11); no provider-abstraction layer beyond this one function is required.
**FR-5.6.2** No embeddings function is required (PageIndex is pure LLM traversal per D9, and GraphRAG retrieval — $6.2 — uses Cypher, not vector search, per this simplified scope).

### 5.7 Patient Registry (`patients.py`)
**FR-5.7.1** A static list/dict, e.g.:
```python
PATIENTS = {
    "patient_a": "Patient A",
    "patient_b": "Patient B",
}
```
used to populate every Streamlit dropdown. Adding a third demo patient means adding one line here plus a PDF in `data/raw/` — no other change required.

---

## 6. Functional Requirements — Backend A: GraphRAG + Neo4j (online instance)

### 6.1 Graph Build (`graph_backend/build.py`)
**FR-6.1.1** Connect to the online Neo4j instance using `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` from `.env` (official `neo4j` Python driver).
**FR-6.1.2** Run a one-time schema-init step (`graph_backend/schema.cypher`, executed via a small Python function) creating uniqueness constraints on `Patient.patient_id`, `Diagnosis.canonical_name`, `Medication.canonical_name`, `LabTest.canonical_name` — no full production migration tooling, just one script run once (or safely re-run, since `CREATE CONSTRAINT IF NOT EXISTS` is idempotent).
**FR-6.1.3** For each chunk + its normalized entities ($5.4), call the LLM ($5.6) with a fixed prompt template to extract simple `(subject, relation, object)` triples using only the node/relationship types in $12.1 (a **small, fixed schema** — do not let the LLM invent new relationship types for this demo).
**FR-6.1.4** Write nodes/relationships with Cypher `MERGE` keyed on `patient_id` + canonical name/date, so re-running the build on the same two PDFs does not duplicate nodes. Every relationship stores `evidence_id` (string, referencing $5.5's JSON file) and `confidence` as properties.
**FR-6.1.5** Build runs **sequentially** over the two patients' chunks (no batching/async needed at this scale).

### 6.2 Graph Retrieval
**FR-6.2.1** A small library of **parameterized Cypher templates** covers the demo's known question shapes (lab-value timeline, diagnosis list, medication/regimen history, staging/biomarker lookup) — see $12.1 for the node/relationship shapes each template targets.
**FR-6.2.2** For questions that don't match a template, fall back to a single LLM-generated Cypher query, constrained by a short schema description in the prompt (labels/relationship types from $12.1 only) and always **parameterized by the selected `patient_id`(s)** from the Streamlit dropdown ($7 of the UI section, $9) — the query text itself is built with the patient filter already applied by the calling Python code (not left to the LLM to remember to add), so scoping cannot be "forgotten."
**FR-6.2.3** For **Individual** mode, the Cypher `WHERE patient_id = $selected_patient` clause uses the one patient chosen in the dropdown.
**FR-6.2.4** For **Group** mode with multiple/())"All patients"(( selected, run the same query once per selected `patient_id` (simple Python loop — no cross-patient JOIN), tag each result set with its `patient_id`, and merge in Python before handing to generation ($8.2) — this keeps per-patient attribution unambiguous for comparison answers (UC-5, UC-6).

---

## 7. Functional Requirements — Backend B: PageIndex (pure LLM traversal, no vector store)

### 7.1 Tree Build (`pageindex_backend/build.py`)
**FR-7.1.1** Build one tree **per patient**, mirroring: `Patient → Report → Page/Section → Chunk (leaf)`, matching the simplified report/page-level chunking from $5.5.1 (no need for a deeper section-within-page hierarchy given the corpus size).
**FR-7.1.2** For every non-leaf node, call the LLM ($5.6) once to produce a short (2–4 sentence) summary of what's beneath it (e.g., a summary of one `Report` node from its child page chunks' text). Store this summary as a plain string field on the node.
**FR-7.1.3** Leaf nodes store `raw_text` and `evidence_id` (pointing into `data/evidence/{patient_id}.json`, $5.5.3).
**FR-7.1.4** Persist the whole tree as one JSON file per patient: `data/pageindex/{patient_id}.json`. No database, no embeddings, no vector index — this file **is** the index (D5, D9).

### 7.2 Tree Retrieval (pure LLM reasoning)
**FR-7.2.1** Load the JSON tree file for the selected patient(s) directly from disk (this load **is** the scope boundary for Individual mode — the code simply never opens another patient's file).
**FR-7.2.2** Traversal algorithm: starting at the root, present the LLM with the question + the summaries of the current node's children, and ask it to choose which child(ren) to open next (or "none, stop here"); recurse until reaching leaves or a small fixed depth/breadth limit (e.g., max depth 3, max 6 nodes expanded) — implemented as a simple recursive Python function, no separate ranking/embedding step.
**FR-7.2.3** For timeline-style questions (UC-1), allow the traversal step to select **multiple** sibling nodes (e.g., all `Report` nodes whose summary mentions the relevant lab/finding) rather than only the single best match.
**FR-7.2.4** For Group mode with multiple patients selected, run $7.2.1–$7.2.3 independently per patient's tree file and tag each leaf found with its `patient_id` before merging in Python (mirrors $6.2.4's approach, keeping the two backends' Group behavior symmetric).

---

## 8. Retrieval Orchestration & Answer Generation (in-process, no API layer)

### 8.1 Orchestration (`orchestrate.py`)
**FR-8.1.1** One Python function, called directly by the Streamlit app:
```python
def answer_question(question: str, mode: str, selected_patients: list[str], backend: str) -> dict:
    ...
    return {"answer": str, "citations": [evidence_id, ...], "backend_used": backend}
```
`mode` is `"individual"` or `"group"`; `selected_patients` is always a list (length 1 for Individual, 1+ or all `PATIENTS.keys()` for Group's "All patients" option — the Streamlit layer resolves "All" into the full list before calling this function, so `orchestrate.py` never needs to know about the UI concept of "All").
**FR-8.1.2** Dispatch to $6.2 (if `backend == "graph"`) or $7.2 (if `backend == "pageindex"`), producing a common intermediate `facts` list: `[{"text": ..., "patient_id": ..., "evidence_id": ...}, ...]`.

### 8.2 Answer Generation
**FR-8.2.1** Build one LLM prompt from the collected `facts` + the original question, instructing the model to: answer only from the given facts; cite each claim with a bracketed marker referencing an `evidence_id`; say "not documented in the available records" when facts are insufficient; and, if the question reads as asking for medical advice/diagnosis rather than "what is documented," answer only with documented facts and add one short disclaimer line (D14).
**FR-8.2.2** After generation, a simple grounding check: any citation marker in the LLM's output that doesn't match an `evidence_id` present in `facts` is dropped from the final citation list shown in the UI (basic hallucination guard, kept from the original scope since it's cheap and directly supports the evidence requirement).
**FR-8.2.3** For Group comparison questions (UC-5), the prompt explicitly instructs the LLM to structure the answer per patient (e.g., a short paragraph or table row per patient) rather than blending facts from different patients into one uncited sentence.

---

## 9. Streamlit UI (the only user-facing surface)

### 9.1 Layout
**FR-9.1.1** Sidebar (top to bottom):
1. **Mode dropdown**: `Individual` / `Group`.
2. **Patient selector**:
   - If `Individual`: a single-select dropdown of `PATIENTS` ($5.7) — simulates "I am this patient."
   - If `Group`: a multi-select dropdown of `PATIENTS`, plus a checkbox/option for **"All patients."**
3. **Backend toggle**: `GraphRAG (Neo4j)` / `PageIndex`.
4. *(Optional, nice-to-have, not required)* a "Compare both backends" checkbox that runs the same question through both and shows them side by side.

**FR-9.1.2** Main panel: a simple chat input + chat history (`st.chat_input`/`st.chat_message`), calling `orchestrate.answer_question(...)` ($8.1) on submit.

**FR-9.1.3** Each answer renders with its citation markers as clickable elements (Streamlit buttons/expanders keyed by `evidence_id`); clicking shows an **Evidence panel**: the page image (`st.image` on `page_image_path`), the raw OCR text (`st.code`/`st.text`), report type/date, and a confidence badge (simple colored `st.markdown`/`st.progress`: green ≥0.8, yellow 0.5–0.8, red <0.5).

**FR-9.1.4** No login screen, no session persistence beyond the current browser session's `st.session_state` (used only to keep chat history) — no accounts, no database of users.

### 9.2 Ingestion Trigger (optional simple admin affordance)
**FR-9.2.1** A small "Rebuild index" button (sidebar or separate tab) that calls the ingestion → normalize → chunk → build-graph/build-pageindex functions in sequence for both demo PDFs, with a `st.spinner`/`st.progress` showing which stage is running — this replaces any separate ingestion service; it's just a button that calls the same Python functions a CLI script would call.

---

## 10. Non-Functional Requirements (trimmed to what still matters for a demo)

- **NFR-1 (Evidence, kept):** Every answer must show at least one citation with confidence; low-confidence (handwritten/cursive) evidence must be visually distinguishable in the UI, never presented identically to high-confidence typed text.
- **NFR-2 (No advice):** The system never outputs a treatment recommendation; it reports only what's documented, per D14.
- **NFR-3 (Reproducibility):** Re-running "Rebuild index" ($9.2) on the same two PDFs produces the same graph node/relationship counts and the same PageIndex tree shape (idempotent via `MERGE` in Neo4j and file-overwrite for JSON/tree files) — no duplication on repeat runs.
- **NFR-4 (Simplicity over performance):** No specific latency/throughput target is mandated; the demo dataset is 2 patients, so correctness and clarity of the demo take priority over speed. A generation call finishing in "a reasonable number of seconds for a live demo" is sufficient — no formal SLA.
- **NFR-5 (Local-only):** The system must run fully on one developer machine with only two external dependencies: the online Neo4j instance and the chosen LLM/OCR APIs. No other network services.

---

## 11. Configuration

A single `.env` file, loaded once at startup (e.g., via `python-dotenv`):

```
# LLM
LLM_PROVIDER=<provider name>
LLM_API_KEY=<key>
LLM_MODEL=<model name>

# OCR
OCR_ENGINE=chandra   # or: mineru
OCR_API_KEY=<key if the chosen OCR engine needs one>

# Neo4j (online instance, credentials supplied externally)
NEO4J_URI=<uri>
NEO4J_USERNAME=<username>
NEO4J_PASSWORD=<password>
```

No other configuration layer (no per-environment config classes, no secrets manager, no feature flags) is required for the demo.

---

## 12. Data Model

### 12.1 Neo4j Graph Schema (kept small and fixed — this is the schema the LLM extraction prompt in $6.1.3 and the Cypher templates in $6.2.1 are both constrained to)

**Node labels:**
`Patient {patient_id, display_label}`, `Report {report_id, patient_id, report_type, report_date}`, `Diagnosis {canonical_name}`, `Procedure {canonical_name}`, `Medication {canonical_name}`, `Regimen {canonical_name}`, `LabTest {canonical_name}`, `LabResult {value, unit, date}`, `Staging {t, n, m, date}`, `Biomarker {marker, value, date}`.

**Relationships (each carries `evidence_id`, `confidence` as properties):**
- `(Patient)-[:HAS_REPORT]->(Report)`
- `(Report)-[:STATES_DIAGNOSIS]->(Diagnosis)`
- `(Patient)-[:UNDERWENT {date}]->(Procedure)`
- `(Report)-[:ADMINISTERED {dose, cycle}]->(Medication)` / `(Report)-[:ADMINISTERED]->(Regimen)`
- `(Regimen)-[:CONTAINS]->(Medication)`
- `(Report)-[:HAS_RESULT]->(LabResult)-[:OF_TEST]->(LabTest)`
- `(Report)-[:HAS_STAGING]->(Staging)`
- `(Report)-[:HAS_BIOMARKER]->(Biomarker)`

This is intentionally the **entire** schema for the demo — no `Provider`/`Facility`/`BodySite`/`Finding` nodes unless a specific demo question needs them; add only if a required use case in $4.2 cannot otherwise be answered.

### 12.2 PageIndex Tree JSON (per patient, `data/pageindex/{patient_id}.json`)
```json
{
  "patient_id": "patient_a",
  "root": {
    "node_id": "root",
    "summary": "...",
    "children": [
      {
        "node_id": "report_0007",
        "node_type": "Report",
        "report_type": "RADIOLOGY_PET_CT",
        "report_date": "2016-12-02",
        "summary": "...",
        "children": [
          {
            "node_id": "report_0007_page_3",
            "node_type": "Page",
            "raw_text": "...",
            "evidence_id": "patient_a__report_0007__page_3__chunk_0"
          }
        ]
      }
    ]
  }
}
```

### 12.3 Local File Layout (replaces all databases except Neo4j)
```
data/
├── raw/
│   ├── patient_a.pdf
│   └── patient_b.pdf
├── pages/
│   ├── patient_a/page_1.png ...
│   └── patient_b/page_1.png ...
├── ocr/
│   ├── patient_a/page_1.json ...
│   └── patient_b/page_1.json ...
├── reports/
│   ├── patient_a.json          # list of detected reports + page ranges
│   └── patient_b.json
├── normalization/
│   └── oncology_terms.json     # the shared abbreviation dictionary ($5.4.1)
├── evidence/
│   ├── patient_a.json          # flat list of EvidenceRecord ($5.5.2)
│   └── patient_b.json
└── pageindex/
    ├── patient_a.json
    └── patient_b.json
```
Neo4j (online) is the only non-file store, holding the graph described in $12.1.

---

## 13. Pipeline Walkthrough (developer-facing, sequential)

1. Put `patient_a.pdf`, `patient_b.pdf` in `data/raw/`.
2. Run `python ingest.py` (or click "Rebuild index" in Streamlit, $9.2): for each PDF → render pages ($5.1) → OCR each page in order ($5.2) → detect reports via anchor list ($5.3) → normalize terms per chunk ($5.4) → write chunks + evidence JSON ($5.5).
3. Run `python graph_backend/build.py`: connect to the online Neo4j instance, extract triples per chunk via LLM, `MERGE` into the graph ($6.1).
4. Run `python pageindex_backend/build.py`: build the per-patient tree JSON with LLM-generated summaries ($7.1).
5. Launch `streamlit run app.py`: pick mode → pick patient(s) → pick backend → ask a question → `orchestrate.answer_question(...)` runs retrieval ($6.2 or $7.2) → generation ($8.2) → Streamlit renders the answer + evidence panel ($9.1).

Steps 2–4 can be combined behind the single "Rebuild index" button ($9.2) for a one-click demo reset.

---

## 14. Testing (lightweight, matching demo scope)

- **Manual smoke test script** (`tests/manual_checklist.md`): run all six use cases ($4.2) against both backends and confirm each answer has at least one citation and the citation resolves to a real evidence record.
- **A handful of `pytest` unit tests** are still worth keeping (cheap, high value) for the parts most likely to silently break:
  - Dictionary normalization: known abbreviations map correctly ($5.4.1 examples).
  - Report splitting: given the two known PDFs' OCR output, the expected number/order of reports is detected ($5.3).
  - Evidence write/read round-trip: a chunk always produces one resolvable evidence JSON entry ($5.5.2).
- No integration test harness, no CI pipeline, no load testing — a demo does not need them.

---

## 15. Project Structure (flat, no service boundaries)

```
medical-rag-demo/
├── app.py                       # Streamlit entry point ($9)
├── ingest.py                    # $5.1
├── ocr.py                       # $5.2 (OCRProvider + both implementations)
├── split_reports.py             # $5.3
├── normalize.py                 # $5.4
├── chunk.py                     # $5.5
├── evidence_store.py            # $5.5 read/write helpers
├── llm_client.py                # $5.6
├── patients.py                  # $5.7
├── orchestrate.py               # $8.1
├── graph_backend/
│   ├── schema.cypher            # $6.1.2
│   ├── build.py                 # $6.1
│   └── retrieve.py              # $6.2
├── pageindex_backend/
│   ├── build.py                 # $7.1
│   └── retrieve.py              # $7.2
├── data/                        # $12.3
├── tests/
│   └── ... ($14)
├── .env                         # $11 (not committed)
├── .env.example
└── requirements.txt
```

---

## 16. Traceability Matrix (client simplification instruction → where addressed)

| # | Client Instruction | Addressed In |
|---|---|---|
| 1 | No Docker | $1.2, $15 (flat structure, `pip`/`streamlit run` only) |
| 2 | No audit logs | $1.2, D12 |
| 3 | No Redis/Celery | $1.2, D3, $5.1.4 |
| 4 | Online Neo4j instance | D4, $6.1.1, $11 |
| 5 | Local filesystem, no S3 | D6, $12.3 |
| 6 | No API endpoints/auth/OpenAPI | D1, $8 (in-process orchestration) |
| 7 | Dropdown-based mode/patient selection | D7, $9.1.1 |
| 8 | No encrypted IDs/opaque IDs/access policy | D7, $5.7 |
| 9 | Evidence as local JSON, no Postgres | D8, D5, $5.5.3, $12.3 |
| 10 | Sequential ingestion, no async workers | D3, $5.1.4, $6.1.5 |
| 11 | Minimal config | D11, $11 |
| 12 | No vector store for PageIndex | D9, $7.2 |
| 13 | Simplified chunking/report-boundary detection | D10, $5.3 |

*(Original functional requirements — Individual RAG, Group RAG, GraphRAG+Neo4j, PageIndex, evidence-first answers, terminology normalization, shared/common components, Streamlit UI — remain fully covered, just implemented at demo scale throughout $4–$9.)*

---

*End of SRS (Demo Scope, v2.0).*
