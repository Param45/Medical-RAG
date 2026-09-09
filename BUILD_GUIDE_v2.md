# Medical Records RAG (Demo) — BUILD_GUIDE
## Step-by-Step Implementation Tasks for Your Coding Agent

> **How to use this file:**
> Each task below is a self-contained instruction block. To execute one, tell your coding agent:
> `"Implement Task X.X as mentioned in the BUILD_GUIDE."`
> Tasks are numbered in dependency order — **do not skip ahead**; later tasks assume earlier ones exist.
> After each task: run/inspect the output before moving to the next task.
>
> **Every task is tagged with who performs it:**
> - 🤖 **AGENT TASK** — your coding agent can do this end-to-end (write code, run local commands, run local models, run tests).
> - 🧑‍💻 **YOUR TASK (manual)** — requires you personally: signing up for an external service, obtaining/pasting a secret, or making a judgment call your agent cannot make on your behalf (e.g., "does this Hindi OCR text look right?").
> - 🤝 **MIXED** — the agent writes/runs the code, but a step inside the task needs your input or your eyes (clearly marked inline with 🧑‍💻).
>
> This guide assumes the entire system runs **locally** except the LLM API calls (per project scope). MinerU (OCR) also runs **locally on your machine**.

---

## CONTEXT DOCUMENT

Your coding agent must read the SRS before writing any code. At the top of **every** task prompt, include:

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md in this repo before writing any code.
This SRS is the source of truth for schemas, file layout, function signatures, and scope.
Do not add features, services, databases, or infrastructure that the SRS explicitly excludes
(see SRS $1.2 "Out of scope" and $3 "Design Decisions").
```

(This instruction is already embedded in every task prompt below — you don't need to add it again.)

---

---

# PHASE 0 — ENVIRONMENT & MANUAL PREREQUISITES
## Goal: A machine that can run OCR, talk to Neo4j, and talk to an LLM API — before any project code is written.

---

## TASK 0.1 — 🧑‍💻 YOUR TASK (manual): Obtain External Credentials

You must personally do the following **before Phase 1 can be tested end-to-end** (the agent cannot sign up for services or generate secrets on your behalf):

1. **LLM API key** — Sign up with whichever LLM provider you intend to use (e.g., Anthropic Claude API, OpenAI). Generate an API key. Keep it handy — you will paste it into `.env` in Task 0.5.
2. **Online Neo4j instance** — Create a free/hosted Neo4j instance (e.g., Neo4j AuraDB Free tier, or any Neo4j server you have network access to). Note down:
   - `NEO4J_URI` (e.g., `neo4j+s://xxxxxx.databases.neo4j.io`)
   - `NEO4J_USERNAME` (usually `neo4j`)
   - `NEO4J_PASSWORD`
3. Confirm you can reach both from your local machine (e.g., log into the Neo4j browser console once to confirm the instance is live).

**Nothing to build yet — this is account/credential setup only.**

---

## TASK 0.2 — 🧑‍💻 YOUR TASK (manual): Prepare the Source PDFs

1. Rename/copy your two sample patient PDF bundles to:
   - `patient_a.pdf`
   - `patient_b.pdf`
2. You will place these into `data/raw/` once the folder exists (Task 0.3 creates it). If you'd rather do it now, that's fine — just make sure they exist there before Task 1.4 (first OCR run).

> Note: both PDFs contain a mix of **English and Hindi (Devanagari script)** pages/sections (e.g., consent forms). Keep this in mind — it drives OCR language configuration in Task 0.4 and normalization in Task 1.6. No action needed from you here beyond having the files ready.

---

## TASK 0.3 — 🤖 AGENT TASK: System Package & Python Environment Setup

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md before writing any code.

Task: Set up the local development environment for the Medical RAG demo project.

1. Create a Python virtual environment (Python 3.10 or 3.11) named `.venv` in the project root.

2. Create `requirements.txt` with (at minimum) these dependency groups — pin reasonable versions:
   - PDF/image handling: pypdf, pdf2image, Pillow
   - Neo4j driver: neo4j
   - Env/config: python-dotenv
   - Streamlit: streamlit
   - Testing: pytest
   - MinerU OCR package (see Task 0.4 for exact package name/setup — install it here as well)
   - Whatever HTTP client the chosen LLM provider's official Python SDK requires

3. Check whether Poppler (required by pdf2image for PDF→image rendering) is installed on this
   system (`pdftoppm -v`). If not installed, print clear OS-specific installation instructions
   (apt/brew/choco) and STOP — this is a system-level dependency the agent may not be able to
   install without sudo/admin rights, so surface it clearly rather than silently failing.

4. Create the full folder structure exactly as specified in SRS $12.3 and $15:
   medical-rag-demo/
   ├── app.py
   ├── ingest.py
   ├── ocr.py
   ├── split_reports.py
   ├── normalize.py
   ├── chunk.py
   ├── evidence_store.py
   ├── llm_client.py
   ├── patients.py
   ├── orchestrate.py
   ├── graph_backend/
   │   ├── __init__.py
   │   ├── schema.cypher
   │   ├── build.py
   │   └── retrieve.py
   ├── pageindex_backend/
   │   ├── __init__.py
   │   ├── build.py
   │   └── retrieve.py
   ├── data/
   │   ├── raw/            (empty, .gitkeep)
   │   ├── pages/           (empty, .gitkeep)
   │   ├── ocr/              (empty, .gitkeep)
   │   ├── reports/          (empty, .gitkeep)
   │   ├── normalization/
   │   ├── evidence/         (empty, .gitkeep)
   │   └── pageindex/        (empty, .gitkeep)
   ├── tests/
   │   └── manual_checklist.md   (leave as a stub for now)
   ├── .env
   ├── .env.example
   ├── .gitignore
   └── requirements.txt

5. `.env.example` must list exactly the variables from SRS $11 (LLM_PROVIDER, LLM_API_KEY,
   LLM_MODEL, OCR_ENGINE, OCR_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) with
   placeholder values and one-line comments. Copy it to `.env` (still with placeholders —
   the human will fill in real values in Task 0.5).

6. `.gitignore` must exclude: `.venv/`, `.env`, `data/pages/`, `data/ocr/`, `__pycache__/`,
   `*.pyc`, and any large generated artifacts. Do NOT gitignore `data/raw/` (the source PDFs
   should be trackable for this demo) or `data/normalization/oncology_terms.json` (this is a
   hand-curated dictionary and should be versioned).

Output: `pip install -r requirements.txt` succeeds inside `.venv`, and the folder tree above exists.
Do not write any pipeline logic yet — this task is scaffolding only.
```

---

## TASK 0.4 — 🤝 MIXED: Install & Configure MinerU (Local OCR) with Hindi + English Support

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, especially SRS $5.2 (OCR) and $1.4
(reference dataset — note the corpus mixes typed English reports, hand-filled English tables,
and bilingual English/Hindi consent forms).

Task: Install and locally configure MinerU as the OCR engine for this project.

1. Install MinerU (package name `magic-pdf` / `mineru`, per its current PyPI distribution) into
   the project's `.venv`, along with its model-weight download step (MinerU ships/downloads
   layout detection, OCR, table-recognition, and formula-recognition models — run whichever
   official "download models" command MinerU documents, e.g. a `python download_models.py`-style
   script or `mineru-models-download` and cache them under a local directory such as
   `~/.cache/mineru` or the project's own `models/` folder — do NOT re-download on every run).

2. Confirm CPU-only operation is acceptable for the demo (no GPU is assumed available). If MinerU
   defaults to GPU and none is present, explicitly set it to CPU mode in the config and note the
   expected slower runtime in a code comment.

3. Configure MinerU's OCR language settings for **bilingual documents**:
   - MinerU's underlying OCR (PaddleOCR-based) supports multiple language packs, including Hindi
     (`hi`) and English (`en`). Configure the pipeline to run OCR with **both language models
     available**, either via MinerU's built-in multi-language/auto-detect mode if supported, or
     by running a two-pass strategy per page (English pass + Hindi pass) and keeping whichever
     pass yields higher-confidence/more-plausible output for that page — implement this as a
     configurable strategy, not a hardcoded assumption, since document mix will vary.
   - Store the language-mode decision per page in the OCR output (see Task 1.3's JSON schema) so
     downstream normalization knows whether a chunk is Devanagari or Latin script.

4. Write a tiny standalone smoke-test script `ocr_smoketest.py` (temporary, can live at project
   root) that: takes one sample page image (ask the human to supply a path, or render page 1 of
   `data/raw/patient_a.pdf` if it exists), runs it through MinerU, and prints the extracted text
   + confidence to the console.

5. Do NOT wire this into `ocr.py`/the `OCRProvider` interface yet — that's Task 1.3. This task is
   purely "prove MinerU runs locally and produces text for both English and Hindi input."

Output: running `python ocr_smoketest.py` produces readable OCR text on the console for a sample
page, and the model weights are cached locally (confirm they are NOT re-downloaded on a second run).
```

**🧑‍💻 After the agent finishes this task, YOU must:**
- Actually run `ocr_smoketest.py` against a **Hindi consent-form page** from one of the PDFs (e.g., the "सहमति पत्र" consent page seen in the sample bundle) and **visually check** the Devanagari output looks like real Hindi text and not garbage/mojibake. This is a judgment call your agent cannot verify — you are the one who can read (or paste into a translator to sanity-check) the output.
- If the output is garbled, tell the agent explicitly what's wrong (e.g., "the language mode isn't switching to Hindi for this page — it's forcing English OCR") so it can adjust the strategy in Task 0.4 before you proceed.

---

## TASK 0.5 — 🧑‍💻 YOUR TASK (manual): Fill in `.env`

Open `.env` (created in Task 0.3) and replace every placeholder with your real values from Task 0.1:
```
LLM_PROVIDER=...
LLM_API_KEY=...
LLM_MODEL=...
OCR_ENGINE=mineru
OCR_API_KEY=            # leave blank — MinerU runs fully locally, no API key needed
NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
```
**Do not commit this file** (it's already gitignored per Task 0.3). This is the only manual configuration step in the whole project.

---

## TASK 0.6 — 🤖 AGENT TASK: Verify Neo4j Connectivity

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md (SRS $6.1.1, $11).

Task: Write a tiny standalone script `neo4j_smoketest.py` that:
1. Loads NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD from `.env` using python-dotenv.
2. Opens a connection using the official `neo4j` Python driver.
3. Runs `RETURN 1 AS ok` and prints the result.
4. Closes the driver cleanly.
5. Prints a clear success/failure message (do not swallow exceptions silently).

Output: running `python neo4j_smoketest.py` prints a success message, confirming the online
Neo4j instance is reachable with the credentials the human placed in `.env` in Task 0.5.
```

---

# PHASE 1 — COMMON CORE PIPELINE
## Goal: PDFs on disk → OCR'd, normalized, chunked, evidence-tracked JSON (shared by both backends). Maps to SRS $5.

---

## TASK 1.1 — 🤖 AGENT TASK: Patient Registry

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $5.7.

Task: Implement `patients.py`.

1. Define:
   PATIENTS: dict[str, str] = {
       "patient_a": "Patient A",
       "patient_b": "Patient B",
   }

2. Add two small helpers:
   - `get_display_label(patient_id: str) -> str`
   - `all_patient_ids() -> list[str]`

3. Add a short module docstring pointing back to SRS $5.7, noting that adding a third demo
   patient means adding one dict entry here plus a PDF in data/raw/ — nothing else changes.

Output: `python -c "from patients import PATIENTS; print(PATIENTS)"` prints the two entries.
```

---

## TASK 1.2 — 🤖 AGENT TASK: PDF Ingestion → Page Images

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $5.1 (FR-5.1.1–FR-5.1.4).

Task: Implement `ingest.py`.

1. Function `discover_patient_pdfs() -> dict[str, Path]`:
   scans `data/raw/*.pdf`, derives `patient_id` from the filename stem (e.g.,
   `patient_a.pdf` → `"patient_a"`), and returns a mapping. Do NOT hardcode "exactly two files."

2. Function `render_pdf_to_pages(pdf_path: Path, patient_id: str) -> list[Path]`:
   uses `pdf2image.convert_from_path` at 250 DPI, saves each page as
   `data/pages/{patient_id}/page_{n}.png` (1-indexed), creating the directory if needed, and
   returns the list of saved paths in page order. Skip re-rendering a page if the PNG already
   exists (simple idempotency check by file existence — no need for hashing).

3. Function `ingest_all() -> dict[str, list[Path]]`:
   calls `discover_patient_pdfs()`, then `render_pdf_to_pages(...)` for each, **sequentially,
   one patient at a time, one page at a time** (per SRS FR-5.1.4 — no threading/async), printing
   progress like `"[patient_a] rendered page 3/47"` to the console.

4. Add a `if __name__ == "__main__": ingest_all()` entry point so it's runnable directly:
   `python ingest.py`.

5. Write a `pytest` test in `tests/test_ingest.py` that, given a tiny 1-page fixture PDF
   (generate one on the fly with `reportlab` or reuse a trivial PDF you create in the test
   fixture folder), asserts exactly one PNG is produced at the expected path.

Output: `python ingest.py` (with the two real PDFs already in data/raw/, placed there per
Task 0.2) populates `data/pages/patient_a/` and `data/pages/patient_b/` with one PNG per page.
```

---

## TASK 1.3 — 🤖 AGENT TASK: OCR Provider Interface + MinerU Implementation (Bilingual)

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $5.2 (FR-5.2.1–FR-5.2.5).

Task: Implement `ocr.py`, wiring in the MinerU setup proven working in Task 0.4.

1. Define an abstract interface:
   class OCRProvider(ABC):
       def ocr_page(self, image_path: Path) -> PageOCRResult: ...

   `PageOCRResult` is a dataclass/typed dict matching SRS $5.2.2:
   { page_number, raw_text, is_table, is_handwritten, confidence, script ("latin"|"devanagari"|"mixed") }
   (the `script` field is an addition needed for this project's bilingual corpus — see SRS $1.4
   and this task's language-handling requirement below; keep the rest of the schema exactly as
   the SRS specifies.)

2. Implement `MinerUProvider(OCRProvider)`:
   - Wraps the local MinerU pipeline configured in Task 0.4.
   - Runs the bilingual strategy decided in Task 0.4 (auto-detect or dual-pass) and populates
     `script` based on which language model matched / which pass was kept.
   - `is_handwritten`: use whatever signal MinerU's layout model exposes for handwriting vs
     printed text; if none is available, default to `False` and note this limitation in a
     code comment (per SRS FR-5.2.2, this is acceptable — "rely on whatever the chosen OCR
     engine reports, defaulting to typed if the engine gives no signal", SRS FR-5.5.4).
   - `is_table`: use a simple visual-density heuristic (per SRS FR-5.2.3) — e.g., high count of
     short numeric tokens arranged in a grid-like pattern — or MinerU's own table-region output
     if available. Do not build a trained classifier.

3. Also stub `ChandraOCRProvider(OCRProvider)` with a `NotImplementedError` body and a docstring
   noting it exists only to satisfy the interface for future swap-in (per SRS FR-5.2.1) — it does
   not need to work for this project.

4. Function `get_ocr_provider() -> OCRProvider` reads `OCR_ENGINE` from `.env` and returns the
   matching instance (`mineru` → MinerUProvider; anything else → raise a clear error).

5. Function `ocr_all_pages(patient_id: str) -> None`:
   for every PNG under `data/pages/{patient_id}/`, in page-number order, call `ocr_page(...)`
   and write the result as `data/ocr/{patient_id}/page_{n}.json`. Sequential, per SRS FR-5.1.4.
   Skip a page if its OCR JSON already exists AND is newer than the PNG (simple idempotency).

6. Implement the near-duplicate skip logic from SRS FR-5.2.5: while iterating a patient's pages
   in order, keep a running list of already-seen `raw_text` strings (normalized: lowercase,
   whitespace-collapsed); if a new page's normalized text has >90% similarity (use
   `difflib.SequenceMatcher` ratio) to any previously seen page for the SAME patient, keep only
   the higher-confidence one — write BOTH json files, but mark the lower-confidence one with an
   extra field `"duplicate_of": <page_number>` so downstream chunking (Task 1.7) can skip it, and
   print a log line noting the skip.

7. `if __name__ == "__main__":` iterate `patients.all_patient_ids()` and call `ocr_all_pages` for
   each — runnable as `python ocr.py`.

Output: `python ocr.py` (after `python ingest.py` has already produced page PNGs) populates
`data/ocr/patient_a/` and `data/ocr/patient_b/` with one JSON file per page.
```

**🧑‍💻 After the agent finishes this task, YOU must:**
- Spot-check a handful of `data/ocr/*/page_*.json` files, especially: (a) a typed English radiology report page, (b) a hand-filled flowsheet table page, (c) a Hindi consent-form page, (d) a cursive free-hand progress-note page. Confirm the `raw_text` is roughly right for (a)/(c), the `is_table` flag is `true` for (b), and note that (d) will likely have low-quality text — that's expected (SRS D13/NFR-1), just confirm the pipeline didn't crash on it.
- This manual review step cannot be delegated — judging "is this OCR output good enough" for medical text requires a human who can read the source pages.

---

## TASK 1.4 — 🤖 AGENT TASK: Run Full OCR Pass

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md.

Task: Run the ingestion + OCR pipeline end-to-end and produce a short console/markdown summary.

1. Run `python ingest.py` then `python ocr.py` (or write a tiny `run_ocr_pipeline.py` that calls
   both in sequence) against the two real PDFs.

2. After completion, print a summary table to the console (and optionally save as
   `data/ocr/_summary.md`):
   - Per patient: total pages, count of `is_table=true` pages, count of `is_handwritten=true`
     pages, count flagged `script="devanagari"`/`"mixed"`, count of pages with `confidence < 0.5`,
     and count of pages marked `duplicate_of` (i.e., skipped as near-duplicates).

Output: the summary table, so both you and the human (Task 1.3's manual review) know at a glance
how much of the corpus is low-confidence before normalization begins.
```

---

## TASK 1.5 — 🤖 AGENT TASK: Report Boundary Splitting

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $5.3 (FR-5.3.1–FR-5.3.4) and SRS $1.4/$12.

Task: Implement `split_reports.py`.

1. Define a constant anchor list (module-level, easy to extend) mapping a substring match
   (case-insensitive, checked against each page's OCR `raw_text`) to a `report_type` enum value
   from SRS $12.3's list. Seed it with anchors observed in the reference corpus, e.g.:
   - "RADIOLOGY UNIT" + "CECT" → RADIOLOGY_CECT
   - "ULTRASOUND" (and not "MAMMOGRAM"/"CHEST & BREAST" already matched) → RADIOLOGY_USG
   - "MAMMOGRAPHY" / "MAMMOGRAM" → RADIOLOGY_MAMMOGRAM
   - "PET-CT" / "PET CT" / "POSITRON EMISSION" → RADIOLOGY_PET_CT
   - "BONE SCAN" / "MDP WHOLE BODY" → RADIOLOGY_BONE_SCAN
   - "HISTOPATHOLOGY REPORT" / "Histopathology Report" → HISTOPATHOLOGY
   - "CYTOPATHOLOGY REPORT" → CYTOPATHOLOGY
   - "Flowsheet Medical Oncology" → ONCOLOGY_FLOWSHEET
   - "DAYCARE DRUGS ADMINISTERED" → CHEMO_DRUG_ADMIN_RECORD
   - "DISCHARGE SUMMARY" → DISCHARGE_SUMMARY
   - "OPERATIVE" / "SURGICAL ONCOLOGY" → SURGICAL_OPERATIVE_NOTE
   - "ANAESTHESIA RECORD" / "PRE ANAESTHETIC" / "PAC" → ANESTHESIA_RECORD
   - "ECHOCARDIOGRAPHY REPORT" → ECHOCARDIOGRAPHY
   - "सहमति पत्र" / "CONSENT" → CONSENT_FORM   (Hindi + English anchor — bilingual match)
   - CBC/biochemistry lab report letterheads (e.g., "HAEMATOLOGY", "BIOCHEMISTRY", "Department
     Of Pathology" without "Histopathology"/"Cytopathology" in the same page) → LAB_REPORT_GENERAL
   - Fallback: if no anchor matches and no report is currently open → CLINICIAN_PROGRESS_NOTE

2. Function `split_patient_reports(patient_id: str) -> list[ReportSpan]` where `ReportSpan` is
   `{report_id, report_type, page_start, page_end, report_date}`:
   - Load all OCR JSONs for the patient in page order (skip any page marked `duplicate_of`,
     per Task 1.3).
   - Walk pages in order; on an anchor match, close the currently-open report span (if any) and
     open a new one; pages matching no anchor extend the currently-open span (SRS FR-5.3.3).
   - `report_id` = `f"report_{page_start:04d}"`.

3. Function `extract_report_date(text: str) -> str | None`: regex for `DD-MM-YYYY`, `DD/MM/YYYY`,
   `DD.MM.YY`, `DD-MON-YYYY` (case-insensitive month abbreviations) applied to the first ~500
   characters of the report's opening page; return ISO `YYYY-MM-DD` or `None` if no confident
   match (SRS FR-5.3.4). Note some dates in the corpus are 2-digit years — assume 20xx for years
   00–79, 19xx otherwise, and log an explicit assumption comment.

4. Persist to `data/reports/{patient_id}.json` as a list of `ReportSpan` dicts.

5. `if __name__ == "__main__":` run for both patients and print a per-patient table of
   detected reports (type, page range, date) to the console for quick eyeballing.

6. Write `tests/test_split_reports.py`: using the real OCR output already produced in Task 1.4,
   assert that the number of detected reports for each patient is within a small reasonable
   range and that at least one `HISTOPATHOLOGY`, one `RADIOLOGY_PET_CT`, and one
   `ONCOLOGY_FLOWSHEET` report are found per patient (these report types are confirmed present
   in the reference corpus).

Output: `python split_reports.py` prints a clean per-patient report list; `data/reports/*.json`
files exist.
```

**🧑‍💻 After the agent finishes this task, YOU must:**
- Glance at the printed report list per patient and confirm it roughly matches what you know is in the PDFs (e.g., "yes, there really are ~6 radiology reports, ~4 histopathology reports, several flowsheet pages"). If the anchor list is clearly missing/mis-splitting a report type, tell the agent which report and what page it's on so it can add/fix an anchor.

---

## TASK 1.6 — 🤖 AGENT TASK: Terminology Normalization Dictionary + Normalizer (Bilingual)

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $5.4 (FR-5.4.1–FR-5.4.4).

Task: Implement the normalization dictionary and `normalize.py`.

1. Create `data/normalization/oncology_terms.json` seeded with entries for every abbreviation
   observed in the reference corpus, at minimum:
   MRM, NAC, BCC, IDC, NST, NOS, LVI, EC (regimen), AC (regimen), IMF (regimen), 5FU, MO,
   PET-CT, CECT, USG, IRCH, OPD, ER, PR, HER2/Her2neu, Ki67, TNM prefix letters (p/y/r),
   FNAC, LN (lymph node), Hb, WBC, ANC, ESR, LFT, RFT, CBC.
   Each entry: `{"canonical": "...", "type": "Procedure|Regimen|Medication|Diagnosis|Modifier|
   Biomarker|LabTest|Finding"}` per SRS FR-5.4.1's example shape.

2. ALSO add a small `hindi_terms` section to the same file (or a sibling file
   `data/normalization/hindi_terms.json` — developer's choice, document which) mapping the
   Hindi/Devanagari words/phrases actually seen in the corpus's consent-form boilerplate (e.g.,
   "सहमति पत्र" → {"canonical": "Consent Form", "type": "DocumentSection"}, "रोगी के हस्ताक्षर" →
   {"canonical": "Patient Signature", "type": "DocumentSection"}) so these bilingual sections are
   still classifiable rather than silently dropped. This dictionary does not need to be
   exhaustive — only cover phrases that actually recur in the two sample PDFs.

3. Implement `normalize.py`:
   - `dictionary_match(term: str) -> NormalizedEntity | None`: exact + simple fuzzy match
     (e.g., `difflib.get_close_matches`) against BOTH dictionaries loaded from step 1/2.
   - `regex_parsers(text: str) -> list[NormalizedEntity]`: implement the structured parsers from
     SRS FR-5.4.2 — TNM staging (`T\d[a-z]?\s*N\d[a-z]?\s*M[0x]`), hormone-receptor scores
     (`ER[\s\-]*[+\-]?\s*\d/\d`, `PR[\s\-]*[+\-]?\s*\d/\d`, `Her-?2\s*neu[\s\-:]*\d\+`),
     Nottingham score triplets (e.g., "Tubule Formation - 3", "Nuclear Pleomorphism - 2",
     "Mitosis - 3", "Total Score - 8, Grade 3").
   - `llm_normalize(term: str, context: str) -> NormalizedEntity | None`: calls `llm_client.chat`
     (built in Task 2.1 — if that task isn't done yet, stub this function to raise
     `NotImplementedError` with a clear TODO, and return to wire it up after Task 2.1) with a
     prompt listing the dictionary's allowed canonical terms/types and instructing "return null
     if none apply" (SRS FR-5.4.2, step 3).
   - `normalize_chunk_text(raw_text: str, script: str) -> list[NormalizedEntity]`: tokenizes the
     text into candidate terms/phrases (simple whitespace + punctuation-aware splitting is
     sufficient — no need for a full NER model) and runs dictionary → regex → LLM fallback in
     that order per SRS FR-5.4.2, tagging each result's `method`.
   - `normalize_date(raw: str) -> str | None`: reuse/import the date regex from Task 1.5.
   - Each `NormalizedEntity` keeps `raw_text, normalized_term, entity_type, method, confidence`
     (confidence inherited from the source chunk's OCR confidence, minus a fixed penalty, e.g.
     -0.1, if `method == "llm"` — per SRS FR-5.4.3).

4. Write `tests/test_normalize.py` asserting: "MRM" → Procedure/"Modified Radical Mastectomy" via
   dictionary; "T2N0M0" → parsed TNM dict via regex; "ER 8/8" → parsed biomarker via regex; an
   unknown random string → `method == "llm"` or `None` (not a false-positive dictionary match).

Output: `data/normalization/oncology_terms.json` (+ Hindi terms file) exist and are
human-readable/editable; `pytest tests/test_normalize.py` passes for the non-LLM cases.
```

---

## TASK 1.7 — 🤖 AGENT TASK: Chunking + Evidence Store

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $5.5 (FR-5.5.1–FR-5.5.4) and $12.2 for the
exact EvidenceRecord JSON shape.

Task: Implement `chunk.py` and `evidence_store.py`.

1. `evidence_store.py`:
   - `EvidenceRecord` typed dict/dataclass matching SRS $5.5.2 exactly (evidence_id, patient_id,
     report_id, report_type, report_date, page_number, raw_text, source_type, confidence,
     page_image_path).
   - `append_evidence(patient_id: str, record: EvidenceRecord) -> None`: loads
     `data/evidence/{patient_id}.json` (creating an empty list if missing), appends, writes back.
   - `load_evidence(patient_id: str) -> list[EvidenceRecord]`.
   - `get_evidence_by_id(patient_id: str, evidence_id: str) -> EvidenceRecord | None`.

2. `chunk.py`:
   - `source_type_for_page(ocr_result: dict) -> str`: implement SRS FR-5.5.4's heuristic mapping
     from `is_table`/`is_handwritten` OCR flags to `"typed" | "tabular_handwritten" |
     "cursive_handwritten"`.
   - `chunk_report(patient_id: str, report_span: dict) -> list[Chunk]`:
     - For non-table report types: one chunk per page in the report's page range (SRS FR-5.5.1).
     - For `ONCOLOGY_FLOWSHEET` report type: if the OCR'd page's table extraction (Task 1.3,
       step re: `is_table`) produced structured `rows`, emit one chunk per row (per date) PLUS
       one chunk that is a simple textual summary of the whole table for that page (SRS FR-5.5.1
       / FR-5.2.3). If table extraction fell back to plain text, just emit one chunk for the
       page (same as the non-table case) — do not fail.
   - Each chunk immediately becomes one `EvidenceRecord` via `append_evidence(...)`, with
     `evidence_id = f"{patient_id}__{report_id}__page_{n}__chunk_{i}"` (SRS $5.5.2 format).
   - Each chunk also runs `normalize.normalize_chunk_text(...)` (Task 1.6) and stores the
     resulting `normalized_entities` list ALONGSIDE the evidence record (e.g., as an in-memory
     `Chunk` object returned to the caller — normalized entities do not need their own JSON file
     per SRS; they are consumed directly by the graph/PageIndex builders in Phase 2/3). Also
     write this in-memory chunk list (including normalized entities) to
     `data/reports/{patient_id}_chunks.json` so Phase 2/3 builders don't need to redo
     OCR+normalization from scratch — this file is the single canonical chunk source both
     backends read from (keep Phase 2 and Phase 3 consistent).
   - `chunk_all_patients() -> dict[str, list[Chunk]]`: for each patient, load `data/reports/*.json`
     (Task 1.5) and OCR JSONs (Task 1.3), run `chunk_report` for every report span, skipping any
     page marked `duplicate_of`. Sequential, per SRS FR-5.1.4's spirit.

3. `if __name__ == "__main__":` run `chunk_all_patients()` and print, per patient, total chunk
   count and total evidence records written.

4. Write `tests/test_chunk_evidence.py`: build one small synthetic report span + fake OCR JSON,
   run `chunk_report`, assert exactly one evidence record per expected chunk and that
   `evidence_store.get_evidence_by_id(...)` can retrieve each one back (SRS's "round-trip" test
   from $14).

Output: `python chunk.py` (after Tasks 1.1–1.6 have run) populates `data/evidence/patient_a.json`
and `data/evidence/patient_b.json` with the full evidence list for each patient, plus
`data/reports/patient_a_chunks.json` / `data/reports/patient_b_chunks.json`.
```

---

## TASK 1.8 — 🤝 MIXED: End-to-End Phase 1 Smoke Test

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md.

Task: Write `run_pipeline_phase1.py`, a single script that runs, in order:
ingest.ingest_all() → ocr.py's per-patient OCR loop → split_reports (both patients) →
chunk.chunk_all_patients() — with a clear console banner between each stage
("=== STAGE: OCR ===" etc.) and a final summary: total pages, total reports detected, total
chunks, total evidence records, split by patient.

Output: `python run_pipeline_phase1.py` runs the whole Common Core pipeline start to finish
against the two real PDFs and prints the final summary with no unhandled exceptions.
```

**🧑‍💻 After the agent finishes this task, YOU must:**
- Run it once, read the summary, and sanity-check the numbers feel right (e.g., "yes, ~35 pages, ~13 reports, ~35+ chunks for patient A" — you know your source PDFs' rough size). This is the checkpoint before spending LLM API calls in Phase 2/3, so it's worth 5 minutes of your attention.

---

# PHASE 2 — GRAPHRAG BACKEND (NEO4J)
## Goal: A queryable knowledge graph in your online Neo4j instance. Maps to SRS $6.

---

## TASK 2.1 — 🤖 AGENT TASK: LLM Client Wrapper

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $5.6 (FR-5.6.1–FR-5.6.2).

Task: Implement `llm_client.py`.

1. `chat(messages: list[dict], system: str | None = None) -> str`: loads `LLM_PROVIDER`,
   `LLM_API_KEY`, `LLM_MODEL` from `.env`; calls the corresponding provider's official Python
   SDK; returns the plain text response. Keep this to ONE function as SRS FR-5.6.1 specifies —
   no provider-abstraction class hierarchy needed.
2. Add basic retry-once-on-transient-error handling (simple try/except + one retry), and raise
   a clear exception on repeated failure (no silent fallback to a fake response).
3. No embeddings function is needed (SRS FR-5.6.2) — do not add one.
4. Write `tests/test_llm_client.py` with the actual API call mocked (do not spend real API
   credits in automated tests) asserting the function is called with the right model/messages
   shape.

5. Now go back and finish Task 1.6's `llm_normalize` stub — wire it to call `llm_client.chat`.

Output: a small manual script `llm_smoketest.py` that sends "Say hello in one word." and prints
the response — run once to confirm your `LLM_API_KEY` from Task 0.5 actually works.
```

---

## TASK 2.2 — 🤖 AGENT TASK: Neo4j Schema Init Script

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $6.1.2 and $12.1 (the fixed graph schema
— do not add node labels or relationship types beyond what $12.1 lists).

Task: Create `graph_backend/schema.cypher` containing idempotent constraint statements
(`CREATE CONSTRAINT IF NOT EXISTS ...`) for:
- Patient.patient_id (unique)
- Diagnosis.canonical_name (unique)
- Medication.canonical_name (unique)
- LabTest.canonical_name (unique)

Also add a small Python function `run_schema_init()` in `graph_backend/build.py` (create the file
now, fill in the rest in Task 2.3) that reads `schema.cypher`, splits on `;`, and executes each
statement against the Neo4j instance from `.env` using the driver already proven in Task 0.6.

Output: running `python -c "from graph_backend.build import run_schema_init; run_schema_init()"`
completes without error; confirm in the Neo4j browser console (🧑‍💻 quick manual check) that the
constraints now exist (`SHOW CONSTRAINTS`).
```

---

## TASK 2.3 — 🤖 AGENT TASK: Graph Build (Entity/Relation Extraction + MERGE)

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $6.1 (FR-6.1.1–FR-6.1.5) and $12.1
(the fixed node/relationship schema — the LLM extraction prompt in step 2 below must be
constrained to ONLY these labels/relationship types).

Task: Finish `graph_backend/build.py`.

1. `get_driver()`: returns a Neo4j driver built from `.env` credentials (reuse Task 0.6's logic).

2. `extract_triples(chunk_text: str, normalized_entities: list, report_meta: dict) -> list[dict]`:
   calls `llm_client.chat` with a fixed prompt that:
   - Provides the SRS $12.1 schema (node labels + relationship types) verbatim as the ONLY
     allowed vocabulary.
   - Provides the chunk's raw text + its already-normalized entities (Task 1.6/1.7) as context
     (pre-normalized terms should be reused, not re-invented, wherever possible).
   - Asks for a JSON list of triples: `{"subject_label": ..., "subject_name": ..., "relation":
     ..., "object_label": ..., "object_name": ..., "properties": {...}}`.
   - Instructs: "if nothing in this chunk maps cleanly to the schema, return an empty list" —
     do not force extraction.
   Parse the JSON response defensively (wrap in try/except, log and skip on malformed output —
   do not crash the whole build over one bad LLM response).

3. `write_triples_to_neo4j(driver, patient_id: str, report_meta: dict, triples: list[dict],
   evidence_id: str, confidence: float) -> None`:
   For each triple, run a `MERGE`-based Cypher statement (one per node type it might touch) that:
   - `MERGE`s the `Patient` node on `patient_id`.
   - `MERGE`s the `Report` node on `report_id` (link `(Patient)-[:HAS_REPORT]->(Report)`).
   - `MERGE`s the subject/object nodes on their canonical name (per SRS $12.1's unique
     constraints) — reuse existing nodes rather than duplicating (SRS FR-6.1.4).
   - `MERGE`s the relationship itself, and sets `evidence_id` and `confidence` as properties on
     the relationship (SRS FR-6.1.4) — use `SET` after `MERGE` so re-running doesn't create
     duplicate relationship instances for the same triple.

4. `build_graph_for_patient(patient_id: str) -> None`: loads the patient's chunks from
   `data/reports/{patient_id}_chunks.json` (written in Task 1.7 — this is the canonical chunk
   source; do NOT re-run OCR/normalization here), and for each chunk calls `extract_triples(...)`
   then `write_triples_to_neo4j(...)`. **Sequential — one chunk at a time** (SRS FR-6.1.5).

5. `if __name__ == "__main__":` run `run_schema_init()` then `build_graph_for_patient(...)` for
   every `patients.all_patient_ids()`, printing progress per chunk.

6. Write `tests/test_graph_build.py` covering `extract_triples`'s JSON parsing logic against a
   few hand-written mock LLM responses (valid JSON, malformed JSON, empty list) — mock the LLM
   call, do not hit the real API or a real Neo4j instance in automated tests.

Output: `python graph_backend/build.py` populates your online Neo4j instance with nodes/
relationships for both patients. Re-running it a second time does not change the node/
relationship COUNT (idempotency, SRS NFR-3) — 🧑‍💻 verify this yourself with a quick
`MATCH (n) RETURN count(n)` before/after a second run in the Neo4j browser.
```

---

## TASK 2.4 — 🤖 AGENT TASK: Graph Retrieval (Cypher Templates + LLM Fallback)

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $6.2 (FR-6.2.1–FR-6.2.4).

Task: Implement `graph_backend/retrieve.py`.

1. Define a small library of parameterized Cypher template functions, each ALWAYS taking a
   `patient_ids: list[str]` argument and looping over it in Python (SRS FR-6.2.4 — never a
   single cross-patient Cypher query):
   - `get_lab_trend(patient_id: str, test_name: str) -> list[dict]` — timeline of LabResult
     values for a given LabTest.canonical_name.
   - `get_diagnoses(patient_id: str) -> list[dict]` — all Diagnosis nodes + report dates.
   - `get_medication_history(patient_id: str) -> list[dict]` — all Medication/Regimen
     administrations with dates/doses/cycles.
   - `get_staging_and_biomarkers(patient_id: str) -> list[dict]` — Staging + Biomarker nodes.
   Each returns rows that already include `evidence_id` and `confidence` (read off the
   relationship properties per SRS $12.1) so nothing downstream needs a second lookup.

2. `classify_intent(question: str) -> str`: a simple keyword/LLM-based classifier returning one
   of `"lab_trend" | "diagnosis_list" | "medication_history" | "staging_biomarker" | "open_ended"`
   (SRS FR-6.2.1, step 1). A cheap approach (keyword matching first, LLM only if ambiguous) is
   fine — this does not need to be a trained classifier.

3. `open_ended_query(patient_ids: list[str], question: str) -> list[dict]` (SRS FR-6.2.2): builds
   ONE Cypher query via `llm_client.chat`, giving the LLM the SRS $12.1 schema as its only
   allowed vocabulary, with an explicit instruction that the query MUST filter on
   `patient_id IN $patient_ids` — but the calling Python code passes `$patient_ids` as a bound
   parameter itself (never string-interpolated), so scoping cannot be bypassed even if the LLM
   forgets the WHERE clause wording (defense in depth per SRS FR-6.2.2's intent).

4. `retrieve(question: str, patient_ids: list[str]) -> list[dict]`: the single public entry
   point — classifies intent, dispatches to the matching template (looped per patient_id, per
   SRS FR-6.2.4) or falls back to `open_ended_query`, and returns a flat list of
   `{"text": <human-readable fact string>, "patient_id": ..., "evidence_id": ...}` dicts (the
   common `facts` contract SRS $8.1 expects from both backends).

5. Write `tests/test_graph_retrieve.py` covering `classify_intent`'s keyword paths (mock the LLM
   fallback) — do not require a live Neo4j connection for these unit tests; mock the driver calls.

Output: a manual script `graph_query_smoketest.py` that calls
`retrieve("What is my hemoglobin trend?", ["patient_a"])` against the REAL Neo4j instance built
in Task 2.3, and prints the returned facts.
```

**🧑‍💻 After the agent finishes this task, YOU must:**
- Run `graph_query_smoketest.py` for 2–3 of the use cases in SRS $4.2 and eyeball whether the returned facts look medically sensible for the patient in question. Only you can judge "does this look right" against the source PDFs.

---

# PHASE 3 — PAGEINDEX BACKEND
## Goal: A per-patient reasoning-traversable tree, no vector store. Maps to SRS $7.

---

## TASK 3.1 — 🤖 AGENT TASK: PageIndex Tree Build

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $7.1 (FR-7.1.1–FR-7.1.4) and $12.2 for
the exact tree JSON shape.

Task: Implement `pageindex_backend/build.py`.

1. `build_tree_for_patient(patient_id: str) -> dict`:
   - Load `data/reports/{patient_id}.json` (report spans, Task 1.5) and
     `data/reports/{patient_id}_chunks.json` (the SAME canonical chunk source Task 2.3 reads from
     — both backends must build from identical data).
   - Build the 3-level tree exactly as SRS $12.2 shows: `Patient (root) → Report → Page (leaf)`.
   - For each `Report` node, call `llm_client.chat` ONCE with the concatenated text of all its
     child page chunks (truncate/summarize input if it's very long — a report is at most a
     handful of pages in this corpus) and ask for a 2–4 sentence summary (SRS FR-7.1.2). Store
     this as the node's `"summary"` field.
   - For the `root` node, call `llm_client.chat` ONCE more with the concatenation of all
     Report-level summaries just generated, asking for a whole-patient overview summary
     (SRS FR-7.1.2's "non-leaf node" requirement applies to the root too).
   - Leaf (`Page`) nodes store `raw_text` and `evidence_id` directly (SRS FR-7.1.3) — no LLM call
     needed for leaves.

2. `save_tree(patient_id: str, tree: dict) -> None`: writes to `data/pageindex/{patient_id}.json`
   (SRS FR-7.1.4) — this file IS the index, no database involved.

3. `if __name__ == "__main__":` build + save for every `patients.all_patient_ids()`, sequential,
   printing progress (SRS's overall "no async" stance applies here too).

4. Write `tests/test_pageindex_build.py`: given a tiny synthetic 1-report, 2-page patient fixture
   (mock the LLM summary calls to return a fixed string), assert the resulting tree has the
   expected 3-level shape and that leaf `evidence_id`s match real entries in
   `evidence_store.load_evidence(...)`.

Output: `python pageindex_backend/build.py` produces `data/pageindex/patient_a.json` and
`data/pageindex/patient_b.json`.
```

---

## TASK 3.2 — 🤖 AGENT TASK: PageIndex Retrieval (Pure LLM Reasoning Traversal)

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $7.2 (FR-7.2.1–FR-7.2.4). Reminder: NO
embeddings, NO vector similarity, anywhere in this file — selection is 100% LLM reasoning over
node summaries, per SRS $7.2 heading and SRS D9.

Task: Implement `pageindex_backend/retrieve.py`.

1. `load_tree(patient_id: str) -> dict`: reads `data/pageindex/{patient_id}.json` directly (this
   read IS the Individual-mode scope boundary per SRS FR-7.2.1 — the code never opens another
   patient's file for an Individual-mode query).

2. `select_children(question: str, node: dict, allow_multi: bool) -> list[dict]`: calls
   `llm_client.chat`, presenting the question plus each child's `"summary"` (and `node_id`), and
   asks the LLM to return a JSON list of `node_id`s to descend into — zero, one, or (if
   `allow_multi=True`) several (SRS FR-7.2.3, for timeline-style questions). Parse defensively;
   on malformed output, default to expanding no children rather than crashing.

3. `traverse(question: str, patient_id: str, max_depth: int = 3, max_nodes_expanded: int = 6,
   allow_multi: bool = False) -> list[dict]` (SRS FR-7.2.2):
   - Recursive/iterative walk starting at root, calling `select_children` at each level, stopping
     at `max_depth` or `max_nodes_expanded`, or when a node returns "none relevant."
   - Collect every LEAF reached along any explored path into a list.
   - Return each leaf as `{"text": <leaf raw_text>, "patient_id": ..., "evidence_id": ...}` — the
     same common `facts` shape SRS $8.1 expects (matching graph_backend's output contract).

4. `should_allow_multi(question: str) -> bool`: a simple heuristic (keyword check for
   "trend"/"progress"/"over time"/"history"/"all" — or reuse `graph_backend.retrieve
   .classify_intent`'s `"lab_trend"` result if convenient) deciding whether to pass
   `allow_multi=True` into `traverse` (SRS FR-7.2.3).

5. `retrieve(question: str, patient_ids: list[str]) -> list[dict]`: the public entry point —
   loops `traverse(...)` independently per `patient_id` in the input list (SRS FR-7.2.4, mirrors
   `graph_backend.retrieve`'s per-patient loop design) and tags/merges results.

6. Write `tests/test_pageindex_retrieve.py` mocking `llm_client.chat`'s node-selection responses
   against a small fixed fake tree, asserting: depth/breadth limits are respected, and multi-node
   selection works when `allow_multi=True`.

Output: a manual script `pageindex_query_smoketest.py` calling
`retrieve("What is my hemoglobin trend?", ["patient_a"])` against the REAL tree built in Task 3.1.
```

**🧑‍💻 After the agent finishes this task, YOU must:**
- As in Task 2.4, run 2–3 real questions through this and sanity-check the results look right. Also compare, side-by-side, what GraphRAG (Task 2.4) vs PageIndex (this task) each returned for the SAME question — this comparison is the whole point of building both backends, and only you can judge whether the two feel meaningfully different/better for different question types.

---

# PHASE 4 — ORCHESTRATION & ANSWER GENERATION
## Goal: One function that turns "question + mode + patients + backend" into a cited answer. Maps to SRS $8.

---

## TASK 4.1 — 🤖 AGENT TASK: Orchestration Layer

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $8.1 and $8.2 (FR-8.1.1–FR-8.2.3).

Task: Implement `orchestrate.py`.

1. `answer_question(question: str, mode: str, selected_patients: list[str], backend: str) ->
   dict` — exact signature from SRS FR-8.1.1.
   - Validate `mode in ("individual", "group")` and `backend in ("graph", "pageindex")`; raise a
     clear `ValueError` otherwise (this function must never silently misroute — SRS FR-8.1.1
     notes the Streamlit layer already resolves "All patients" into a full list before calling
     this, so no special-casing of "All" belongs here).
   - Dispatch: `backend == "graph"` → `graph_backend.retrieve.retrieve(question,
     selected_patients)`; `backend == "pageindex"` → `pageindex_backend.retrieve.retrieve(
     question, selected_patients)`. Both already return the common `facts` list shape (Tasks 2.4
     step 4 / 3.2 step 5).

2. `generate_answer(question: str, facts: list[dict], mode: str) -> tuple[str, list[str]]`
   (SRS $8.2):
   - Build one prompt from `facts` + `question`, instructing the LLM (SRS FR-8.2.1): answer only
     from given facts; cite each claim with `[evidence_id]`-style markers; say "not documented in
     the available records" when insufficient; if the question reads as seeking medical
     advice/diagnosis/treatment recommendation, answer only with what's documented and append one
     short disclaimer sentence (reuse wording consistent with SRS D14).
   - If `mode == "group"` and `len({f["patient_id"] for f in facts}) > 1`, add an explicit
     instruction (SRS FR-8.2.3) to structure the answer per patient (one paragraph or table row
     per `patient_id`), never blending uncited cross-patient claims into one sentence.
   - Call `llm_client.chat(...)`, get the raw answer text.
   - Grounding check (SRS FR-8.2.2): regex-extract every `[evidence_id]`-style marker from the
     answer text; drop/flag any that don't appear in the `evidence_id` set of `facts`; return the
     (possibly filtered) answer plus the final clean list of valid citation `evidence_id`s.

3. `answer_question` calls `generate_answer` internally and returns
   `{"answer": str, "citations": list[str], "backend_used": backend}` (SRS FR-8.1.1's exact
   return contract).

4. Write `tests/test_orchestrate.py` mocking both backends' `retrieve()` and `llm_client.chat`,
   asserting: invalid `mode`/`backend` raise; the grounding check correctly strips a fabricated
   citation marker not present in the mocked facts.

Output: a manual script `orchestrate_smoketest.py` that calls `answer_question(...)` for one
question from EACH use case in SRS $4.2 (UC-1 through UC-6), for BOTH backends, and prints the
results.
```

**🧑‍💻 After the agent finishes this task, YOU must:**
- Read through the six-use-cases-times-two-backends output from `orchestrate_smoketest.py`. This is the real functional acceptance test of the whole pipeline before UI work begins — worth 15–20 minutes of careful reading against the source PDFs.

---

# PHASE 5 — STREAMLIT UI
## Goal: The only user-facing surface. Maps to SRS $9.

---

## TASK 5.1 — 🤖 AGENT TASK: App Skeleton + Sidebar Controls

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $9.1 (FR-9.1.1, FR-9.1.4).

Task: Implement the skeleton of `app.py`.

1. `st.set_page_config(page_title="Medical Records RAG (Demo)", layout="wide")`.
2. Sidebar, top to bottom, exactly per SRS FR-9.1.1:
   - Mode dropdown: `st.selectbox("Mode", ["Individual", "Group"])`.
   - Patient selector:
     - Individual → `st.selectbox("Patient", list(patients.PATIENTS.values()))`, mapped back to
       `patient_id`.
     - Group → `st.multiselect("Patients", list(patients.PATIENTS.values()))` PLUS a
       `st.checkbox("All patients")` that, if checked, overrides the multiselect and resolves to
       `patients.all_patient_ids()` — this "All" resolution happens HERE in the UI layer, never
       inside `orchestrate.py` (per Task 4.1's note and SRS FR-8.1.1).
   - Backend toggle: `st.radio("Backend", ["GraphRAG (Neo4j)", "PageIndex"])`, mapped to
     `"graph"`/`"pageindex"`.
   - Optional checkbox: "Compare both backends" (SRS FR-9.1.1's optional nice-to-have).
3. `st.session_state` holds chat history as a list of `{"role": "user"|"assistant", "content":
   ..., "citations": [...] }` dicts — no other persistence (SRS FR-9.1.4: no accounts, no DB of
   users).
4. Leave the main panel as a placeholder ("Chat UI — Task 5.2") and the evidence panel as a
   placeholder ("Evidence viewer — Task 5.3") for now.

Output: `streamlit run app.py` opens a working sidebar with all controls wired to
`st.session_state`, no chat functionality yet.
```

---

## TASK 5.2 — 🤖 AGENT TASK: Chat Interface Wired to Orchestration

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $9.1.2.

Task: Complete the main panel of `app.py`.

1. Render existing `st.session_state` chat history with `st.chat_message("user"/"assistant")`.
2. `st.chat_input("Ask a question about the medical records...")` — on submit:
   - Append the user message to history and render it.
   - Resolve the current sidebar selections into `mode`, `selected_patients` (list of
     `patient_id`s, resolving "All patients" here per Task 5.1), and `backend`.
   - If the "Compare both backends" checkbox (Task 5.1) is OFF: call
     `orchestrate.answer_question(question, mode, selected_patients, backend)` once, show a
     `st.spinner("Thinking...")` while it runs, then render the answer as an assistant message,
     storing its `citations` list alongside it in session state.
   - If "Compare both backends" is ON: call `answer_question` twice (`backend="graph"` and
     `backend="pageindex"`), render BOTH answers side by side using `st.columns(2)`, each
     labeled with its backend name.
3. Wrap the `answer_question` call in try/except; on error, render a clear
   `st.error(f"Something went wrong: {e}")` instead of crashing the app.

Output: end-to-end chat works in the browser — pick Individual/patient_a/GraphRAG, ask "What is
my health progress after multiple reports?", get a cited answer rendered in the chat.
```

---

## TASK 5.3 — 🤖 AGENT TASK: Evidence Panel

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $9.1.3.

Task: Add the evidence viewer to `app.py`.

1. For each assistant message rendered (Task 5.2), below the answer text, render one small
   button/expander per citation `evidence_id` (e.g., `st.expander(f"Source: {evidence_id}")`).
2. Inside each expander, call `evidence_store.get_evidence_by_id(patient_id, evidence_id)`
   (patient_id parsed out of the `evidence_id` string per its `{patient_id}__...` format from
   SRS $5.5.2) and render:
   - `st.image(record["page_image_path"])` — the full source page.
   - `st.code(record["raw_text"])` — the raw OCR text (SRS FR-9.1.3: never hide the raw text
     behind only the normalized/paraphrased version).
   - `st.caption(f"{record['report_type']} — {record['report_date']} — page
     {record['page_number']}")`.
   - A confidence badge: green if `confidence >= 0.8`, yellow if `0.5 <= confidence < 0.8`, red
     if `confidence < 0.5` (SRS FR-9.1.3's exact thresholds) — implement with colored
     `st.markdown` (e.g., `:green[High confidence]` / `:orange[Medium confidence]` /
     `:red[Low confidence]`, Streamlit's markdown color syntax) plus the numeric value.

Output: clicking any citation in the chat opens the full evidence trail — page image, raw text,
report metadata, confidence — matching SRS's "evidence is extremely important" requirement.
```

---

## TASK 5.4 — 🤖 AGENT TASK: "Rebuild Index" Admin Button

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $9.2 (FR-9.2.1).

Task: Add a "⚙️ Admin" tab/section to `app.py` (e.g., via `st.tabs(["Chat", "Admin"])`,
restructuring Tasks 5.1–5.3's content into the "Chat" tab).

1. In the Admin tab, a single button: "Rebuild index from data/raw/".
2. On click, run — with a `st.status(...)` block showing live stage progress — in order:
   `ingest.ingest_all()` → OCR loop (Task 1.3) → `split_reports` for both patients (Task 1.5) →
   `chunk.chunk_all_patients()` (Task 1.7) → `graph_backend.build`'s schema-init + build for both
   patients (Task 2.2/2.3) → `pageindex_backend.build` for both patients (Task 3.1).
3. On completion, show a summary (page/report/chunk/evidence counts per patient, reusing Task
   1.8's summary logic) and a success toast (`st.toast`).
4. Wrap the whole thing in try/except with a clear `st.error` on failure, and make sure a partial
   failure (e.g., Neo4j unreachable) doesn't leave the button in a stuck/spinning state.

Output: clicking "Rebuild index" in the running Streamlit app re-runs the entire pipeline
end-to-end from inside the UI — the one-click demo reset described in SRS $13's closing note.
```

**🧑‍💻 After the agent finishes this task, YOU must:**
- Do a full manual walkthrough of the running app: switch between Individual/Group, single/multi/all-patient selection, GraphRAG/PageIndex/Compare-both, and click through several citations. This is the final UX acceptance check and is inherently a human judgment task.

---

# PHASE 6 — TESTING & POLISH
## Goal: Confidence the demo won't break live. Maps to SRS $14.

---

## TASK 6.1 — 🤖 AGENT TASK: Consolidate Automated Tests

```
CONTEXT FILE: Read Medical_RAG_System_SRS_Demo.md, SRS $14.

Task: Review every `tests/test_*.py` file written across Tasks 1.2–4.1, ensure they all run
cleanly with a single `pytest` invocation from the project root (fix any import-path issues,
add a `conftest.py`/`pytest.ini` with `pythonpath = .` if needed), and none of them require a
live Neo4j connection or spend real LLM API credits (mock external calls throughout, per each
task's instructions).

Output: `pytest` (no arguments) passes fully and fast (well under a minute), safe to run
repeatedly without cost or external dependencies.
```

---

## TASK 6.2 — 🧑‍💻 YOUR TASK (manual): Execute the Manual Smoke Test Checklist

Context: SRS $14 ("Manual smoke test script").

1. Ask the agent to fill out `tests/manual_checklist.md` (this part IS 🤖 agent-doable — have it
   generate a checklist document listing all 6 use cases from SRS $4.2, each with a checkbox for
   "GraphRAG: citation present & resolves" / "PageIndex: citation present & resolves").
2. **You personally** then run through the live Streamlit app and tick off every box, reading
   each answer against what you know is true in the source PDFs, and paying special attention to:
   - Low-confidence (cursive/handwritten) evidence is visibly flagged red/orange, never shown as
     confidently as typed text (SRS NFR-1).
   - A Group "compare Patient A vs Patient B" answer never blends uncited cross-patient claims
     into one sentence (SRS FR-8.2.3).
   - Re-clicking "Rebuild index" (Task 5.4) twice in a row doesn't duplicate Neo4j nodes/
     relationships or change the PageIndex tree shape (SRS NFR-3) — 🧑‍💻 spot-check via the
     Neo4j browser's `MATCH (n) RETURN count(n)`.
   - Hindi/Devanagari content (consent forms) is preserved and displayed correctly in the
     evidence panel, not mojibake.

This final acceptance pass **cannot be delegated** — it is the human judgment step the whole
"evidence-first" design exists to support.

---

## QUICK REFERENCE — Task Execution Order

```
PHASE 0 — Environment (do first, mostly manual):
  0.1  🧑‍💻 Get LLM API key + online Neo4j credentials
  0.2  🧑‍💻 Place patient_a.pdf / patient_b.pdf
  0.3  🤖 Project scaffold + requirements.txt + folder structure
  0.4  🤝 Install & configure MinerU locally (bilingual EN/HI)
  0.5  🧑‍💻 Fill in .env
  0.6  🤖 Verify Neo4j connectivity

PHASE 1 — Common Core Pipeline:
  1.1  🤖 Patient registry
  1.2  🤖 PDF → page images
  1.3  🤖 OCR provider (MinerU) + bilingual handling
  1.4  🤖 Run full OCR pass + summary
  1.5  🤖 Report boundary splitting
  1.6  🤖 Normalization dictionary + normalizer
  1.7  🤖 Chunking + evidence store
  1.8  🤝 End-to-end Phase 1 smoke test

PHASE 2 — GraphRAG Backend:
  2.1  🤖 LLM client wrapper
  2.2  🤖 Neo4j schema init
  2.3  🤖 Graph build (extraction + MERGE)
  2.4  🤖 Graph retrieval (templates + LLM fallback)

PHASE 3 — PageIndex Backend:
  3.1  🤖 Tree build
  3.2  🤖 Tree retrieval (pure LLM traversal)

PHASE 4 — Orchestration:
  4.1  🤖 Orchestration + generation + grounding check

PHASE 5 — Streamlit UI:
  5.1  🤖 App skeleton + sidebar
  5.2  🤖 Chat interface
  5.3  🤖 Evidence panel
  5.4  🤖 Rebuild-index admin button

PHASE 6 — Testing:
  6.1  🤖 Consolidate automated tests
  6.2  🧑‍💻 Manual smoke test checklist (final human acceptance pass)
```

---

## PRO TIPS FOR USING THESE PROMPTS WITH YOUR CODING AGENT

1. **One task at a time.** Don't combine tasks — each is scoped to be completable and testable in one session.
2. **Always point it at the SRS.** Every prompt already opens with the "read the SRS first" line — don't let the agent improvise architecture beyond what's written there (see SRS $1.2 and $3 for exactly what's excluded and why).
3. **Test before moving on.** After each 🤖 task, run whatever the task's "Output:" line describes. Only proceed once it works.
4. **Do the 🧑‍💻 steps yourself, in order.** Several later tasks assume a manual step already happened (e.g., Task 1.3's manual OCR review before Task 1.5's splitting, Task 1.8's checkpoint before Phase 2 spends LLM credits). Skipping them just moves the failure later and makes it more expensive to debug.
5. **Reference earlier tasks when reporting bugs.** E.g.: *"In Task 2.3, the graph build is creating duplicate Medication nodes — the MERGE key doesn't seem to be matching correctly. Here's the Cypher it generated: [paste]. Here's the mismatch I found in Neo4j browser: [paste]."*
6. **The Hindi/bilingual handling is the highest-risk part.** If MinerU's language switching (Task 0.4) doesn't work well, that risk shows up again at Task 1.3 (OCR), Task 1.5 (splitting — Hindi anchors), Task 1.6 (normalization — Hindi dictionary), and Task 6.2 (final QA). Don't skip the manual review in Task 0.4/1.3 — catching a language-mode bug there is far cheaper than catching it in Task 6.2.
7. **Keep the two backends symmetric.** Whenever you ask the agent to fix something in `graph_backend/retrieve.py`, check whether `pageindex_backend/retrieve.py` has the analogous issue (both were built to the same `facts` contract in SRS $8.1 — bugs in "per-patient tagging for Group mode," for instance, tend to appear in both places).

---

*End of BUILD_GUIDE.*
