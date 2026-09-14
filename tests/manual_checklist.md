# Medical Records RAG Demo — Manual Acceptance & Smoke Test Checklist

> **Purpose**: Human-in-the-loop clinical acceptance verification checklist (SRS §14, Task 6.2).
> Ensure that all 6 core use cases (UC-1 through UC-6) are executed in the live Streamlit UI across both GraphRAG and PageIndex backends, verifying factual accuracy, strict citation grounding, and evidence viewer presentation.

---

## 🚀 Pre-Flight Checklist
- [x] Streamlit web application launched (`streamlit run app.py`).
- [x] Neo4j instance reachable and populated with knowledge graph nodes/relationships.
- [x] PageIndex tree JSON files present under `data/pageindex/`.
- [x] Raw scanned page images available under `data/pages/`.
- [x] Automated test suite passing (`python -m pytest`).

---

## 📋 Use Case Acceptance Matrix (SRS §4.2)

### UC-1: Individual Mode — Longitudinal Health Progress / Lab Trends
- **Query**: `"What is my health progress after multiple reports?"`
- **Target Patient**: `Patient A (R. Sharma)`
- **Expected Content**: Chronological timeline of clinical events, PET-CT metabolic findings, lab values across chemotherapy cycles, and postoperative status.
- [ ] **GraphRAG**: Answer presents chronological progress with grounded citations (`[patient_a__...]`).
- [ ] **GraphRAG**: Clicking citations opens evidence panel with OCR text, page image, and confidence badge.
- [ ] **PageIndex**: Answer presents chronological progress with grounded citations (`[patient_a__...]`).
- [ ] **PageIndex**: Clicking citations opens evidence panel with OCR text, page image, and confidence badge.

---

### UC-2: Individual Mode — Diagnosis & Clinical History
- **Query**: `"What diseases was I suffering from in the last year?"`
- **Target Patient**: `Patient A (R. Sharma)`
- **Expected Content**: Documented carcinoma diagnosis (Invasive Ductal Carcinoma / Infiltrating Duct Carcinoma of breast), histopathology grade, TNM clinical/pathological staging, and PET-CT metabolic findings.
- [ ] **GraphRAG**: Accurate diagnosis list with exact dates/reports cited.
- [ ] **GraphRAG**: All citation IDs resolve to matching evidence records without 404/missing warnings.
- [ ] **PageIndex**: Accurate diagnosis list with exact dates/reports cited.
- [ ] **PageIndex**: All citation IDs resolve to matching evidence records without 404/missing warnings.

---

### UC-3: Individual Mode — Medication & Chemotherapy Regimens
- **Query**: `"What chemotherapy/medications have I received and when?"`
- **Target Patient**: `Patient A (R. Sharma)`
- **Expected Content**: Chemotherapy cycles (e.g. Epirubicin + Cyclophosphamide / EC, Paclitaxel / Taxol), administration dates, dosages, and supportive medications.
- [ ] **GraphRAG**: Specific drug names, administration cycles, and dates cited directly from flowsheets/records.
- [ ] **GraphRAG**: Evidence viewer displays raw OCR text and flowsheet scan.
- [ ] **PageIndex**: Specific drug names, administration cycles, and dates cited directly from flowsheets/records.
- [ ] **PageIndex**: Evidence viewer displays raw OCR text and flowsheet scan.

---

### UC-4: Group Mode — Single Patient Clinical Review
- **Query**: `"Give me the treatment details for Patient B"`
- **Target Patient**: `Patient B (S. Patel)`
- **Expected Content**: Complete treatment summary for Patient B, including surgical resection (MRM / Lumpectomy / WLE), chemotherapy administration, radiation therapy notes, and follow-up lab trends.
- [ ] **GraphRAG**: Comprehensive treatment summary with citations scoped strictly to `patient_b`.
- [ ] **GraphRAG**: No data leakage from `patient_a`.
- [ ] **PageIndex**: Comprehensive treatment summary with citations scoped strictly to `patient_b`.
- [ ] **PageIndex**: No data leakage from `patient_a`.

---

### UC-5: Group Mode — Multi-Patient Comparative Analysis
- **Query**: `"Compare health recovery of Patient A vs Patient B"`
- **Target Patients**: `Patient A (R. Sharma)` AND `Patient B (S. Patel)`
- **Expected Content**: Clear side-by-side or structured per-patient section comparing tumor staging, surgical interventions, chemotherapy tolerability, and post-treatment status.
- [ ] **GraphRAG**: Structured per-patient comparison; NO uncited blending of cross-patient claims into a single sentence (SRS FR-8.2.3).
- [ ] **GraphRAG**: Distinct citations for Patient A (`patient_a__...`) and Patient B (`patient_b__...`).
- [ ] **PageIndex**: Structured per-patient comparison; NO uncited blending of cross-patient claims into a single sentence.
- [ ] **PageIndex**: Distinct citations for Patient A (`patient_a__...`) and Patient B (`patient_b__...`).

---

### UC-6: Group Mode — Cohort Biomarker Query ("All Patients")
- **Query**: `"Which patients have HER2-positive status documented?"`
- **Target Patients**: `All patients` (checkbox selected)
- **Expected Content**: Biomarker evaluation for all patients in cohort with specific IHC / FISH status (e.g. HER2 score 0/1+/2+/3+), citing exact histopathology/IHC reports.
- [ ] **GraphRAG**: Explicitly identifies biomarker status per patient with cited evidence IDs.
- [ ] **GraphRAG**: High confidence scores on typed pathology reports (green badges).
- [ ] **PageIndex**: Explicitly identifies biomarker status per patient with cited evidence IDs.
- [ ] **PageIndex**: High confidence scores on typed pathology reports (green badges).

---

## 🔍 UI & Clinical Safety Checks (SRS §9, §14)
- [ ] **Evidence Confidence Flagging (SRS NFR-1)**:
  - Typed reports display `:green[● High confidence (>=80%)]`.
  - Tabular / mixed reports display `:orange[● Medium confidence (50-79%)]`.
  - Cursive/handwritten sections visibly flagged `:red[● Low confidence (<50%)]`.
- [ ] **Medical Disclaimer Guardrail (SRS D14)**:
  - Diagnostic or clinical queries display standard medical advice disclaimer sentence.
- [ ] **Bilingual / Hindi Content**:
  - Hindi consent forms and patient signatures are preserved in evidence text without mojibake (`data/normalization/hindi_terms.json`).
- [ ] **Admin Rebuild Verification (SRS NFR-3, FR-9.2.1)**:
  - Clicking "🚀 Rebuild Index from data/raw/" in the Admin tab completes end-to-end without crashing.
  - Rebuilding twice does not create duplicate graph nodes or alter PageIndex tree structures.
