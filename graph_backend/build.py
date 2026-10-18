"""
Medical Records RAG Demo — Graph Build & Knowledge Graph Construction (SRS §6.1, §12.1)

Connects to online Neo4j instance, initializes schema constraints, extracts
entity/relation triples from chunks via LLM, and MERGEs them into the knowledge graph.

Tasks:
- Task 2.2: Schema initialization (run_schema_init)
- Task 2.3: Graph construction (extract_triples, write_triples_to_neo4j, build_graph_for_patient)
"""

import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Ensure project root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure .env is loaded
load_dotenv()

from patients import all_patient_ids, get_display_label
from llm_client import chat, parse_json_list
from normalize import canonicalize_lab_name, compute_abnormal_flag, resolve_lab_test_name


# Temporary session Neo4j instance configuration (defaults allow seamless container startup without Azure env config)
TEMP_NEO4J_CONFIG = {
    "uri": os.getenv("TEMP_NEO4J_URI", "neo4j+s://af2857f2.databases.neo4j.io"),
    "username": os.getenv("TEMP_NEO4J_USERNAME", "af2857f2"),
    "password": os.getenv("TEMP_NEO4J_PASSWORD", "nv-uzGLpiAx2_14ead9Cu1ytUwSW0mZnaTzOvF0AojQ"),
    "database": os.getenv("TEMP_NEO4J_DATABASE", "af2857f2"),
}


def get_driver(
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
):
    """
    Creates and returns an authenticated Neo4j driver using provided or .env credentials.
    Supports fallback to neo4j+ssc:// if standard SSL verification fails.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError(
            "neo4j package is required. Run `pip install neo4j`."
        ) from e

    uri = uri or os.getenv("NEO4J_URI")
    username = username or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD")

    if not uri or not password:
        raise ValueError(
            "NEO4J_URI or NEO4J_PASSWORD is not set in environment / .env file."
        )

    # The +ssc schemes encrypt but do not verify the server certificate, so falling
    # back to them silently turns off authentication of the database on a link
    # carrying patient records — and makes a TLS-intercepting proxy indistinguishable
    # from an attacker. The fallback therefore has to be asked for explicitly.
    allow_unverified = (os.getenv("NEO4J_ALLOW_SELF_SIGNED_CERT", "").strip().lower()
                        in ("1", "true", "yes"))

    uris_to_try = [uri]
    if allow_unverified:
        if uri.startswith("neo4j+s://"):
            uris_to_try.append(uri.replace("neo4j+s://", "neo4j+ssc://"))
        elif uri.startswith("bolt+s://"):
            uris_to_try.append(uri.replace("bolt+s://", "bolt+ssc://"))

    last_error: Optional[Exception] = None
    for target_uri in uris_to_try:
        try:
            driver = GraphDatabase.driver(target_uri, auth=(username, password))
            driver.verify_connectivity()
            if target_uri != uri:
                print(
                    "[NEO4J] WARNING: connected with certificate verification DISABLED "
                    f"({target_uri.split('://')[0]}://) because "
                    "NEO4J_ALLOW_SELF_SIGNED_CERT is set. The server's identity is NOT "
                    "verified. Use this only against a trusted network or a local instance."
                )
            return driver
        except Exception as exc:
            last_error = exc

    # The driver buries a cert failure two levels inside an ExceptionGroup under a
    # generic ServiceUnavailable, so search the formatted traceback rather than
    # trying to match a particular nesting shape.
    hint = ""
    if not allow_unverified and uri.startswith(("neo4j+s://", "bolt+s://")):
        import traceback
        detail = "".join(
            traceback.format_exception(type(last_error), last_error, last_error.__traceback__)
        ) if last_error else ""
        if "CERTIFICATE_VERIFY" in detail.upper():
            hint = (
                "\nThe server certificate could not be verified, which usually means a "
                "TLS-intercepting proxy or antivirus sits on the path. Fix the trust store, "
                "or — accepting that the database identity will then NOT be checked — set "
                "NEO4J_ALLOW_SELF_SIGNED_CERT=true."
            )

    raise RuntimeError(
        f"Failed to connect to Neo4j instance at {uri}: {last_error}{hint}"
    ) from last_error


def get_driver_for_patient(patient_id: str):
    """
    Returns appropriate driver: temporary Neo4j instance (af2857f2) for temporary session patients,
    or the default .env instance for baseline patients.
    """
    from patients import is_temp_patient
    if is_temp_patient(patient_id):
        return get_driver(
            uri=TEMP_NEO4J_CONFIG["uri"],
            username=TEMP_NEO4J_CONFIG["username"],
            password=TEMP_NEO4J_CONFIG["password"],
        )
    return get_driver()


def parse_cypher_statements(cypher_text: str) -> List[str]:
    """
    Parses raw Cypher file content into individual executable statements,
    stripping single-line (// and --) and block comments.
    """
    # Remove block comments /* ... */
    cleaned = re.sub(r"/\*.*?\*/", "", cypher_text, flags=re.DOTALL)

    # Process line by line to strip single-line comments
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("--"):
            continue
        # Strip inline comments
        line_no_comment = re.sub(r"(//|--).*$", "", line)
        lines.append(line_no_comment)

    full_text = "\n".join(lines)
    # Split by semicolon and filter empty statements
    statements = [stmt.strip() for stmt in full_text.split(";") if stmt.strip()]
    return statements


def run_schema_init(driver=None, schema_path: Optional[Path | str] = None) -> List[str]:
    """
    Executes schema constraints from schema.cypher against the Neo4j instance (SRS §6.1.2).

    Args:
        driver: Optional existing Neo4j driver instance. If None, creates a temporary one.
        schema_path: Path to schema.cypher. Defaults to graph_backend/schema.cypher.

    Returns:
        List of executed Cypher statements.
    """
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.cypher"
    else:
        schema_path = Path(schema_path)

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        content = f.read()

    statements = parse_cypher_statements(content)
    if not statements:
        print("[WARN] No executable Cypher statements found in schema file.")
        return []

    should_close_driver = False
    if driver is None:
        driver = get_driver()
        should_close_driver = True

    executed_statements: List[str] = []
    try:
        with driver.session() as session:
            print("=== Executing Neo4j Schema Initialization ===")
            for idx, stmt in enumerate(statements, 1):
                compact_stmt = " ".join(stmt.split())
                print(f"[{idx}/{len(statements)}] Executing: {compact_stmt}")
                session.run(stmt)
                executed_statements.append(stmt)
            print(f"=== Schema Initialization Succeeded ({len(executed_statements)} statements) ===")
    finally:
        if should_close_driver and driver is not None:
            driver.close()

    return executed_statements


# ---------- Fixed schema validation sets ----------
# Any LLM-extracted triple whose node/relation type isn't here gets rejected and logged.
VALID_NODE_LABELS = {
    "Patient", "Report", "Diagnosis", "Procedure", "Medication", "Regimen",
    "LabTest", "LabResult", "Staging", "Biomarker",
    "ChemoAdministration", "MedicationAdministration", "TreatmentPlan",
}
VALID_RELATION_TYPES = {
    "STATES_DIAGNOSIS", "UNDERWENT", "ADMINISTERED", "CONTAINS",
    "HAS_RESULT", "OF_TEST", "HAS_STAGING", "HAS_BIOMARKER",
    "SUGGESTS_TEST", "HAS_CHEMO_ADMIN", "HAS_MED_ADMIN", "HAS_TREATMENT_PLAN", "COMPARED_TO",
}


# JSON schema used to constrain local decoding. The enums are the same sets above,
# so a constrained model cannot emit an off-schema relation or label at all —
# object_name is required and non-empty, which also rules out the placeholder
# triples the local model used to pad its reply with until it was truncated.
# Decomposed extraction. Measured failure: asked for everything at once, the local
# model enumerates one or two items from a list and stops — Qwen-7B returned 1 fact
# from a chemo record containing 7, MedGemma missed 4 of 5 premedications and 3 of 5
# biomarkers. "List every drug given, including premedications" is a tractable
# instruction where "extract all clinical triples" is not. One narrow pass per
# category, each with its relation enum restricted so the model cannot drift.
EXTRACTION_PASSES: List[Dict[str, Any]] = [
    {
        "name": "diagnoses",
        "relations": ["STATES_DIAGNOSIS"],
        "labels": ["Report", "Diagnosis"],
        "ask": (
            "EVERY condition, disease or finding this report states the patient HAS. "
            "Skip anything the text negates or calls inconclusive."
        ),
    },
    {
        "name": "procedures",
        "relations": ["UNDERWENT", "SUGGESTS_TEST"],
        "labels": ["Patient", "Report", "Procedure", "LabTest"],
        "ask": (
            "EVERY procedure, scan, biopsy or operation the patient underwent "
            "(relation UNDERWENT), and EVERY test advised or planned but not yet done "
            "(relation SUGGESTS_TEST)."
        ),
    },
    {
        "name": "medications",
        "relations": ["ADMINISTERED", "CONTAINS"],
        "labels": ["Report", "Regimen", "Medication"],
        "ask": (
            "EVERY drug administered or prescribed, INCLUDING premedications and "
            "supportive drugs such as antiemetics and steroids, plus the regimen or "
            "protocol name if one is written. List each drug as its own triple."
        ),
    },
    {
        "name": "labs",
        "relations": ["HAS_RESULT"],
        "labels": ["Report", "LabResult", "LabTest"],
        "ask": (
            "EVERY laboratory test result. object_name is the TEST NAME; put the "
            "number in properties.value. Where a line pairs two tests with two values "
            "(\"Blood Urea/Creatinine 26/1.1\"), emit one triple per test. Skip any "
            "figure whose test name you cannot identify."
        ),
    },
    {
        "name": "staging_biomarkers",
        "relations": ["HAS_STAGING", "HAS_BIOMARKER"],
        "labels": ["Report", "Staging", "Biomarker"],
        "ask": (
            "EVERY biomarker or receptor reported, each as its own triple, with its "
            "polarity in properties.value (positive/negative) exactly as written. "
            "Include a TNM stage ONLY if the text spells one out; never infer one."
        ),
    },
]


def build_pass_schema(pass_spec: Dict[str, Any], max_triples: int) -> Dict[str, Any]:
    """Extraction schema narrowed to one category's relations and labels."""
    schema = json.loads(json.dumps(EXTRACTION_JSON_SCHEMA))
    item = schema["properties"]["triples"]["items"]
    item["properties"]["relation"]["enum"] = sorted(pass_spec["relations"])
    item["properties"]["subject_label"]["enum"] = sorted(pass_spec["labels"])
    item["properties"]["object_label"]["enum"] = sorted(pass_spec["labels"])
    schema["properties"]["triples"]["maxItems"] = max(1, int(max_triples))
    return schema


def build_extraction_schema(max_triples: int) -> Dict[str, Any]:
    """Extraction schema with the array bounded to `max_triples` items."""
    schema = json.loads(json.dumps(EXTRACTION_JSON_SCHEMA))
    schema["properties"]["triples"]["maxItems"] = max(1, int(max_triples))
    return schema


def triple_budget_for(ocr_confidence: Optional[float], text_length: int = 0) -> int:
    """
    How many facts a chunk is allowed to yield.

    A flat cap is filled regardless of how much the text supports: on a page scoring
    0.936 OCR confidence, containing one legible fact, the model produced twelve.
    Unreadable text cannot support many facts, so the budget scales with OCR
    confidence and with how much text there is to draw on. This bounds invention in
    proportion to evidence instead of trusting the model to abstain.
    """
    ceiling = int(os.getenv("EXTRACTION_MAX_TRIPLES") or 12)
    conf = ocr_confidence if isinstance(ocr_confidence, (int, float)) else None

    if conf is None:
        budget = ceiling
    elif conf >= 0.99:
        budget = ceiling
    elif conf >= 0.975:
        budget = max(1, int(ceiling * 0.75))
    elif conf >= 0.95:
        budget = max(1, int(ceiling * 0.4))
    else:
        budget = max(1, int(ceiling * 0.25))

    # No length-based cap. Text length is a bad proxy for fact density: a flowsheet
    # line like "Hb 11.0, WBC 10200, Plt 522000" is 30 characters and carries three
    # facts, so any per-character divisor that constrains noise also truncates dense
    # clean text. OCR confidence is the signal that actually distinguishes the two.
    return budget


EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            # An unbounded array is a repetition trap: under greedy sampling the model
            # kept emitting items until max_tokens, producing 72 "facts" for a chunk
            # containing 3 (68 of them invented) and taking 11 minutes. The bound makes
            # the grammar close the array; minItems 0 keeps "nothing here" expressible,
            # which chunk 6 of the gold set requires.
            "minItems": 0,
            "maxItems": int(os.getenv("EXTRACTION_MAX_TRIPLES") or 12),
            "items": {
                "type": "object",
                "properties": {
                    "subject_label": {"type": "string", "enum": sorted(VALID_NODE_LABELS)},
                    "subject_name": {"type": "string"},
                    "relation": {"type": "string", "enum": sorted(VALID_RELATION_TYPES)},
                    "object_label": {"type": "string", "enum": sorted(VALID_NODE_LABELS)},
                    "object_name": {"type": "string", "minLength": 1},
                    "properties": {"type": "object"},
                },
                "required": [
                    "subject_label", "subject_name", "relation",
                    "object_label", "object_name",
                ],
            },
        }
    },
    "required": ["triples"],
}


# This prompt text is measured. eval/results/FINAL_shipping_config.json was produced
# by SHA f6dfcb406d35 of this string; changing it invalidates that comparison, so run
# eval/run_extraction_eval.py again after any edit here.
#
# Tried and reverted: scoping rule 5b to diagnoses only, so that a procedure performed
# or a test advised in an inconclusive report would still be extracted. It read as an
# obvious improvement and measured as a clear loss — recall 0.423 -> 0.231, F1
# 0.400 -> 0.250 — for one negation error removed, and it did not recover the chunk it
# was written for. Prompt edits here have non-local effects: that one also changed the
# output of two chunks it had nothing to do with.
EXTRACTION_SYSTEM_PROMPT = """You are an expert clinical knowledge graph extraction assistant for general and specialized medical records.
Your task is to extract structured medical entities and relationships from clinical report text into a strict knowledge graph schema.

ALLOWED NODE LABELS:
- Patient: {patient_id}
- Report: {report_id}
- Diagnosis: {canonical_name} (e.g., 'Metastatic Breast Cancer', 'Essential Hypertension', 'Type 2 Diabetes Mellitus', 'Coronary Artery Disease', 'Pneumonia')
- Procedure: {canonical_name} (e.g., 'Modified Radical Mastectomy', 'Coronary Angiography', 'Echocardiography', 'Hemodialysis', 'CECT Chest Abdomen Pelvis')
- Medication: {canonical_name} (e.g., 'Paclitaxel', 'Metformin', 'Atorvastatin', 'Aspirin', 'Doxorubicin', 'Ceftriaxone')
- Regimen: {canonical_name} (e.g., 'AC Regimen', 'Dual Antiplatelet Therapy', 'Anti-Tubercular Therapy')
- LabTest: {canonical_name} (e.g., 'Hemoglobin', 'Serum Creatinine', 'Glycated Hemoglobin (HbA1c)', 'Cardiac Troponin I', 'Platelet Count', 'SGOT')
- LabResult: {value, unit, date, result_date}
- Staging: {t, n, m, date, stage}
- Biomarker: {marker, value, date} (e.g., marker='ER', value='8/8' or marker='LVEF', value='55%' or marker='HER2', value='3+')
- ChemoAdministration: {cycle_number, date, regimen, medications}
- MedicationAdministration: {cycle_number, date, regimen, medications}
- TreatmentPlan: {regimen, planned_cycles, confidence_note}

ALLOWED RELATIONSHIPS:
- (Report)-[:STATES_DIAGNOSIS]->(Diagnosis)
- (Patient)-[:UNDERWENT {date}]->(Procedure)
- (Report)-[:ADMINISTERED {dose, cycle}]->(Medication)
- (Report)-[:ADMINISTERED]->(Regimen)
- (Regimen)-[:CONTAINS]->(Medication)
- (Report)-[:HAS_RESULT]->(LabResult)  [and (LabResult)-[:OF_TEST]->(LabTest)]
- (Report)-[:HAS_STAGING]->(Staging)
- (Report)-[:HAS_BIOMARKER]->(Biomarker)
- (Report)-[:SUGGESTS_TEST {date}]->(LabTest or Procedure)  — for tests that were SUGGESTED/ADVISED but not necessarily performed. Look for language like "advised", "to be done", "kept for", "plan for", "suggested", "recommended".
- (Report)-[:HAS_CHEMO_ADMIN]->(ChemoAdministration)  — for explicit chemotherapy cycle administrations with cycle numbers. Extract cycle_number from CYCLE/DAY fields or mentions like "Cycle 3", "C3D1", "#3".
- (Report)-[:HAS_MED_ADMIN]->(MedicationAdministration)  — for scheduled drug courses, dialysis sessions, or radiation fractions with sequence/cycle numbers.
- (Report)-[:HAS_TREATMENT_PLAN]->(TreatmentPlan)  — for treatment plans mentioning total planned cycles or courses, e.g., "4EC → 4T", "plan: EC #4", "6 cycles of AC", "12 hemodialysis sessions". Extract planned_cycles as a number.
- (Report)-[:COMPARED_TO {comparison_text}]->(Report)  — for radiology reports that state comparison with a previous scan, e.g., "as compared to previous scan dated...", "no significant change compared to prior study". Capture the comparison statement verbatim.

SPECIAL INSTRUCTIONS FOR SUGGESTED TESTS:
When text says a test was "advised", "to be done", "suggested", "recommended", "planned", or "kept for", extract it as SUGGESTS_TEST (not HAS_RESULT). Only extract HAS_RESULT when the test was actually PERFORMED and a result value exists.

SPECIAL INSTRUCTIONS FOR CYCLES & SESSIONS:
When text contains cycle or session information ("Cycle 3", "C3D1", "CYCLE/DAY: 3/1", "#3", "Session 4"), extract an administration node with an explicit cycle_number integer. Do NOT make the LLM count or infer cycle numbers — only extract what is explicitly stated.

SPECIAL INSTRUCTIONS FOR LAB RESULTS:
Include result_date in properties if a specific date for the result is available (distinct from the report date). This is especially important for flowsheet tables where each column has its own date.

INSTRUCTIONS:
1. Extract ONLY facts explicitly stated in the chunk text and pre-normalized entities.
2. Use canonical names where pre-normalized entities are supplied.
3. If no relevant clinical entities or relations exist in the chunk, return an empty list [].
3a. OMIT any relation you did not find. Never emit a placeholder triple with a null,
    empty or "None" object_name — emit only triples carrying a real documented fact.
3b. Reply with JSON ONLY. No reasoning, commentary, explanation or markdown before
    or after it.
4a. Return an object of the form {"triples": [ ... ]}.

NEGATION AND UNCERTAINTY — READ BEFORE EXTRACTING
A clinical report records what was looked for as much as what was found. Reporting a
ruled-out finding as a diagnosis is worse than missing it: it puts a condition the
patient does not have into their record.
5a. NEVER extract a finding that the text negates. "No ascites", "no pericardial
    effusion", "no evidence of metastasis", "ruled out", "within normal limits",
    "unremarkable", "shows no abnormality" all mean the finding is ABSENT. Omit it.
5b. NEVER turn an inconclusive or equivocal result into a positive one. "Inconclusive",
    "indeterminate", "?significance", "suspicious", "cannot be excluded",
    "awaited" are not diagnoses. Omit them.
5c. Extract ONE clinical entity per triple. "Metastatic breast cancer with pleural
    effusion" is TWO facts: a diagnosis of metastatic breast cancer and a diagnosis of
    pleural effusion. Never join two findings with "and" or "with" in one object_name.
5d. object_name must be the NAME of a thing, never a measurement. "11.0", "26/1.1",
    "522000" and "9.3/4" are values. For a lab result, object_name is the test name
    ("Platelet Count") and the number belongs in properties.value.
5e. Choose the relation that matches the fact, not the first plausible one. A condition
    the report states is STATES_DIAGNOSIS. A procedure the patient had is UNDERWENT. A
    drug given is ADMINISTERED. A test advised but not yet done is SUGGESTS_TEST.
    HAS_TREATMENT_PLAN is only for an explicitly documented future plan.

WORKED EXAMPLE
Chunk text: "CECT Chest: Metastatic carcinoma right breast with ascites. Advised LFT."
Correct reply:
{"triples": [
  {"subject_label": "Report", "subject_name": "report_0001", "relation": "STATES_DIAGNOSIS",
   "object_label": "Diagnosis", "object_name": "Metastatic Breast Cancer", "properties": {}},
  {"subject_label": "Report", "subject_name": "report_0001", "relation": "STATES_DIAGNOSIS",
   "object_label": "Diagnosis", "object_name": "Ascites", "properties": {}},
  {"subject_label": "Patient", "subject_name": "patient_a", "relation": "UNDERWENT",
   "object_label": "Procedure", "object_name": "CECT Chest", "properties": {}},
  {"subject_label": "Report", "subject_name": "report_0001", "relation": "SUGGESTS_TEST",
   "object_label": "LabTest", "object_name": "Liver Function Test", "properties": {}}
]}
4. Output MUST be valid JSON: a list of objects with the exact schema:
[
  {
    "subject_label": "Report" | "Patient" | "Regimen" | "LabResult",
    "subject_name": "...",
    "relation": "STATES_DIAGNOSIS" | "UNDERWENT" | "ADMINISTERED" | "CONTAINS" | "HAS_RESULT" | "HAS_STAGING" | "HAS_BIOMARKER" | "SUGGESTS_TEST" | "HAS_CHEMO_ADMIN" | "HAS_MED_ADMIN" | "HAS_TREATMENT_PLAN" | "COMPARED_TO",
    "object_label": "Diagnosis" | "Procedure" | "Medication" | "Regimen" | "LabResult" | "Staging" | "Biomarker" | "ChemoAdministration" | "MedicationAdministration" | "TreatmentPlan" | "LabTest" | "Report",
    "object_name": "...",
    "properties": { ... }
  }
]
"""


def _extraction_system_prompt() -> str:
    """System prompt for extraction, with the worked example optionally removed."""
    prompt = EXTRACTION_SYSTEM_PROMPT
    if os.getenv("EXTRACTION_FEW_SHOT", "false").strip().lower() not in ("1", "true", "yes"):
        # Measured: the worked example cost recall rather than helping it
        # (Qwen-7B recall 0.308 -> 0.423 with it removed). A short example anchors
        # the model to a short answer. Off by default; the switch keeps it measurable.
        prompt = prompt.split("WORKED EXAMPLE")[0].rstrip()
    return prompt


def _unwrap_triples(items: List[Any]) -> List[Dict[str, Any]]:
    """
    Accept either shape a model replies with: the {"triples": [...]} object that
    constrained decoding produces, or a bare array from an unconstrained reply.
    """
    if len(items) == 1 and isinstance(items[0], dict) and "triples" in items[0]:
        inner = items[0].get("triples")
        return [t for t in inner if isinstance(t, dict)] if isinstance(inner, list) else []
    return [t for t in items if isinstance(t, dict)]


def _validate_triples(
    candidates: List[Dict[str, Any]],
    report_id: str,
    budget: int,
) -> List[Dict[str, Any]]:
    """
    Drop off-schema, empty and duplicate triples, then apply the evidence budget.
    Shared by the single-pass and decomposed paths so both enforce one contract.
    """
    valid_triples: List[Dict[str, Any]] = []
    seen = set()

    for item in candidates:
        if not isinstance(item, dict):
            continue
        relation = item.get("relation", "")
        subject_label = item.get("subject_label", "")
        object_label = item.get("object_label", "")
        object_name = str(item.get("object_name") or "").strip()

        # Placeholder triples for relations the model did not find carry no fact.
        if not object_name and not item.get("properties"):
            continue

        if relation not in VALID_RELATION_TYPES:
            print(f"  [SCHEMA] Rejected off-schema relation {relation!r} in {report_id}")
            continue
        if subject_label and subject_label not in VALID_NODE_LABELS:
            print(f"  [SCHEMA] Rejected off-schema subject label {subject_label!r} in {report_id}")
            continue
        if object_label and object_label not in VALID_NODE_LABELS:
            print(f"  [SCHEMA] Rejected off-schema object label {object_label!r} in {report_id}")
            continue

        # The local model repeats itself: one chunk yielded "Blood" four times.
        key = (relation.upper(), object_name.lower())
        if key in seen:
            continue
        seen.add(key)

        valid_triples.append(item)

    if len(valid_triples) > budget:
        print(
            f"  [BUDGET] {report_id}: {len(valid_triples)} facts exceed the evidence "
            f"budget of {budget}; keeping the first {budget}."
        )
        valid_triples = valid_triples[:budget]

    return valid_triples


def extract_triples(
    chunk_text: str,
    normalized_entities: List[Dict[str, Any]],
    report_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extracts structured knowledge graph triples from a chunk using the LLM (SRS §6.1.3).
    Constrained strictly to the fixed schema. Rejects any triple with off-schema labels/relations.
    """
    patient_id = report_meta.get("patient_id", "patient")
    report_id = report_meta.get("report_id", "report")
    report_type = report_meta.get("report_type", "CLINICAL_REPORT")
    report_date = report_meta.get("report_date", "")

    budget = triple_budget_for(report_meta.get("confidence"), len(chunk_text or ""))

    # Format normalized entities for context
    norm_summary = []
    for ent in normalized_entities:
        norm_summary.append(
            f"- '{ent.get('raw_text')}': canonical='{ent.get('normalized_term')}', type={ent.get('entity_type')}"
        )
    norm_context_str = "\n".join(norm_summary) if norm_summary else "(None)"

    def _prompt_for(task_line: str) -> str:
        return (
            f"REPORT METADATA:\n"
            f"- Patient ID: {patient_id}\n"
            f"- Report ID: {report_id}\n"
            f"- Report Type: {report_type}\n"
            f"- Report Date: {report_date}\n\n"
            f"PRE-NORMALIZED ENTITIES IN CHUNK:\n"
            f"{norm_context_str}\n\n"
            f"RAW CHUNK TEXT:\n"
            f"\"\"\"\n{chunk_text[:3500]}\n\"\"\"\n\n"
            f"{task_line}"
        )

    prompt = _prompt_for(
        "Extract all clinical triples matching the schema. Return ONLY valid JSON list of triples."
    )

    # Permanent patients are extracted with the cloud model; the ephemeral upload
    # sandbox stays on the local model so an uploaded report never leaves the host.
    from patients import is_temp_patient
    use_cloud = not is_temp_patient(patient_id)

    # One narrow pass per category. MEASURED WORSE, so off by default; kept because
    # the result is counter-intuitive enough to be worth re-testing on another model.
    #
    #                    single-pass F1   decomposed F1   unsupported facts
    #   Qwen2.5-7B           0.400           0.339           2 -> 8
    #   MedGemma-4B          0.364           0.194           5 -> 18
    #
    # Asking "extract EVERY drug" per category pressures the model to produce
    # something in each category whether or not the chunk contains one. The
    # clearest sign: a 126-character fragment of pure OCR noise, which the
    # single-pass run correctly left empty, yielded 3 invented facts per-category.
    # Abstention is a capability that per-category prompting destroys.
    decompose = (
        not use_cloud
        and os.getenv("EXTRACTION_DECOMPOSE", "false").strip().lower() in ("1", "true", "yes")
    )
    if decompose:
        collected: List[Dict[str, Any]] = []
        for spec in EXTRACTION_PASSES:
            pass_budget = max(1, min(budget, 6))
            try:
                reply = chat(
                    messages=[{"role": "user", "content": _prompt_for(
                        f"Extract ONLY the following, and nothing else:\n{spec['ask']}\n"
                        f"Allowed relations: {', '.join(spec['relations'])}.\n"
                        f"If the chunk contains none, return {{\"triples\": []}}."
                    )}],
                    system=_extraction_system_prompt(),
                    task="build",
                    use_external=False,
                    json_schema=build_pass_schema(spec, pass_budget),
                )
            except Exception as exc:
                print(f"  [WARN] {spec['name']} pass failed for {report_id}: {exc}")
                continue

            kept = _unwrap_triples(parse_json_list(reply))
            print(f"  [{spec['name']}] {len(kept)} candidate(s) for {report_id}")
            collected.extend(kept)

        if collected:
            return _validate_triples(collected, report_id, budget)
        print(f"  [WARN] No parseable triples in any pass for {report_id}")
        return []

    try:
        response_text = chat(
            messages=[{"role": "user", "content": prompt}],
            system=_extraction_system_prompt(),
            task="build",
            use_external=use_cloud,
            json_schema=build_extraction_schema(budget),
        )
        if not response_text or not response_text.strip():
            return []

        parsed = _unwrap_triples(parse_json_list(response_text))

        # Not retried on an empty reply. A grammar does make {"triples": []} a valid
        # complete answer, and one clean chunk does come back empty — but an
        # unconstrained retry was measured and recovered nothing on it (identical
        # recall 0.423 / F1 0.400, one extra model call per empty chunk). The cause is
        # upstream: the model suppressing a whole chunk, not the grammar.
        if not parsed:
            print(f"  [WARN] No parseable triples in reply for {report_id}")
            return []

        return _validate_triples(parsed, report_id, budget)

    except Exception as exc:
        print(f"  [WARN] Triple extraction failed for {report_id}: {exc}")
        return []


def write_triples_to_neo4j(
    driver,
    patient_id: str,
    report_meta: Dict[str, Any],
    triples: List[Dict[str, Any]],
    evidence_id: str,
    confidence: float,
) -> int:
    """
    Writes extracted triples to Neo4j using idempotent Cypher MERGE statements (SRS §6.1.4).
    Attaches evidence_id and confidence to every relationship.
    Canonicalizes lab/medication names before MERGE to prevent duplicate nodes.
    Computes abnormal_flag on LabResult nodes against static reference ranges.
    """
    raw_report_id = report_meta.get("report_id", "")
    # Scope report_id per patient to guarantee partition isolation
    report_id = f"{patient_id}__{raw_report_id}" if not raw_report_id.startswith(f"{patient_id}__") else raw_report_id
    report_type = report_meta.get("report_type", "")
    report_date = report_meta.get("report_date") or ""
    display_label = get_display_label(patient_id)

    written_count = 0

    with driver.session() as session:
        # 1. Base Patient, Report, and HAS_REPORT relationship
        session.run(
            """
            MERGE (p:Patient {patient_id: $patient_id})
            ON CREATE SET p.display_label = $display_label
            MERGE (r:Report {report_id: $report_id})
            ON CREATE SET r.patient_id = $patient_id,
                          r.raw_report_id = $raw_report_id,
                          r.report_type = $report_type,
                          r.report_date = $report_date
            MERGE (p)-[rel:HAS_REPORT]->(r)
            SET rel.evidence_id = $evidence_id, rel.confidence = $confidence
            """,
            patient_id=patient_id,
            display_label=display_label,
            report_id=report_id,
            raw_report_id=raw_report_id,
            report_type=report_type,
            report_date=report_date,
            evidence_id=evidence_id,
            confidence=confidence,
        )

        for triple in triples:
            relation = triple.get("relation")
            obj_label = triple.get("object_label", "")
            obj_name = triple.get("object_name", "")
            props = triple.get("properties") or {}

            if not obj_name and not props:
                continue

            if relation == "STATES_DIAGNOSIS":
                canonical = str(obj_name).strip()
                if canonical:
                    session.run(
                        """
                        MERGE (r:Report {report_id: $report_id})
                        MERGE (d:Diagnosis {canonical_name: $canonical_name})
                        MERGE (r)-[rel:STATES_DIAGNOSIS]->(d)
                        SET rel.evidence_id = $evidence_id, rel.confidence = $confidence
                        """,
                        report_id=report_id,
                        canonical_name=canonical,
                        evidence_id=evidence_id,
                        confidence=confidence,
                    )
                    written_count += 1

            elif relation == "UNDERWENT":
                # Canonicalize procedure name
                canonical = canonicalize_lab_name(str(obj_name).strip())
                proc_date = props.get("date") or report_date or ""
                if canonical:
                    session.run(
                        """
                        MERGE (p:Patient {patient_id: $patient_id})
                        MERGE (pr:Procedure {canonical_name: $canonical_name})
                        MERGE (p)-[rel:UNDERWENT]->(pr)
                        SET rel.date = $proc_date,
                            rel.evidence_id = $evidence_id,
                            rel.confidence = $confidence
                        """,
                        patient_id=patient_id,
                        canonical_name=canonical,
                        proc_date=proc_date,
                        evidence_id=evidence_id,
                        confidence=confidence,
                    )
                    written_count += 1

            elif relation == "ADMINISTERED":
                # Canonicalize medication/regimen name
                canonical = canonicalize_lab_name(str(obj_name).strip())
                dose = props.get("dose", "")
                cycle = props.get("cycle", "")
                if obj_label == "Regimen" or "regimen" in canonical.lower():
                    session.run(
                        """
                        MERGE (r:Report {report_id: $report_id})
                        MERGE (reg:Regimen {canonical_name: $canonical_name})
                        MERGE (r)-[rel:ADMINISTERED]->(reg)
                        SET rel.evidence_id = $evidence_id, rel.confidence = $confidence
                        """,
                        report_id=report_id,
                        canonical_name=canonical,
                        evidence_id=evidence_id,
                        confidence=confidence,
                    )
                    written_count += 1
                else:
                    session.run(
                        """
                        MERGE (r:Report {report_id: $report_id})
                        MERGE (m:Medication {canonical_name: $canonical_name})
                        MERGE (r)-[rel:ADMINISTERED]->(m)
                        SET rel.dose = $dose,
                            rel.cycle = $cycle,
                            rel.evidence_id = $evidence_id,
                            rel.confidence = $confidence
                        """,
                        report_id=report_id,
                        canonical_name=canonical,
                        dose=dose,
                        cycle=cycle,
                        evidence_id=evidence_id,
                        confidence=confidence,
                    )
                    written_count += 1

            elif relation == "CONTAINS":
                # Canonicalize both regimen and medication names
                reg_name = canonicalize_lab_name(str(triple.get("subject_name", "")).strip())
                med_name = canonicalize_lab_name(str(obj_name).strip())
                if reg_name and med_name:
                    session.run(
                        """
                        MERGE (reg:Regimen {canonical_name: $reg_name})
                        MERGE (m:Medication {canonical_name: $med_name})
                        MERGE (reg)-[rel:CONTAINS]->(m)
                        SET rel.evidence_id = $evidence_id, rel.confidence = $confidence
                        """,
                        reg_name=reg_name,
                        med_name=med_name,
                        evidence_id=evidence_id,
                        confidence=confidence,
                    )
                    written_count += 1

            elif relation == "HAS_RESULT":
                raw_test_name = str(obj_name or props.get("test_name", "")).strip()
                # Reject values/units/row-ids the extractor puts in the test-name slot, and
                # merge spelling variants so one test keeps one node (and therefore one trend).
                test_name = resolve_lab_test_name(raw_test_name)
                if raw_test_name and not test_name:
                    print(f"  [SKIP] Lab result discarded: {raw_test_name!r} is not a test name")
                val = str(props.get("value", "")).strip()
                unit = str(props.get("unit", "")).strip()
                # Use result_date if available, fall back to date, then report_date
                res_date = str(props.get("result_date") or props.get("date") or report_date or "").strip()

                # Compute abnormal_flag deterministically against reference ranges
                abnormal_flag = compute_abnormal_flag(test_name, val) or ""

                if test_name:
                    session.run(
                        """
                        MERGE (r:Report {report_id: $report_id})
                        CREATE (lr:LabResult {value: $value, unit: $unit, date: $date,
                                             result_date: $result_date, abnormal_flag: $abnormal_flag})
                        MERGE (lt:LabTest {canonical_name: $test_name})
                        MERGE (r)-[rel1:HAS_RESULT]->(lr)
                        SET rel1.evidence_id = $evidence_id, rel1.confidence = $confidence
                        MERGE (lr)-[rel2:OF_TEST]->(lt)
                        SET rel2.evidence_id = $evidence_id, rel2.confidence = $confidence
                        """,
                        report_id=report_id,
                        value=val,
                        unit=unit,
                        date=res_date,
                        result_date=res_date,
                        abnormal_flag=abnormal_flag,
                        test_name=test_name,
                        evidence_id=evidence_id,
                        confidence=confidence,
                    )
                    written_count += 1

            elif relation == "HAS_STAGING":
                t_val = str(props.get("t") or obj_name or "").strip()
                n_val = str(props.get("n", "")).strip()
                m_val = str(props.get("m", "")).strip()
                st_date = str(props.get("date") or report_date or "").strip()
                session.run(
                    """
                    MERGE (r:Report {report_id: $report_id})
                    CREATE (st:Staging {t: $t, n: $n, m: $m, date: $date})
                    MERGE (r)-[rel:HAS_STAGING]->(st)
                    SET rel.evidence_id = $evidence_id, rel.confidence = $confidence
                    """,
                    report_id=report_id,
                    t=t_val,
                    n=n_val,
                    m=m_val,
                    date=st_date,
                    evidence_id=evidence_id,
                    confidence=confidence,
                )
                written_count += 1

            elif relation == "HAS_BIOMARKER":
                marker = str(props.get("marker") or triple.get("subject_name") or obj_name or "").strip()
                val = str(props.get("value") or obj_name or "").strip()
                bm_date = str(props.get("date") or report_date or "").strip()
                session.run(
                    """
                    MERGE (r:Report {report_id: $report_id})
                    CREATE (b:Biomarker {marker: $marker, value: $val, date: $date})
                    MERGE (r)-[rel:HAS_BIOMARKER]->(b)
                    SET rel.evidence_id = $evidence_id, rel.confidence = $confidence
                    """,
                    report_id=report_id,
                    marker=marker,
                    val=val,
                    date=bm_date,
                    evidence_id=evidence_id,
                    confidence=confidence,
                )
                written_count += 1

            elif relation == "SUGGESTS_TEST":
                # Test was SUGGESTED/ADVISED but not necessarily performed
                raw_test = str(obj_name).strip()
                canonical_test = resolve_lab_test_name(raw_test) or canonicalize_lab_name(raw_test)
                suggest_date = str(props.get("date") or report_date or "").strip()
                if canonical_test:
                    # Determine target node label
                    if obj_label == "Procedure":
                        session.run(
                            """
                            MERGE (r:Report {report_id: $report_id})
                            MERGE (pr:Procedure {canonical_name: $canonical_name})
                            MERGE (r)-[rel:SUGGESTS_TEST]->(pr)
                            SET rel.date = $date,
                                rel.evidence_id = $evidence_id,
                                rel.confidence = $confidence
                            """,
                            report_id=report_id,
                            canonical_name=canonical_test,
                            date=suggest_date,
                            evidence_id=evidence_id,
                            confidence=confidence,
                        )
                    else:
                        session.run(
                            """
                            MERGE (r:Report {report_id: $report_id})
                            MERGE (lt:LabTest {canonical_name: $canonical_name})
                            MERGE (r)-[rel:SUGGESTS_TEST]->(lt)
                            SET rel.date = $date,
                                rel.evidence_id = $evidence_id,
                                rel.confidence = $confidence
                            """,
                            report_id=report_id,
                            canonical_name=canonical_test,
                            date=suggest_date,
                            evidence_id=evidence_id,
                            confidence=confidence,
                        )
                    written_count += 1

            elif relation in ("HAS_CHEMO_ADMIN", "HAS_MED_ADMIN"):
                # Explicit chemotherapy or medication cycle/course administration with cycle number
                cycle_number = props.get("cycle_number")
                admin_date = str(props.get("date") or report_date or "").strip()
                regimen_name = canonicalize_lab_name(str(props.get("regimen") or obj_name or "").strip())
                medications = props.get("medications", [])
                
                if relation == "HAS_MED_ADMIN" or obj_label == "MedicationAdministration":
                    admin_id = f"{report_id}__med_{cycle_number or 'unknown'}_{admin_date}"
                    label_name = "MedicationAdministration"
                    rel_name = "HAS_MED_ADMIN"
                else:
                    admin_id = f"{report_id}__chemo_{cycle_number or 'unknown'}_{admin_date}"
                    label_name = "ChemoAdministration"
                    rel_name = "HAS_CHEMO_ADMIN"

                if cycle_number is not None:
                    try:
                        cycle_num_int = int(cycle_number)
                    except (ValueError, TypeError):
                        cycle_num_int = -1
                else:
                    cycle_num_int = -1

                session.run(
                    f"""
                    MERGE (r:Report {{report_id: $report_id}})
                    MERGE (admin:{label_name} {{admin_id: $admin_id}})
                    ON CREATE SET admin.cycle_number = $cycle_number,
                                  admin.date = $date,
                                  admin.regimen = $regimen,
                                  admin.medications = $medications
                    MERGE (r)-[rel:{rel_name}]->(admin)
                    SET rel.evidence_id = $evidence_id, rel.confidence = $confidence
                    """,
                    report_id=report_id,
                    admin_id=admin_id,
                    cycle_number=cycle_num_int,
                    date=admin_date,
                    regimen=regimen_name,
                    medications=medications if isinstance(medications, list) else [str(medications)],
                    evidence_id=evidence_id,
                    confidence=confidence,
                )
                written_count += 1

            elif relation == "HAS_TREATMENT_PLAN":
                # Treatment plan with planned cycle count (LLM-extracted from prose)
                regimen_name = canonicalize_lab_name(str(props.get("regimen") or obj_name or "").strip())
                planned_cycles = props.get("planned_cycles")
                confidence_note = str(props.get("confidence_note", "inferred from prose")).strip()
                plan_id = f"{patient_id}__plan_{regimen_name}"

                if planned_cycles is not None:
                    try:
                        planned_int = int(planned_cycles)
                    except (ValueError, TypeError):
                        planned_int = -1
                else:
                    planned_int = -1

                # Lower confidence since this is inferred from prose, not a structured field
                plan_confidence = min(confidence, 0.7)
                session.run(
                    """
                    MERGE (r:Report {report_id: $report_id})
                    MERGE (tp:TreatmentPlan {plan_id: $plan_id})
                    ON CREATE SET tp.regimen = $regimen,
                                  tp.planned_cycles = $planned_cycles,
                                  tp.confidence_note = $confidence_note
                    MERGE (r)-[rel:HAS_TREATMENT_PLAN]->(tp)
                    SET rel.evidence_id = $evidence_id, rel.confidence = $confidence
                    """,
                    report_id=report_id,
                    plan_id=plan_id,
                    regimen=regimen_name,
                    planned_cycles=planned_int,
                    confidence_note=confidence_note,
                    evidence_id=evidence_id,
                    confidence=plan_confidence,
                )
                written_count += 1

            elif relation == "COMPARED_TO":
                # Radiology comparison between reports
                comparison_text = str(props.get("comparison_text") or obj_name or "").strip()
                target_report_ref = str(props.get("target_report") or "").strip()
                if comparison_text:
                    session.run(
                        """
                        MERGE (r:Report {report_id: $report_id})
                        SET r.comparison_text = $comparison_text,
                            r.compared_to_ref = $target_report_ref
                        """,
                        report_id=report_id,
                        comparison_text=comparison_text,
                        target_report_ref=target_report_ref,
                    )
                    written_count += 1

    return written_count


def build_graph_for_patient(
    patient_id: str,
    driver=None,
    data_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Constructs the knowledge graph for a patient sequentially (SRS FR-6.1.5).
    Reads chunks from data/reports/{patient_id}_chunks.json.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    chunks_file = data_dir / "reports" / f"{patient_id}_chunks.json"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found at: {chunks_file}")

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    should_close_driver = False
    if driver is None:
        driver = get_driver()
        should_close_driver = True

    stats = {
        "patient_id": patient_id,
        "total_chunks": len(chunks_data),
        "triples_extracted": 0,
        "triples_written": 0,
    }

    print(f"\n--- Building Knowledge Graph for {patient_id} ({len(chunks_data)} chunks) ---")
    try:
        for idx, chunk in enumerate(chunks_data, 1):
            ev = chunk.get("evidence_record", {})
            norm_entities = chunk.get("normalized_entities", [])
            evidence_id = ev.get("evidence_id", f"{patient_id}_chunk_{idx}")
            raw_text = ev.get("raw_text", "")
            confidence = float(ev.get("confidence", 0.9))

            report_meta = {
                "patient_id": patient_id,
                "report_id": ev.get("report_id", ""),
                "report_type": ev.get("report_type", ""),
                "report_date": ev.get("report_date", ""),
            }

            print(f"[{idx}/{len(chunks_data)}] Extracting triples for {evidence_id} ...", end=" ", flush=True)
            triples = extract_triples(raw_text, norm_entities, report_meta)
            stats["triples_extracted"] += len(triples)

            written = write_triples_to_neo4j(
                driver=driver,
                patient_id=patient_id,
                report_meta=report_meta,
                triples=triples,
                evidence_id=evidence_id,
                confidence=confidence,
            )
            stats["triples_written"] += written
            print(f"extracted: {len(triples)}, written: {written}")

            # Short delay between chunks for rate limiting
            time.sleep(0.2)

    finally:
        if should_close_driver and driver is not None:
            driver.close()

    print(f"Finished {patient_id}: {stats['triples_extracted']} triples extracted, {stats['triples_written']} written.")
    return stats


def build_all_patients(driver=None) -> Dict[str, Dict[str, Any]]:
    """
    Initializes schema constraints and builds knowledge graph for all registered patients (SRS §6.1).
    """
    should_close_driver = False
    if driver is None:
        driver = get_driver()
        should_close_driver = True

    all_stats: Dict[str, Dict[str, Any]] = {}

    try:
        # Step 1: Ensure constraints exist
        run_schema_init(driver=driver)

        # Step 2: Build graph for each patient sequentially
        for p_id in all_patient_ids():
            stats = build_graph_for_patient(patient_id=p_id, driver=driver)
            all_stats[p_id] = stats

    finally:
        if should_close_driver and driver is not None:
            driver.close()

    print("\n==================================================")
    print("         KNOWLEDGE GRAPH BUILD COMPLETE           ")
    print("==================================================")
    for p_id, stats in all_stats.items():
        print(f"[{p_id}] Chunks: {stats['total_chunks']}, Extracted: {stats['triples_extracted']}, Written: {stats['triples_written']}")
        # A build that writes nothing used to look like a success and left an empty
        # graph that only showed up later as "Not documented in the available records".
        if stats["total_chunks"] and not stats["triples_written"]:
            print(
                f"[{p_id}] WARNING: no facts were written from {stats['total_chunks']} chunks. "
                "The graph for this patient is EMPTY and every query will come back "
                "undocumented. Check the extraction warnings above."
            )
    print("==================================================")
    return all_stats


if __name__ == "__main__":
    build_all_patients()
