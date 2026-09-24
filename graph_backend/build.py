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
from llm_client import chat
from normalize import canonicalize_lab_name, compute_abnormal_flag


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

    uris_to_try = [uri]
    if uri.startswith("neo4j+s://"):
        uris_to_try.append(uri.replace("neo4j+s://", "neo4j+ssc://"))
    elif uri.startswith("bolt+s://"):
        uris_to_try.append(uri.replace("bolt+s://", "bolt+ssc://"))

    last_error: Optional[Exception] = None
    for target_uri in uris_to_try:
        try:
            driver = GraphDatabase.driver(target_uri, auth=(username, password))
            driver.verify_connectivity()
            return driver
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to connect to Neo4j instance at {uri}: {last_error}"
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

    # Format normalized entities for context
    norm_summary = []
    for ent in normalized_entities:
        norm_summary.append(
            f"- '{ent.get('raw_text')}': canonical='{ent.get('normalized_term')}', type={ent.get('entity_type')}"
        )
    norm_context_str = "\n".join(norm_summary) if norm_summary else "(None)"

    prompt = (
        f"REPORT METADATA:\n"
        f"- Patient ID: {patient_id}\n"
        f"- Report ID: {report_id}\n"
        f"- Report Type: {report_type}\n"
        f"- Report Date: {report_date}\n\n"
        f"PRE-NORMALIZED ENTITIES IN CHUNK:\n"
        f"{norm_context_str}\n\n"
        f"RAW CHUNK TEXT:\n"
        f"\"\"\"\n{chunk_text[:3500]}\n\"\"\"\n\n"
        f"Extract all clinical triples matching the schema. Return ONLY valid JSON list of triples."
    )

    try:
        response_text = chat(
            messages=[{"role": "user", "content": prompt}],
            system=EXTRACTION_SYSTEM_PROMPT,
            task="build",
        )
        if not response_text or not response_text.strip():
            return []

        raw_resp = response_text.strip()
        # Remove any reasoning/thought tags from local model if present
        raw_resp = re.sub(r"<thought>.*?</thought>", "", raw_resp, flags=re.DOTALL).strip()

        # Extract JSON array block if enclosed in text/markdown
        match = re.search(r"(\[\s*\{.*\}\s*\])", raw_resp, re.DOTALL)
        if match:
            raw_resp = match.group(1)
        elif raw_resp.startswith("```"):
            raw_resp = re.sub(r"^```(?:json)?\s*", "", raw_resp)
            raw_resp = re.sub(r"\s*```$", "", raw_resp)

        parsed = json.loads(raw_resp.strip())
        if not isinstance(parsed, list):
            return []

        # Schema validation: reject any triple with off-schema labels/relations
        valid_triples = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            relation = item.get("relation", "")
            subject_label = item.get("subject_label", "")
            object_label = item.get("object_label", "")

            # Reject off-schema relations
            if relation not in VALID_RELATION_TYPES:
                print(f"  [SCHEMA] Rejected off-schema relation '{relation}' in {report_id}")
                continue

            # Reject off-schema node labels (warn but don't crash)
            if subject_label and subject_label not in VALID_NODE_LABELS:
                print(f"  [SCHEMA] Rejected off-schema subject label '{subject_label}' in {report_id}")
                continue
            if object_label and object_label not in VALID_NODE_LABELS:
                print(f"  [SCHEMA] Rejected off-schema object label '{object_label}' in {report_id}")
                continue

            valid_triples.append(item)

        return valid_triples

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
                # Canonicalize lab test name to prevent duplicate nodes
                raw_test_name = str(obj_name or props.get("test_name", "")).strip()
                test_name = canonicalize_lab_name(raw_test_name)
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
                canonical_test = canonicalize_lab_name(raw_test)
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
    print("==================================================")
    return all_stats


if __name__ == "__main__":
    build_all_patients()
