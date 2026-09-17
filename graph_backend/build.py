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


def get_driver():
    """
    Creates and returns an authenticated Neo4j driver using .env credentials.
    Supports fallback to neo4j+ssc:// if standard SSL verification fails.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError(
            "neo4j package is required. Run `pip install neo4j`."
        ) from e

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not password:
        raise ValueError(
            "NEO4J_URI or NEO4J_PASSWORD is not set in environment / .env file."
        )

    uris_to_try = [uri]
    if uri.startswith("neo4j+s://"):
        uris_to_try.append(uri.replace("neo4j+s://", "neo4j+ssc://"))

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


EXTRACTION_SYSTEM_PROMPT = """You are an expert clinical oncology knowledge graph extraction assistant.
Your task is to extract structured medical entities and relationships from clinical report text into a strict knowledge graph schema.

ALLOWED NODE LABELS:
- Patient: {patient_id}
- Report: {report_id}
- Diagnosis: {canonical_name} (e.g., 'Metastatic Breast Cancer', 'Invasive Ductal Carcinoma')
- Procedure: {canonical_name} (e.g., 'Modified Radical Mastectomy', 'CECT Chest Abdomen Pelvis')
- Medication: {canonical_name} (e.g., 'Paclitaxel', 'Trastuzumab', 'Doxorubicin')
- Regimen: {canonical_name} (e.g., 'AC Regimen', 'Docetaxel-Cyclophosphamide')
- LabTest: {canonical_name} (e.g., 'Hemoglobin', 'Platelet Count', 'Serum Creatinine', 'SGOT', 'SGPT')
- LabResult: {value, unit, date}
- Staging: {t, n, m, date}
- Biomarker: {marker, value, date} (e.g., marker='ER', value='8/8' or marker='HER2', value='3+')

ALLOWED RELATIONSHIPS:
- (Report)-[:STATES_DIAGNOSIS]->(Diagnosis)
- (Patient)-[:UNDERWENT {date}]->(Procedure)
- (Report)-[:ADMINISTERED {dose, cycle}]->(Medication)
- (Report)-[:ADMINISTERED]->(Regimen)
- (Regimen)-[:CONTAINS]->(Medication)
- (Report)-[:HAS_RESULT]->(LabResult)  [and (LabResult)-[:OF_TEST]->(LabTest)]
- (Report)-[:HAS_STAGING]->(Staging)
- (Report)-[:HAS_BIOMARKER]->(Biomarker)

INSTRUCTIONS:
1. Extract ONLY facts explicitly stated in the chunk text and pre-normalized entities.
2. Use canonical names where pre-normalized entities are supplied.
3. If no relevant clinical entities or relations exist in the chunk, return an empty list [].
4. Output MUST be valid JSON: a list of objects with the exact schema:
[
  {
    "subject_label": "Report" | "Patient" | "Regimen" | "LabResult",
    "subject_name": "...",
    "relation": "STATES_DIAGNOSIS" | "UNDERWENT" | "ADMINISTERED" | "CONTAINS" | "HAS_RESULT" | "HAS_STAGING" | "HAS_BIOMARKER",
    "object_label": "Diagnosis" | "Procedure" | "Medication" | "Regimen" | "LabResult" | "Staging" | "Biomarker",
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
    Constrained strictly to the schema in SRS §12.1.
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
        )
        if not response_text or not response_text.strip():
            return []

        raw_resp = response_text.strip()
        if raw_resp.startswith("```"):
            raw_resp = re.sub(r"^```(?:json)?\s*", "", raw_resp)
            raw_resp = re.sub(r"\s*```$", "", raw_resp)

        parsed = json.loads(raw_resp.strip())
        if not isinstance(parsed, list):
            return []

        # Validate triple structure
        valid_triples = []
        valid_relations = {
            "STATES_DIAGNOSIS",
            "UNDERWENT",
            "ADMINISTERED",
            "CONTAINS",
            "HAS_RESULT",
            "OF_TEST",
            "HAS_STAGING",
            "HAS_BIOMARKER",
        }
        for item in parsed:
            if not isinstance(item, dict):
                continue
            relation = item.get("relation", "")
            if relation in valid_relations:
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
                canonical = str(obj_name).strip()
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
                canonical = str(obj_name).strip()
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
                reg_name = str(triple.get("subject_name", "")).strip()
                med_name = str(obj_name).strip()
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
                test_name = str(obj_name or props.get("test_name", "")).strip()
                val = str(props.get("value", "")).strip()
                unit = str(props.get("unit", "")).strip()
                res_date = str(props.get("date") or report_date or "").strip()
                if test_name:
                    session.run(
                        """
                        MERGE (r:Report {report_id: $report_id})
                        CREATE (lr:LabResult {value: $value, unit: $unit, date: $date})
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
