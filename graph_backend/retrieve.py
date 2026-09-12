"""
Medical Records RAG Demo — Graph Retrieval Engine (SRS §6.2)

Provides parameterized Cypher templates for common clinical question categories,
an intent classifier, LLM-generated Cypher fallback for open-ended queries,
and a unified `retrieve(question, patient_ids)` entry point.

Maps to BUILD_GUIDE Task 2.4 and SRS FR-6.2.1–FR-6.2.4.
"""

import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Ensure project root is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from graph_backend.build import get_driver
from llm_client import chat


def get_lab_trend(
    patient_id: str,
    test_name: Optional[str] = None,
    driver=None,
) -> List[Dict[str, Any]]:
    """
    Retrieves timeline of LabResult values for a patient and optional test name (SRS §6.2.1).
    """
    should_close_driver = False
    if driver is None:
        driver = get_driver()
        should_close_driver = True

    query = """
    MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[rel1:HAS_RESULT]->(lr:LabResult)-[rel2:OF_TEST]->(lt:LabTest)
    WHERE $test_name IS NULL OR toLower(lt.canonical_name) CONTAINS toLower($test_name)
    RETURN lt.canonical_name AS test_name,
           lr.value AS value,
           lr.unit AS unit,
           coalesce(lr.date, r.report_date) AS date,
           r.report_id AS report_id,
           rel1.evidence_id AS evidence_id,
           rel1.confidence AS confidence
    ORDER BY date ASC
    """
    results: List[Dict[str, Any]] = []
    try:
        with driver.session() as session:
            records = session.run(query, patient_id=patient_id, test_name=test_name)
            for rec in records:
                t_name = rec.get("test_name", "")
                val = rec.get("value", "")
                unit = rec.get("unit", "")
                dt = rec.get("date", "Unknown date")
                unit_str = f" {unit}" if unit else ""
                fact_text = f"Lab Test '{t_name}': {val}{unit_str} on {dt}"
                results.append({
                    "text": fact_text,
                    "patient_id": patient_id,
                    "evidence_id": rec.get("evidence_id", ""),
                    "confidence": float(rec.get("confidence") or 0.9),
                    "raw_data": dict(rec),
                })
    finally:
        if should_close_driver and driver is not None:
            driver.close()

    return results


def get_diagnoses(
    patient_id: str,
    driver=None,
) -> List[Dict[str, Any]]:
    """
    Retrieves all diagnoses documented for a patient (SRS §6.2.1).
    """
    should_close_driver = False
    if driver is None:
        driver = get_driver()
        should_close_driver = True

    query = """
    MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[rel:STATES_DIAGNOSIS]->(d:Diagnosis)
    RETURN d.canonical_name AS diagnosis,
           r.report_type AS report_type,
           r.report_date AS date,
           r.report_id AS report_id,
           rel.evidence_id AS evidence_id,
           rel.confidence AS confidence
    ORDER BY date ASC
    """
    results: List[Dict[str, Any]] = []
    try:
        with driver.session() as session:
            records = session.run(query, patient_id=patient_id)
            for rec in records:
                diag = rec.get("diagnosis", "")
                dt = rec.get("date") or "undated"
                rpt_type = rec.get("report_type", "Report")
                fact_text = f"Diagnosis '{diag}' documented in {rpt_type} ({dt})"
                results.append({
                    "text": fact_text,
                    "patient_id": patient_id,
                    "evidence_id": rec.get("evidence_id", ""),
                    "confidence": float(rec.get("confidence") or 0.9),
                    "raw_data": dict(rec),
                })
    finally:
        if should_close_driver and driver is not None:
            driver.close()

    return results


def get_medication_history(
    patient_id: str,
    driver=None,
) -> List[Dict[str, Any]]:
    """
    Retrieves all medications, chemotherapy regimens, and procedures for a patient (SRS §6.2.1).
    """
    should_close_driver = False
    if driver is None:
        driver = get_driver()
        should_close_driver = True

    # 1. Medications administered
    query_meds = """
    MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[rel:ADMINISTERED]->(m:Medication)
    RETURN m.canonical_name AS name,
           'Medication' AS item_type,
           rel.dose AS dose,
           rel.cycle AS cycle,
           r.report_date AS date,
           r.report_id AS report_id,
           rel.evidence_id AS evidence_id,
           rel.confidence AS confidence
    ORDER BY date ASC
    """

    # 2. Regimens administered
    query_regs = """
    MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[rel:ADMINISTERED]->(reg:Regimen)
    RETURN reg.canonical_name AS name,
           'Regimen' AS item_type,
           rel.dose AS dose,
           rel.cycle AS cycle,
           r.report_date AS date,
           r.report_id AS report_id,
           rel.evidence_id AS evidence_id,
           rel.confidence AS confidence
    ORDER BY date ASC
    """

    # 3. Procedures undergone
    query_procs = """
    MATCH (p:Patient {patient_id: $patient_id})-[rel:UNDERWENT]->(pr:Procedure)
    RETURN pr.canonical_name AS name,
           'Procedure' AS item_type,
           '' AS dose,
           '' AS cycle,
           rel.date AS date,
           '' AS report_id,
           rel.evidence_id AS evidence_id,
           rel.confidence AS confidence
    ORDER BY date ASC
    """

    results: List[Dict[str, Any]] = []
    try:
        with driver.session() as session:
            # Process Meds
            for rec in session.run(query_meds, patient_id=patient_id):
                dose_info = f" (Dose: {rec['dose']})" if rec.get("dose") else ""
                cycle_info = f" Cycle {rec['cycle']}" if rec.get("cycle") else ""
                dt = rec.get("date") or "undated"
                fact_text = f"Administered Medication '{rec['name']}'{cycle_info}{dose_info} on {dt}"
                results.append({
                    "text": fact_text,
                    "patient_id": patient_id,
                    "evidence_id": rec.get("evidence_id", ""),
                    "confidence": float(rec.get("confidence") or 0.9),
                    "raw_data": dict(rec),
                })

            # Process Regimens
            for rec in session.run(query_regs, patient_id=patient_id):
                dt = rec.get("date") or "undated"
                fact_text = f"Administered Regimen '{rec['name']}' on {dt}"
                results.append({
                    "text": fact_text,
                    "patient_id": patient_id,
                    "evidence_id": rec.get("evidence_id", ""),
                    "confidence": float(rec.get("confidence") or 0.9),
                    "raw_data": dict(rec),
                })

            # Process Procedures
            for rec in session.run(query_procs, patient_id=patient_id):
                dt = rec.get("date") or "undated"
                fact_text = f"Underwent Procedure '{rec['name']}' on {dt}"
                results.append({
                    "text": fact_text,
                    "patient_id": patient_id,
                    "evidence_id": rec.get("evidence_id", ""),
                    "confidence": float(rec.get("confidence") or 0.9),
                    "raw_data": dict(rec),
                })

    finally:
        if should_close_driver and driver is not None:
            driver.close()

    return results


def get_staging_and_biomarkers(
    patient_id: str,
    driver=None,
) -> List[Dict[str, Any]]:
    """
    Retrieves TNM Staging and Biomarkers (ER, PR, HER2, Ki-67) for a patient (SRS §6.2.1).
    """
    should_close_driver = False
    if driver is None:
        driver = get_driver()
        should_close_driver = True

    query_biomarkers = """
    MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[rel:HAS_BIOMARKER]->(b:Biomarker)
    RETURN b.marker AS marker,
           b.value AS value,
           coalesce(b.date, r.report_date) AS date,
           r.report_id AS report_id,
           rel.evidence_id AS evidence_id,
           rel.confidence AS confidence
    ORDER BY date ASC
    """

    query_staging = """
    MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[rel:HAS_STAGING]->(st:Staging)
    RETURN st.t AS t,
           st.n AS n,
           st.m AS m,
           coalesce(st.date, r.report_date) AS date,
           r.report_id AS report_id,
           rel.evidence_id AS evidence_id,
           rel.confidence AS confidence
    ORDER BY date ASC
    """

    results: List[Dict[str, Any]] = []
    try:
        with driver.session() as session:
            # Biomarkers
            for rec in session.run(query_biomarkers, patient_id=patient_id):
                marker = rec.get("marker", "Biomarker")
                val = rec.get("value", "")
                dt = rec.get("date") or "undated"
                fact_text = f"Biomarker '{marker}': {val} ({dt})"
                results.append({
                    "text": fact_text,
                    "patient_id": patient_id,
                    "evidence_id": rec.get("evidence_id", ""),
                    "confidence": float(rec.get("confidence") or 0.9),
                    "raw_data": dict(rec),
                })

            # Staging
            for rec in session.run(query_staging, patient_id=patient_id):
                t_val = rec.get("t") or ""
                n_val = rec.get("n") or ""
                m_val = rec.get("m") or ""
                st_str = f"T:{t_val} N:{n_val} M:{m_val}".strip()
                dt = rec.get("date") or "undated"
                fact_text = f"Staging (TNM): {st_str} ({dt})"
                results.append({
                    "text": fact_text,
                    "patient_id": patient_id,
                    "evidence_id": rec.get("evidence_id", ""),
                    "confidence": float(rec.get("confidence") or 0.9),
                    "raw_data": dict(rec),
                })

    finally:
        if should_close_driver and driver is not None:
            driver.close()

    return results


def classify_intent(question: str) -> str:
    """
    Classifies question intent into one of:
    - "lab_trend"
    - "diagnosis_list"
    - "medication_history"
    - "staging_biomarker"
    - "open_ended"
    """
    q_lower = question.lower()

    # Diagnoses patterns (checked early for diagnosis-specific inquiries)
    if any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in [
        "diagnosis", "diagnoses", "disease", "diseases", "suffering from", "condition",
        "illness", "what cancer", "type of cancer", "pathology finding"
    ]):
        return "diagnosis_list"

    # Staging & Biomarkers patterns
    if any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in [
        "biomarker", "biomarkers", "her2", "her-2", "her2neu", "er/pr", "estrogen",
        "progesterone", "ki67", "ki-67", "staging", "stage", "tnm", "grade", "nottingham",
        "receptor status", "receptors"
    ]):
        return "staging_biomarker"

    # Lab trends patterns (word boundaries prevent 'anc' from matching 'cancer')
    if any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in [
        "hemoglobin", "platelet", "platelets", "creatinine", "sgot", "sgpt", "lft", "kft",
        "bilirubin", "lab test", "lab tests", "lab value", "lab values", "blood count",
        "trend", "trends", "wbc", "anc", "tlc", "calcium", "phosphate", "urea"
    ]):
        return "lab_trend"

    # Medication / Treatment / Regimen / Procedure patterns
    if any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in [
        "medication", "medications", "medicine", "medicines", "drug", "drugs", "chemo",
        "chemotherapy", "regimen", "regimens", "treatment", "treatments", "cycle", "dose",
        "paclitaxel", "doxorubicin", "trastuzumab", "tamoxifen", "letrozole", "surgery",
        "operation", "procedure", "procedures", "underwent", "mrm", "mastectomy"
    ]):
        return "medication_history"

    # LLM classification fallback for ambiguous questions
    prompt = (
        f"Classify the following medical question into exactly ONE of these categories:\n"
        f"- lab_trend\n"
        f"- diagnosis_list\n"
        f"- medication_history\n"
        f"- staging_biomarker\n"
        f"- open_ended\n\n"
        f"Question: \"{question}\"\n\n"
        f"Respond ONLY with the single category name."
    )
    try:
        resp = chat([{"role": "user", "content": prompt}]).strip().lower()
        valid_intents = {"lab_trend", "diagnosis_list", "medication_history", "staging_biomarker", "open_ended"}
        for valid in valid_intents:
            if valid in resp:
                return valid
    except Exception:
        pass

    return "open_ended"


def open_ended_query(
    patient_ids: List[str],
    question: str,
    driver=None,
) -> List[Dict[str, Any]]:
    """
    Fallback LLM-generated Cypher query for open-ended questions (SRS FR-6.2.2).
    Always parameterized by $patient_ids for strict scoping defense-in-depth.
    """
    should_close_driver = False
    if driver is None:
        driver = get_driver()
        should_close_driver = True

    schema_description = """
    Nodes:
    - Patient {patient_id, display_label}
    - Report {report_id, patient_id, report_type, report_date}
    - Diagnosis {canonical_name}
    - Procedure {canonical_name}
    - Medication {canonical_name}
    - Regimen {canonical_name}
    - LabTest {canonical_name}
    - LabResult {value, unit, date}
    - Staging {t, n, m, date}
    - Biomarker {marker, value, date}
    Relationships:
    - (p:Patient)-[:HAS_REPORT]->(r:Report)
    - (r:Report)-[rel:STATES_DIAGNOSIS]->(d:Diagnosis)
    - (p:Patient)-[rel:UNDERWENT {date}]->(pr:Procedure)
    - (r:Report)-[rel:ADMINISTERED {dose, cycle}]->(m:Medication)
    - (r:Report)-[rel:ADMINISTERED]->(reg:Regimen)
    - (reg:Regimen)-[rel:CONTAINS]->(m:Medication)
    - (r:Report)-[rel1:HAS_RESULT]->(lr:LabResult)-[rel2:OF_TEST]->(lt:LabTest)
    - (r:Report)-[rel:HAS_STAGING]->(st:Staging)
    - (r:Report)-[rel:HAS_BIOMARKER]->(b:Biomarker)
    Every relationship carries 'evidence_id' and 'confidence'.
    """

    prompt = (
        f"You are a Neo4j Cypher expert. Generate a single Cypher query to answer this clinical question:\n"
        f"Question: \"{question}\"\n\n"
        f"SCHEMA:\n{schema_description}\n\n"
        f"REQUIREMENTS:\n"
        f"1. You MUST filter on: WHERE p.patient_id IN $patient_ids\n"
        f"2. Return at least: p.patient_id AS patient_id, rel.evidence_id AS evidence_id, rel.confidence AS confidence, and relevant entity properties.\n"
        f"3. Return ONLY the raw Cypher query text without markdown quotes."
    )

    results: List[Dict[str, Any]] = []
    try:
        resp = chat([{"role": "user", "content": prompt}]).strip()
        cypher_query = re.sub(r"^```(?:cypher)?\s*", "", resp)
        cypher_query = re.sub(r"\s*```$", "", cypher_query).strip()

        with driver.session() as session:
            records = session.run(cypher_query, patient_ids=patient_ids)
            for rec in records:
                d = dict(rec)
                p_id = d.get("patient_id", patient_ids[0] if patient_ids else "unknown")
                ev_id = d.get("evidence_id") or ""
                conf = float(d.get("confidence") or 0.9)
                fact_str = ", ".join(f"{k}: {v}" for k, v in d.items() if k not in ("evidence_id", "confidence"))
                results.append({
                    "text": fact_str,
                    "patient_id": p_id,
                    "evidence_id": ev_id,
                    "confidence": conf,
                    "raw_data": d,
                })
    except Exception as exc:
        print(f"  [WARN] Open-ended Cypher execution failed: {exc}")
    finally:
        if should_close_driver and driver is not None:
            driver.close()

    return results


def retrieve(
    question: str,
    patient_ids: List[str],
    driver=None,
) -> List[Dict[str, Any]]:
    """
    Main retrieval entry point for the GraphRAG backend (SRS FR-6.2.1–FR-6.2.4).

    Args:
        question: Clinical question asked by user.
        patient_ids: List of scoped patient IDs (one for Individual mode, one/more for Group mode).
        driver: Optional existing Neo4j driver.

    Returns:
        Flat list of facts dicts matching the SRS §8.1 facts contract:
        `[{"text": str, "patient_id": str, "evidence_id": str, "confidence": float}, ...]`
    """
    if not patient_ids:
        return []

    intent = classify_intent(question)
    print(f"[*] Graph Retrieval Intent: '{intent}' for patients: {patient_ids}")

    should_close_driver = False
    if driver is None:
        driver = get_driver()
        should_close_driver = True

    all_facts: List[Dict[str, Any]] = []

    try:
        # Loop per patient_id (SRS FR-6.2.4)
        for p_id in patient_ids:
            if intent == "lab_trend":
                # Extract optional test name from question if present
                test_match = None
                for candidate in ["hemoglobin", "platelet", "creatinine", "sgot", "sgpt", "bilirubin"]:
                    if candidate in question.lower():
                        test_match = candidate
                        break
                facts = get_lab_trend(patient_id=p_id, test_name=test_match, driver=driver)
                all_facts.extend(facts)

            elif intent == "diagnosis_list":
                facts = get_diagnoses(patient_id=p_id, driver=driver)
                all_facts.extend(facts)

            elif intent == "medication_history":
                facts = get_medication_history(patient_id=p_id, driver=driver)
                all_facts.extend(facts)

            elif intent == "staging_biomarker":
                facts = get_staging_and_biomarkers(patient_id=p_id, driver=driver)
                all_facts.extend(facts)

            else:
                # Open-ended query
                facts = open_ended_query(patient_ids=[p_id], question=question, driver=driver)
                all_facts.extend(facts)

    finally:
        if should_close_driver and driver is not None:
            driver.close()

    return all_facts


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_text = sys.argv[1]
    else:
        query_text = "What is my diagnosis and health progress?"

    retrieved = retrieve(query_text, ["patient_a"])
    print(f"\nRetrieved {len(retrieved)} facts for '{query_text}':")
    for f in retrieved[:5]:
        print(f"  - [{f['patient_id']}] {f['text']} (evidence: {f['evidence_id']}, conf: {f['confidence']:.2f})")
