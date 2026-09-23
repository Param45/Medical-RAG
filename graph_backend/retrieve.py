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

from graph_backend.build import get_driver, get_driver_for_patient
from llm_client import chat
from normalize import canonicalize_lab_name


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
        driver = get_driver_for_patient(patient_id)
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
        driver = get_driver_for_patient(patient_id)
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
        driver = get_driver_for_patient(patient_id)
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
        driver = get_driver_for_patient(patient_id)
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


def get_chemo_cycle_status(
    patient_id: str,
    driver=None,
) -> List[Dict[str, Any]]:
    """
    Deterministic chemo cycle status retrieval.
    Computes completed cycles and remaining cycles in Python, not the LLM.
    Answers: "how many cycles", "remaining cycles", "completed chemo round"
    """
    should_close_driver = False
    if driver is None:
        driver = get_driver_for_patient(patient_id)
        should_close_driver = True

    results: List[Dict[str, Any]] = []
    try:
        with driver.session() as session:
            # Get all chemo or medication administrations with cycle numbers
            admin_records = session.run(
                """
                MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[rel:HAS_CHEMO_ADMIN|HAS_MED_ADMIN]->(ca)
                RETURN ca.cycle_number AS cycle_number,
                       ca.regimen AS regimen,
                       ca.date AS date,
                       ca.admin_id AS admin_id,
                       rel.evidence_id AS evidence_id,
                       rel.confidence AS confidence
                ORDER BY ca.cycle_number ASC
                """,
                patient_id=patient_id,
            )
            cycles = [dict(rec) for rec in admin_records]

            # Get treatment plan
            plan_records = session.run(
                """
                MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[rel:HAS_TREATMENT_PLAN]->(tp:TreatmentPlan)
                RETURN tp.regimen AS regimen,
                       tp.planned_cycles AS planned_cycles,
                       tp.confidence_note AS confidence_note,
                       rel.evidence_id AS evidence_id,
                       rel.confidence AS confidence
                """,
                patient_id=patient_id,
            )
            plans = [dict(rec) for rec in plan_records]

            if cycles:
                # Deterministic computation: completed = MAX(cycle_number)
                max_cycle = max(c["cycle_number"] for c in cycles if c.get("cycle_number") and c["cycle_number"] > 0)
                regimen = cycles[0].get("regimen", "chemotherapy")
                cycle_dates = [f"Cycle {c['cycle_number']} on {c.get('date', 'undated')}" for c in cycles if c.get("cycle_number", 0) > 0]

                if plans:
                    planned = plans[0].get("planned_cycles", -1)
                    if planned and planned > 0:
                        remaining = planned - max_cycle
                        fact_text = (
                            f"[PRE-COMPUTED] Chemotherapy cycle status for {regimen}: "
                            f"Completed {max_cycle} of {planned} planned cycles. "
                            f"{remaining} cycle(s) remaining. "
                            f"Cycle history: {'; '.join(cycle_dates)}."
                        )
                    else:
                        fact_text = (
                            f"[PRE-COMPUTED] Chemotherapy cycle status for {regimen}: "
                            f"Completed {max_cycle} cycle(s). "
                            f"Planned total cycles not found in records. "
                            f"Cycle history: {'; '.join(cycle_dates)}."
                        )
                else:
                    fact_text = (
                        f"[PRE-COMPUTED] Chemotherapy cycle status for {regimen}: "
                        f"Completed {max_cycle} cycle(s). "
                        f"Planned total cycles not found in records. "
                        f"Cycle history: {'; '.join(cycle_dates)}."
                    )

                results.append({
                    "text": fact_text,
                    "patient_id": patient_id,
                    "evidence_id": cycles[0].get("evidence_id", ""),
                    "confidence": float(cycles[0].get("confidence") or 0.9),
                    "is_precomputed": True,
                })
            else:
                results.append({
                    "text": "[PRE-COMPUTED] No chemotherapy cycle administration records found in the graph.",
                    "patient_id": patient_id,
                    "evidence_id": "",
                    "confidence": 1.0,
                    "is_precomputed": True,
                })
    finally:
        if should_close_driver and driver is not None:
            driver.close()

    return results


def get_suggested_vs_performed(
    patient_id: str,
    driver=None,
) -> List[Dict[str, Any]]:
    """
    Deterministic set-difference: suggested tests vs performed tests.
    Answers: "what tests were suggested", "pending tests", "which tests done"
    """
    should_close_driver = False
    if driver is None:
        driver = get_driver_for_patient(patient_id)
        should_close_driver = True

    results: List[Dict[str, Any]] = []
    try:
        with driver.session() as session:
            # Get all suggested tests
            suggested_recs = session.run(
                """
                MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[rel:SUGGESTS_TEST]->(t)
                RETURN t.canonical_name AS test_name,
                       labels(t)[0] AS test_type,
                       rel.date AS suggested_date,
                       r.report_date AS report_date,
                       rel.evidence_id AS evidence_id
                ORDER BY rel.date ASC
                """,
                patient_id=patient_id,
            )
            suggested = {rec["test_name"]: dict(rec) for rec in suggested_recs if rec.get("test_name")}

            # Get all performed tests (those with actual results)
            performed_recs = session.run(
                """
                MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[:HAS_RESULT]->(lr:LabResult)-[:OF_TEST]->(lt:LabTest)
                RETURN DISTINCT lt.canonical_name AS test_name
                """,
                patient_id=patient_id,
            )
            performed_set = {rec["test_name"] for rec in performed_recs if rec.get("test_name")}

            # Also check procedures that were undergone
            undergone_recs = session.run(
                """
                MATCH (p:Patient {patient_id: $patient_id})-[:UNDERWENT]->(pr:Procedure)
                RETURN DISTINCT pr.canonical_name AS test_name
                """,
                patient_id=patient_id,
            )
            undergone_set = {rec["test_name"] for rec in undergone_recs if rec.get("test_name")}
            all_performed = performed_set | undergone_set

            # Deterministic set-difference
            suggested_names = set(suggested.keys())
            pending = suggested_names - all_performed
            completed = suggested_names & all_performed

            ev_id = next(iter(suggested.values()), {}).get("evidence_id", "") if suggested else ""

            fact_text = (
                f"[PRE-COMPUTED] Suggested vs Performed Tests:\n"
                f"Suggested tests: {', '.join(sorted(suggested_names)) if suggested_names else 'None found in records'}.\n"
                f"Performed/completed: {', '.join(sorted(completed)) if completed else 'None'}.\n"
                f"Pending (suggested but not yet performed): {', '.join(sorted(pending)) if pending else 'All suggested tests have been performed'}."
            )

            results.append({
                "text": fact_text,
                "patient_id": patient_id,
                "evidence_id": ev_id,
                "confidence": 0.9,
                "is_precomputed": True,
            })
    finally:
        if should_close_driver and driver is not None:
            driver.close()

    return results


def get_improvement_trend(
    patient_id: str,
    test_name: str = None,
    driver=None,
) -> List[Dict[str, Any]]:
    """
    Deterministic improvement assessment using abnormal_flag comparison.
    Compares first vs last lab values and flags in Python.
    Answers: "am I improving", "getting better", "health progress"
    """
    should_close_driver = False
    if driver is None:
        driver = get_driver_for_patient(patient_id)
        should_close_driver = True

    results: List[Dict[str, Any]] = []
    try:
        with driver.session() as session:
            # Get all lab results ordered by date, optionally filtered by test
            query = """
            MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[rel1:HAS_RESULT]->(lr:LabResult)-[rel2:OF_TEST]->(lt:LabTest)
            WHERE $test_name IS NULL OR toLower(lt.canonical_name) CONTAINS toLower($test_name)
            RETURN lt.canonical_name AS test_name,
                   lr.value AS value,
                   lr.unit AS unit,
                   lr.abnormal_flag AS abnormal_flag,
                   coalesce(lr.result_date, lr.date, r.report_date) AS date,
                   rel1.evidence_id AS evidence_id,
                   rel1.confidence AS confidence
            ORDER BY lt.canonical_name, date ASC
            """
            records = session.run(query, patient_id=patient_id, test_name=test_name)
            all_results = [dict(rec) for rec in records]

            if not all_results:
                results.append({
                    "text": f"[PRE-COMPUTED] No lab results found{' for ' + test_name if test_name else ''} to assess improvement.",
                    "patient_id": patient_id,
                    "evidence_id": "",
                    "confidence": 1.0,
                    "is_precomputed": True,
                })
            else:
                # Group by test name
                from collections import defaultdict
                by_test = defaultdict(list)
                for r in all_results:
                    by_test[r["test_name"]].append(r)

                trend_summaries = []
                for tname, values in by_test.items():
                    if len(values) < 2:
                        continue
                    first = values[0]
                    last = values[-1]
                    first_flag = first.get("abnormal_flag", "")
                    last_flag = last.get("abnormal_flag", "")
                    first_val = first.get("value", "?")
                    last_val = last.get("value", "?")
                    first_date = first.get("date", "?")
                    last_date = last.get("date", "?")
                    unit = first.get("unit", "")
                    unit_str = f" {unit}" if unit else ""

                    # Determine trend direction
                    if first_flag == "low" and last_flag == "normal":
                        direction = "IMPROVING (was low, now normal)"
                    elif first_flag == "high" and last_flag == "normal":
                        direction = "IMPROVING (was high, now normal)"
                    elif first_flag == "normal" and last_flag in ("high", "low"):
                        direction = "WORSENING (was normal, now " + last_flag + ")"
                    elif first_flag == last_flag:
                        direction = "STABLE (" + (last_flag or "normal") + ")"
                    else:
                        direction = f"CHANGED ({first_flag or 'unknown'} → {last_flag or 'unknown'})"

                    trend_summaries.append(
                        f"{tname}: {first_val}{unit_str} ({first_flag or 'N/A'}) on {first_date} → "
                        f"{last_val}{unit_str} ({last_flag or 'N/A'}) on {last_date}. Trend: {direction}."
                    )

                if trend_summaries:
                    fact_text = "[PRE-COMPUTED] Lab Trend Assessment:\n" + "\n".join(trend_summaries)
                else:
                    fact_text = "[PRE-COMPUTED] Not enough data points (need at least 2 values per test) to determine improvement trend."

                results.append({
                    "text": fact_text,
                    "patient_id": patient_id,
                    "evidence_id": all_results[0].get("evidence_id", ""),
                    "confidence": float(all_results[0].get("confidence") or 0.9),
                    "is_precomputed": True,
                })
    finally:
        if should_close_driver and driver is not None:
            driver.close()

    return results


def get_specific_lab_value(
    patient_id: str,
    test_name: str,
    target_date: str = None,
    driver=None,
) -> List[Dict[str, Any]]:
    """
    Retrieves specific lab value(s), optionally filtered to a date/period.
    Answers: "sugar level in March", "hemoglobin on [date]"
    """
    should_close_driver = False
    if driver is None:
        driver = get_driver_for_patient(patient_id)
        should_close_driver = True

    # Canonicalize the test name to match graph nodes
    canonical_test = canonicalize_lab_name(test_name) if test_name else test_name

    results: List[Dict[str, Any]] = []
    try:
        with driver.session() as session:
            query = """
            MATCH (p:Patient {patient_id: $patient_id})-[:HAS_REPORT]->(r:Report)-[rel1:HAS_RESULT]->(lr:LabResult)-[rel2:OF_TEST]->(lt:LabTest)
            WHERE toLower(lt.canonical_name) CONTAINS toLower($test_name)
            RETURN lt.canonical_name AS test_name,
                   lr.value AS value,
                   lr.unit AS unit,
                   lr.abnormal_flag AS abnormal_flag,
                   coalesce(lr.result_date, lr.date, r.report_date) AS date,
                   rel1.evidence_id AS evidence_id,
                   rel1.confidence AS confidence
            ORDER BY date ASC
            """
            records = session.run(query, patient_id=patient_id, test_name=canonical_test or "")
            all_results = [dict(rec) for rec in records]

            # Filter by target date/period if provided
            if target_date and all_results:
                filtered = [r for r in all_results if target_date.lower() in str(r.get("date", "")).lower()]
                if filtered:
                    all_results = filtered

            for rec in all_results:
                t_name = rec.get("test_name", "")
                val = rec.get("value", "")
                unit = rec.get("unit", "")
                flag = rec.get("abnormal_flag", "")
                dt = rec.get("date", "Unknown date")
                unit_str = f" {unit}" if unit else ""
                flag_str = f" ({flag})" if flag else ""
                fact_text = f"[PRE-COMPUTED] {t_name}: {val}{unit_str}{flag_str} on {dt}"
                results.append({
                    "text": fact_text,
                    "patient_id": patient_id,
                    "evidence_id": rec.get("evidence_id", ""),
                    "confidence": float(rec.get("confidence") or 0.9),
                    "is_precomputed": True,
                })
    finally:
        if should_close_driver and driver is not None:
            driver.close()

    return results


def classify_intent(question: str) -> str:
    """
    Classifies question intent into one of:
    - "chemo_cycle_status"
    - "suggested_vs_performed"
    - "improvement_trend"
    - "specific_lab_value"
    - "lab_trend"
    - "diagnosis_list"
    - "medication_history"
    - "staging_biomarker"
    - "open_ended"
    """
    q_lower = question.lower()

    # --- New deterministic intents (checked FIRST, before general ones) ---

    # Chemo / treatment cycle status: "how many cycles", "remaining cycles", "completed chemo", "dialysis sessions"
    if (any(k in q_lower for k in ["chemo", "chemotherapy", "cycle", "cycles", "round", "rounds", "session", "sessions", "fraction", "fractions"]) and
        any(k in q_lower for k in ["remain", "remaining", "left", "complete", "completed", "many", "count", "next", "status", "scheduled"])):
        return "chemo_cycle_status"

    # Suggested vs performed tests: "suggested tests", "pending tests"
    if (("suggested" in q_lower or "advised" in q_lower or "ordered" in q_lower) and
        ("test" in q_lower or "perform" in q_lower or "done" in q_lower or "pending" in q_lower)):
        return "suggested_vs_performed"

    # Improvement trend: "am I improving", "getting better", "improvements in my health"
    if any(k in q_lower for k in ["improving", "improvement", "better", "worse", "progress"]):
        if any(k in q_lower for k in ["health", "condition", "status", "test", "lab", "hemoglobin", "sugar", "creatinine"]):
            return "improvement_trend"

    # Specific lab value at a time: "sugar level", "hemoglobin on", "creatinine in"
    if "trend" not in q_lower and any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in [
        "sugar", "glucose", "creatinine", "hemoglobin", "platelet", "bilirubin", "blood sugar"
    ]):
        if any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in ["level", "reading", "value", "status"]) or re.search(r"\b(?:on|in|at)\s+\d", q_lower):
            return "specific_lab_value"

    # --- Original intents ---

    # Diagnoses patterns (checked early for diagnosis-specific inquiries)
    if any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in [
        "diagnosis", "diagnoses", "disease", "diseases", "suffering from", "condition", "conditions",
        "illness", "what cancer", "type of cancer", "pathology finding", "heart disease", "diabetes",
        "hypertension", "infection", "kidney disease", "liver disease"
    ]):
        return "diagnosis_list"

    # Staging & Biomarkers patterns
    if any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in [
        "biomarker", "biomarkers", "her2", "her-2", "her2neu", "er/pr", "estrogen",
        "progesterone", "ki67", "ki-67", "staging", "stage", "tnm", "grade", "nottingham",
        "receptor status", "receptors", "lvef", "ejection fraction", "hba1c", "a1c", "gcs", "nyha", "ckd stage"
    ]):
        return "staging_biomarker"

    # Lab trends patterns (word boundaries prevent 'anc' from matching 'cancer')
    if any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in [
        "hemoglobin", "platelet", "platelets", "creatinine", "sgot", "sgpt", "lft", "kft",
        "bilirubin", "lab test", "lab tests", "lab value", "lab values", "blood count",
        "trend", "trends", "wbc", "anc", "tlc", "calcium", "phosphate", "urea", "bun",
        "troponin", "bnp", "sodium", "potassium", "chloride", "cholesterol", "lipid", "vitals", "blood pressure", "pulse", "spo2"
    ]):
        return "lab_trend"

    # Medication / Treatment / Regimen / Procedure patterns
    if any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in [
        "medication", "medications", "medicine", "medicines", "drug", "drugs", "chemo",
        "chemotherapy", "regimen", "regimens", "treatment", "treatments", "cycle", "dose",
        "paclitaxel", "doxorubicin", "trastuzumab", "tamoxifen", "letrozole", "surgery",
        "operation", "procedure", "procedures", "underwent", "mrm", "mastectomy",
        "aspirin", "statin", "atorvastatin", "metformin", "insulin", "amlodipine", "losartan",
        "antibiotic", "ceftriaxone", "azithromycin", "meropenem", "paracetamol", "angioplasty",
        "stent", "bypass", "cabg", "dialysis", "endoscopy", "colonoscopy", "intubation"
    ]):
        return "medication_history"

    # LLM classification fallback for ambiguous questions
    prompt = (
        f"Classify the following medical question into exactly ONE of these categories:\n"
        f"- chemo_cycle_status\n"
        f"- suggested_vs_performed\n"
        f"- improvement_trend\n"
        f"- specific_lab_value\n"
        f"- lab_trend\n"
        f"- diagnosis_list\n"
        f"- medication_history\n"
        f"- staging_biomarker\n"
        f"- open_ended\n\n"
        f"Question: \"{question}\"\n\n"
        f"Respond ONLY with the single category name."
    )
    try:
        resp = chat([{"role": "user", "content": prompt}], task="inference").strip().lower()
        valid_intents = {
            "chemo_cycle_status", "suggested_vs_performed", "improvement_trend",
            "specific_lab_value", "lab_trend", "diagnosis_list",
            "medication_history", "staging_biomarker", "open_ended",
        }
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
        driver = get_driver_for_patient(patient_ids[0]) if patient_ids else get_driver()
        should_close_driver = True

    schema_description = """
    Nodes:
    - Patient {patient_id, display_label}
    - Report {report_id, patient_id, report_type, report_date, comparison_text, compared_to_ref}
    - Diagnosis {canonical_name}
    - Procedure {canonical_name}
    - Medication {canonical_name}
    - Regimen {canonical_name}
    - LabTest {canonical_name}
    - LabResult {value, unit, date, result_date, abnormal_flag}
    - Staging {t, n, m, date}
    - Biomarker {marker, value, date}
    - ChemoAdministration {admin_id, cycle_number, date, regimen, medications}
    - TreatmentPlan {plan_id, regimen, planned_cycles, confidence_note}
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
    - (r:Report)-[rel:SUGGESTS_TEST {date}]->(lt:LabTest or pr:Procedure)
    - (r:Report)-[rel:HAS_CHEMO_ADMIN]->(ca:ChemoAdministration)
    - (r:Report)-[rel:HAS_TREATMENT_PLAN]->(tp:TreatmentPlan)
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
        resp = chat([{"role": "user", "content": prompt}], task="inference").strip()
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

    all_facts: List[Dict[str, Any]] = []

    # Loop per patient_id (SRS FR-6.2.4)
    for p_id in patient_ids:
        p_driver = driver or get_driver_for_patient(p_id)
        p_should_close = (driver is None)
        try:
            if intent == "chemo_cycle_status":
                facts = get_chemo_cycle_status(patient_id=p_id, driver=p_driver)
                all_facts.extend(facts)

            elif intent == "suggested_vs_performed":
                facts = get_suggested_vs_performed(patient_id=p_id, driver=p_driver)
                all_facts.extend(facts)

            elif intent == "improvement_trend":
                # Extract optional test name
                test_match = None
                for candidate in ["hemoglobin", "platelet", "creatinine", "sgot", "sgpt", "bilirubin", "sugar", "wbc"]:
                    if candidate in question.lower():
                        test_match = candidate
                        break
                facts = get_improvement_trend(patient_id=p_id, test_name=test_match, driver=p_driver)
                all_facts.extend(facts)

            elif intent == "specific_lab_value":
                # Extract test name and date hints
                test_match = "Random Blood Sugar"  # default for sugar queries
                for candidate_name, canonical in [("hemoglobin", "Hemoglobin"), ("platelet", "Platelet Count"),
                                                   ("creatinine", "Serum Creatinine"), ("sugar", "Random Blood Sugar"),
                                                   ("glucose", "Random Blood Sugar"), ("sgot", "SGOT (AST)"),
                                                   ("sgpt", "SGPT (ALT)")]:
                    if candidate_name in question.lower():
                        test_match = canonical
                        break
                # Try to extract a date/period from the question
                date_match = None
                date_patterns = re.findall(r'(\d{4}[-/]\d{1,2}|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{4}|\d{1,2}[-/]\d{4})', question.lower())
                if date_patterns:
                    date_match = date_patterns[0]
                facts = get_specific_lab_value(patient_id=p_id, test_name=test_match, target_date=date_match, driver=p_driver)
                all_facts.extend(facts)

            elif intent == "lab_trend":
                # Extract optional test name from question if present
                test_match = None
                for candidate in ["hemoglobin", "platelet", "creatinine", "sgot", "sgpt", "bilirubin"]:
                    if candidate in question.lower():
                        test_match = candidate
                        break
                facts = get_lab_trend(patient_id=p_id, test_name=test_match, driver=p_driver)
                all_facts.extend(facts)

            elif intent == "diagnosis_list":
                facts = get_diagnoses(patient_id=p_id, driver=p_driver)
                all_facts.extend(facts)

            elif intent == "medication_history":
                facts = get_medication_history(patient_id=p_id, driver=p_driver)
                all_facts.extend(facts)

            elif intent == "staging_biomarker":
                facts = get_staging_and_biomarkers(patient_id=p_id, driver=p_driver)
                all_facts.extend(facts)

            else:
                # Open-ended query
                facts = open_ended_query(patient_ids=[p_id], question=question, driver=p_driver)
                all_facts.extend(facts)

        finally:
            if p_should_close and p_driver is not None:
                p_driver.close()

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
