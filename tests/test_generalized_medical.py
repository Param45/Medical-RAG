"""
Unit tests verifying generalization of the Medical RAG system across all patient types
and clinical domains (Cardiology, Endocrinology/Diabetes, Pulmonology, Nephrology,
Gastroenterology, Neurology, Infectious Diseases, and General Medicine).
"""

from pathlib import Path
import pytest

from normalize import (
    compute_abnormal_flag,
    dictionary_match,
    load_normalization_dictionaries,
    load_lab_reference_ranges,
    regex_parsers,
    canonicalize_lab_name,
)
from split_reports import match_report_anchor
from graph_backend.retrieve import classify_intent
from patients import (
    PATIENTS,
    get_display_label,
    all_patient_ids,
    discover_patients,
    register_patient,
    format_patient_label,
)


class TestGeneralizedDictionaryMatch:
    """Verify dictionary matching across general medical specialties."""

    def test_cardiovascular_terms(self):
        cases = [
            ("HTN", "Essential Hypertension", "Diagnosis"),
            ("CAD", "Coronary Artery Disease", "Diagnosis"),
            ("MI", "Myocardial Infarction", "Diagnosis"),
            ("STEMI", "ST-Elevation Myocardial Infarction", "Diagnosis"),
            ("CHF", "Congestive Heart Failure", "Diagnosis"),
            ("AF", "Atrial Fibrillation", "Diagnosis"),
            ("ECG", "Electrocardiography", "Procedure"),
            ("CAG", "Coronary Angiography", "Procedure"),
            ("CABG", "Coronary Artery Bypass Graft", "Procedure"),
            ("PTCA", "Percutaneous Transluminal Coronary Angioplasty", "Procedure"),
            ("ASPIRIN", "Aspirin", "Medication"),
            ("ATORVASTATIN", "Atorvastatin", "Medication"),
            ("METOPROLOL", "Metoprolol", "Medication"),
            ("AMLODIPINE", "Amlodipine", "Medication"),
        ]
        for term, expected_canonical, expected_type in cases:
            res = dictionary_match(term)
            assert res is not None, f"Expected match for {term}"
            assert res.normalized_term == expected_canonical
            assert res.entity_type == expected_type

    def test_endocrine_and_diabetes_terms(self):
        cases = [
            ("T2DM", "Type 2 Diabetes Mellitus", "Diagnosis"),
            ("T1DM", "Type 1 Diabetes Mellitus", "Diagnosis"),
            ("DKA", "Diabetic Ketoacidosis", "Diagnosis"),
            ("HYPOTHYROIDISM", "Hypothyroidism", "Diagnosis"),
            ("METFORMIN", "Metformin", "Medication"),
            ("INSULIN", "Insulin", "Medication"),
            ("LEVOTHYROXINE", "Levothyroxine", "Medication"),
            ("HBA1C", "Glycated Hemoglobin (HbA1c)", "LabTest"),
            ("PPBS", "Postprandial Blood Sugar", "LabTest"),
            ("FBS", "Fasting Blood Sugar", "LabTest"),
            ("TSH", "Thyroid Stimulating Hormone", "LabTest"),
        ]
        for term, expected_canonical, expected_type in cases:
            res = dictionary_match(term)
            assert res is not None, f"Expected match for {term}"
            assert res.normalized_term == expected_canonical
            assert res.entity_type == expected_type

    def test_pulmonology_and_respiratory_terms(self):
        cases = [
            ("COPD", "Chronic Obstructive Pulmonary Disease", "Diagnosis"),
            ("ASTHMA", "Bronchial Asthma", "Diagnosis"),
            ("PNEUMONIA", "Pneumonia", "Diagnosis"),
            ("ARDS", "Acute Respiratory Distress Syndrome", "Diagnosis"),
            ("PFT", "Pulmonary Function Test", "Procedure"),
            ("SPIROMETRY", "Spirometry", "Procedure"),
            ("SALBUTAMOL", "Salbutamol", "Medication"),
            ("BUDESONIDE", "Budesonide", "Medication"),
        ]
        for term, expected_canonical, expected_type in cases:
            res = dictionary_match(term)
            assert res is not None, f"Expected match for {term}"
            assert res.normalized_term == expected_canonical
            assert res.entity_type == expected_type

    def test_nephrology_and_renal_terms(self):
        cases = [
            ("CKD", "Chronic Kidney Disease", "Diagnosis"),
            ("AKI", "Acute Kidney Injury", "Diagnosis"),
            ("ESRD", "End-Stage Renal Disease", "Diagnosis"),
            ("HEMODIALYSIS", "Hemodialysis", "Procedure"),
            ("HD", "Hemodialysis", "Procedure"),
            ("BUN", "Blood Urea Nitrogen", "LabTest"),
            ("EGFR", "Estimated Glomerular Filtration Rate", "LabTest"),
        ]
        for term, expected_canonical, expected_type in cases:
            res = dictionary_match(term)
            assert res is not None, f"Expected match for {term}"
            assert res.normalized_term == expected_canonical
            assert res.entity_type == expected_type

    def test_neurology_and_gastro_terms(self):
        cases = [
            ("CVA", "Cerebrovascular Accident (Stroke)", "Diagnosis"),
            ("TIA", "Transient Ischemic Attack", "Diagnosis"),
            ("GERD", "Gastroesophageal Reflux Disease", "Diagnosis"),
            ("CIRRHOSIS", "Cirrhosis of Liver", "Diagnosis"),
            ("PUD", "Peptic Ulcer Disease", "Diagnosis"),
            ("ENDOSCOPY", "Upper GI Endoscopy", "Procedure"),
            ("COLONOSCOPY", "Colonoscopy", "Procedure"),
            ("EEG", "Electroencephalography", "Procedure"),
            ("PANTOPRAZOLE", "Pantoprazole", "Medication"),
            ("PARACETAMOL", "Paracetamol", "Medication"),
        ]
        for term, expected_canonical, expected_type in cases:
            res = dictionary_match(term)
            assert res is not None, f"Expected match for {term}"
            assert res.normalized_term == expected_canonical
            assert res.entity_type == expected_type


class TestGeneralizedRegexParsers:
    """Verify structured regex extraction for vitals, cardiac, glycemic, and clinical severity scores."""

    def test_vital_signs_regex(self):
        text = "Patient presented with BP: 130/85 mmHg, HR: 78 bpm, RR: 18 /min, SpO2: 98%, Temp: 98.6 F, BMI: 24.2 kg/m2"
        entities = regex_parsers(text)

        findings = {e.metadata.get("vital"): e.normalized_term for e in entities if e.metadata and "vital" in e.metadata}
        assert "BP" in findings
        assert "130/85" in findings["BP"]
        assert "HR" in findings
        assert "78" in findings["HR"]
        assert "RR" in findings
        assert "18" in findings["RR"]
        assert "SpO2" in findings
        assert "98%" in findings["SpO2"]
        assert "Temperature" in findings
        assert "98.6" in findings["Temperature"]
        assert "BMI" in findings
        assert "24.2" in findings["BMI"]

    def test_cardiac_and_glycemic_regex(self):
        text = "Echocardiography showed LVEF: 55% with normal LV function. Lab report: HbA1c: 7.2%."
        entities = regex_parsers(text)

        lvef_ents = [e for e in entities if e.metadata and e.metadata.get("marker") == "LVEF"]
        assert len(lvef_ents) >= 1
        assert "55%" in lvef_ents[0].normalized_term

        hba1c_ents = [e for e in entities if e.metadata and e.metadata.get("marker") == "HbA1c"]
        assert len(hba1c_ents) >= 1
        assert "7.2%" in hba1c_ents[0].normalized_term

    def test_clinical_severity_scores_regex(self):
        text = "Patient neurological exam GCS: 15. Cardiac status NYHA Class II. Renal staging CKD Stage 3b."
        entities = regex_parsers(text)

        gcs_ents = [e for e in entities if e.metadata and e.metadata.get("score") == "GCS"]
        assert len(gcs_ents) >= 1
        assert "15" in gcs_ents[0].normalized_term

        nyha_ents = [e for e in entities if e.metadata and e.metadata.get("score") == "NYHA"]
        assert len(nyha_ents) >= 1
        assert "II" in nyha_ents[0].normalized_term

        ckd_ents = [e for e in entities if e.metadata and e.metadata.get("staging") == "CKD"]
        assert len(ckd_ents) >= 1
        assert "3B" in ckd_ents[0].normalized_term


class TestGeneralizedLabAbnormalFlag:
    """Verify compute_abnormal_flag with expanded standard lab ranges."""

    def test_electrolytes_and_glycemic(self):
        # Sodium: 136 - 145
        assert compute_abnormal_flag("Serum Sodium", "138") == "normal"
        assert compute_abnormal_flag("Serum Sodium", "130") == "low"
        assert compute_abnormal_flag("Serum Sodium", "150") == "high"

        # Potassium: 3.5 - 5.0
        assert compute_abnormal_flag("Serum Potassium", "4.2") == "normal"
        assert compute_abnormal_flag("Serum Potassium", "3.1") == "low"
        assert compute_abnormal_flag("Serum Potassium", "5.8") == "high"

        # HbA1c: 4.0 - 5.6 %
        assert compute_abnormal_flag("Glycated Hemoglobin (HbA1c)", "5.2") == "normal"
        assert compute_abnormal_flag("Glycated Hemoglobin (HbA1c)", "8.1") == "high"

    def test_cardiac_and_lipid_markers(self):
        # Cardiac Troponin I: 0.0 - 0.04 ng/mL
        assert compute_abnormal_flag("Cardiac Troponin I", "0.02") == "normal"
        assert compute_abnormal_flag("Cardiac Troponin I", "0.85") == "high"

        # Total Cholesterol: 125 - 200 mg/dL
        assert compute_abnormal_flag("Total Cholesterol", "175") == "normal"
        assert compute_abnormal_flag("Total Cholesterol", "260") == "high"


class TestGeneralizedReportAnchors:
    """Verify report anchor matching for new clinical document types."""

    def test_cardiology_ecg_anchor(self):
        text = "HOSPITAL HEART INSTITUTE\n12-LEAD ECG REPORT\nNormal sinus rhythm, HR 72 bpm"
        assert match_report_anchor(text) == "CARDIOLOGY_ECG"

    def test_pft_anchor(self):
        text = "DEPARTMENT OF PULMONARY MEDICINE\nPULMONARY FUNCTION TEST REPORT\nFEV1/FVC: 78%"
        assert match_report_anchor(text) == "PULMONARY_FUNCTION_TEST"

    def test_radiology_mri_and_xray_anchors(self):
        mri_text = "DEPARTMENT OF RADIODIAGNOSIS\nMRI BRAIN SCAN REPORT WITH CONTRAST"
        assert match_report_anchor(mri_text) == "RADIOLOGY_MRI"

        xray_text = "CITY IMAGING CENTRE\nCHEST X-RAY PA VIEW REPORT\nNormal cardiothoracic ratio"
        assert match_report_anchor(xray_text) == "RADIOLOGY_XRAY"

    def test_endoscopy_and_flowsheet_anchors(self):
        endo_text = "DEPARTMENT OF GASTROENTEROLOGY\nUPPER GI ENDOSCOPY REPORT\nNormal mucosa"
        assert match_report_anchor(endo_text) == "ENDOSCOPY_REPORT"

        vitals_text = "ICU WARD MONITORING\nCLINICAL FLOWSHEET\nTime BP Pulse Temp SpO2"
        assert match_report_anchor(vitals_text) == "CLINICAL_FLOWSHEET"

    def test_medication_admin_and_consultation_anchors(self):
        mar_text = "INPATIENT NURSING WARD\nMEDICATION ADMINISTRATION RECORD (MAR)\nDrug Dose Route"
        assert match_report_anchor(mar_text) == "MEDICATION_ADMIN_RECORD"

        opd_text = "GENERAL MEDICINE CLINIC\nOUTPATIENT CONSULTATION NOTE\nHistory and Assessment"
        assert match_report_anchor(opd_text) == "CONSULTATION_NOTE"


class TestGeneralizedIntentClassification:
    """Verify query intent classification handles general clinical queries."""

    def test_cardiac_and_diabetes_queries(self):
        assert classify_intent("What is my heart disease diagnosis?") == "diagnosis_list"
        assert classify_intent("List all diabetes and hypertension diagnoses") == "diagnosis_list"
        assert classify_intent("What is my latest LVEF and ejection fraction?") == "staging_biomarker"
        assert classify_intent("What is my HbA1c and A1C score?") == "staging_biomarker"

    def test_general_medications_and_cycles(self):
        assert classify_intent("What medications including aspirin and atorvastatin were given?") == "medication_history"
        assert classify_intent("How many hemodialysis sessions or cycles has the patient completed?") == "chemo_cycle_status"

    def test_general_labs_and_vitals(self):
        assert classify_intent("What are the trends in my sodium, potassium, and troponin levels?") == "lab_trend"


class TestDynamicPatientDiscovery:
    """Verify dynamic discovery and registration of patients."""

    def test_format_patient_label(self):
        assert format_patient_label("patient_c") == "Patient C"
        assert format_patient_label("john_doe") == "John Doe"
        assert format_patient_label("sarah_connor") == "Sarah Connor"

    def test_register_patient_dynamically(self):
        register_patient("patient_test_x", "Patient Test X")
        assert "patient_test_x" in PATIENTS
        assert get_display_label("patient_test_x") == "Patient Test X"
        # Cleanup
        PATIENTS.pop("patient_test_x", None)
