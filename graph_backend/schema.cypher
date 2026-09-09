-- Medical Records RAG Demo — Neo4j Schema Initialization (SRS §6.1.2, §12.1)
--
-- Idempotent constraint creation for the fixed graph schema.
-- Run via graph_backend/build.py's run_schema_init() function.
--
-- Node labels with uniqueness constraints:
--   Patient.patient_id, Diagnosis.canonical_name,
--   Medication.canonical_name, LabTest.canonical_name
--
-- Implementation pending — see BUILD_GUIDE Task 2.2.

CREATE CONSTRAINT patient_id_unique IF NOT EXISTS
FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE;

CREATE CONSTRAINT diagnosis_name_unique IF NOT EXISTS
FOR (d:Diagnosis) REQUIRE d.canonical_name IS UNIQUE;

CREATE CONSTRAINT medication_name_unique IF NOT EXISTS
FOR (m:Medication) REQUIRE m.canonical_name IS UNIQUE;

CREATE CONSTRAINT labtest_name_unique IF NOT EXISTS
FOR (lt:LabTest) REQUIRE lt.canonical_name IS UNIQUE;
