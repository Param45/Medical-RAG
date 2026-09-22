// Medical Records RAG Demo — Neo4j Schema Initialization (SRS §6.1.2, §12.1)
//
// Idempotent constraint creation for the fixed graph schema.
// Run via graph_backend/build.py's run_schema_init() function.
//
// Node labels with uniqueness constraints:
//   Patient.patient_id, Diagnosis.canonical_name,
//   Medication.canonical_name, LabTest.canonical_name,
//   ChemoAdministration.admin_id, TreatmentPlan.plan_id,
//   Procedure.canonical_name

CREATE CONSTRAINT patient_id_unique IF NOT EXISTS
FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE;

CREATE CONSTRAINT diagnosis_name_unique IF NOT EXISTS
FOR (d:Diagnosis) REQUIRE d.canonical_name IS UNIQUE;

CREATE CONSTRAINT medication_name_unique IF NOT EXISTS
FOR (m:Medication) REQUIRE m.canonical_name IS UNIQUE;

CREATE CONSTRAINT labtest_name_unique IF NOT EXISTS
FOR (lt:LabTest) REQUIRE lt.canonical_name IS UNIQUE;

CREATE CONSTRAINT procedure_name_unique IF NOT EXISTS
FOR (pr:Procedure) REQUIRE pr.canonical_name IS UNIQUE;

CREATE CONSTRAINT chemo_admin_id_unique IF NOT EXISTS
FOR (ca:ChemoAdministration) REQUIRE ca.admin_id IS UNIQUE;

CREATE CONSTRAINT med_admin_id_unique IF NOT EXISTS
FOR (ma:MedicationAdministration) REQUIRE ma.admin_id IS UNIQUE;

CREATE CONSTRAINT treatment_plan_id_unique IF NOT EXISTS
FOR (tp:TreatmentPlan) REQUIRE tp.plan_id IS UNIQUE;
