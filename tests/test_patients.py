import unittest
from patients import PATIENTS, get_display_label, all_patient_ids


class TestPatientsRegistry(unittest.TestCase):
    def test_patients_registry(self):
        self.assertIsInstance(PATIENTS, dict)
        self.assertEqual(PATIENTS.get("patient_a"), "Patient A")
        self.assertEqual(PATIENTS.get("patient_b"), "Patient B")
        self.assertEqual(len(PATIENTS), 2)

    def test_get_display_label(self):
        self.assertEqual(get_display_label("patient_a"), "Patient A")
        self.assertEqual(get_display_label("patient_b"), "Patient B")
        self.assertEqual(get_display_label("unknown_patient"), "unknown_patient")

    def test_all_patient_ids(self):
        patient_ids = all_patient_ids()
        self.assertIsInstance(patient_ids, list)
        self.assertEqual(patient_ids, ["patient_a", "patient_b"])


if __name__ == "__main__":
    unittest.main()
