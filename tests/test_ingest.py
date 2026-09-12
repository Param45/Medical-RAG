import tempfile
import unittest
from pathlib import Path
from reportlab.pdfgen import canvas

from ingest import discover_patient_pdfs, render_pdf_to_pages, ingest_all


def create_dummy_pdf(output_path: Path, text: str = "Test PDF Page") -> Path:
    """Create a minimal 1-page PDF fixture."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path))
    c.drawString(100, 750, text)
    c.save()
    return output_path


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.raw_dir = self.root_path / "raw"
        self.pages_dir = self.root_path / "pages"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy patient PDF
        self.pdf_file = self.raw_dir / "patient_test.pdf"
        create_dummy_pdf(self.pdf_file, text="Patient Test Record")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discover_patient_pdfs(self):
        discovered = discover_patient_pdfs(raw_dir=self.raw_dir)
        self.assertIn("patient_test", discovered)
        self.assertEqual(discovered["patient_test"], self.pdf_file)

    def test_render_pdf_to_pages(self):
        page_paths = render_pdf_to_pages(
            pdf_path=self.pdf_file,
            patient_id="patient_test",
            output_dir=self.pages_dir,
            dpi=100,
        )
        self.assertEqual(len(page_paths), 1)
        expected_page = self.pages_dir / "patient_test" / "page_1.png"
        self.assertEqual(page_paths[0], expected_page)
        self.assertTrue(expected_page.exists())

        # Test idempotency - re-running skips re-rendering
        mtime_before = expected_page.stat().st_mtime
        page_paths_second = render_pdf_to_pages(
            pdf_path=self.pdf_file,
            patient_id="patient_test",
            output_dir=self.pages_dir,
            dpi=100,
        )
        self.assertEqual(len(page_paths_second), 1)
        self.assertEqual(expected_page.stat().st_mtime, mtime_before)

    def test_ingest_all(self):
        results = ingest_all(
            raw_dir=self.raw_dir,
            output_base_dir=self.pages_dir,
            dpi=100,
        )
        self.assertIn("patient_test", results)
        self.assertEqual(len(results["patient_test"]), 1)
        self.assertTrue((self.pages_dir / "patient_test" / "page_1.png").exists())


if __name__ == "__main__":
    unittest.main()
