import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from ocr import (
    PageOCRResult,
    OCRProvider,
    detect_table_heuristic,
    get_ocr_provider,
    MinerUProvider,
    normalize_text_for_dedup,
    ocr_all_pages,
)


class DummyMockOCRProvider(OCRProvider):
    """Mock OCR provider returning predetermined results."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0

    def ocr_page(self, image_path: Path | str, page_number: int = None) -> PageOCRResult:
        self.call_count += 1
        img_p = Path(image_path)
        if page_number is None:
            page_number = int(img_p.stem.replace("page_", ""))
        
        if page_number in self.responses:
            res = self.responses[page_number]
            res.page_number = page_number
            return res

        return PageOCRResult(
            page_number=page_number,
            raw_text=f"Default text for page {page_number}",
            is_table=False,
            is_handwritten=False,
            confidence=0.95,
            script="latin",
        )


class TestOCR(unittest.TestCase):
    def test_page_ocr_result_dict_roundtrip(self):
        res = PageOCRResult(
            page_number=3,
            raw_text="Test raw text",
            is_table=True,
            is_handwritten=False,
            confidence=0.88,
            script="latin",
            duplicate_of=1,
        )
        d = res.to_dict()
        self.assertEqual(d["page_number"], 3)
        self.assertEqual(d["duplicate_of"], 1)

        res_loaded = PageOCRResult.from_dict(d)
        self.assertEqual(res_loaded.page_number, 3)
        self.assertEqual(res_loaded.raw_text, "Test raw text")
        self.assertEqual(res_loaded.is_table, True)
        self.assertEqual(res_loaded.duplicate_of, 1)

    def test_detect_table_heuristic(self):
        # Flowsheet keyword detection
        flowsheet_text = "DEPARTMENT OF ONCOLOGY\nFlowsheet Medical Oncology\nHB/PCV Platelates WBC ANC"
        self.assertTrue(detect_table_heuristic([], flowsheet_text))

        # Plain prose report
        prose_text = "CECT CHEST AND ABDOMEN\nIMPRESSION: No evidence of distant metastatic disease observed."
        self.assertFalse(detect_table_heuristic([], prose_text))

    def test_get_ocr_provider_factory(self):
        with patch("ocr.PytorchPaddleOCR", object):
            with patch.dict("os.environ", {"OCR_ENGINE": "mineru"}):
                provider = get_ocr_provider()
                self.assertIsInstance(provider, MinerUProvider)

        with patch.dict("os.environ", {"OCR_ENGINE": "unsupported_engine"}):
            with self.assertRaises(ValueError):
                get_ocr_provider()

    def test_ocr_all_pages_orchestration_and_dedup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pages_dir = temp_path / "pages"
            ocr_dir = temp_path / "ocr"
            patient_pages = pages_dir / "patient_test"
            patient_pages.mkdir(parents=True, exist_ok=True)

            # Create 3 dummy page images
            for p_num in (1, 2, 3):
                img = Image.new("RGB", (100, 100), color=(255, 255, 255))
                img.save(patient_pages / f"page_{p_num}.png")

            # Setup responses: Page 1 and Page 3 are near-duplicates
            discharge_summary_text = (
                "DISCHARGE SUMMARY HOSPITAL FOR CANCER TREATMENT "
                "PATIENT WAS ADMITTED FOR CHEMOTHERAPY COURSE 5 "
                "VITALS STABLE AT DISCHARGE FOLLOW UP IN 3 WEEKS"
            )
            mock_provider = DummyMockOCRProvider(
                responses={
                    1: PageOCRResult(
                        page_number=1,
                        raw_text=discharge_summary_text,
                        is_table=False,
                        is_handwritten=False,
                        confidence=0.98,
                        script="latin",
                    ),
                    2: PageOCRResult(
                        page_number=2,
                        raw_text="LAB REPORT HAEMATOLOGY HB 12.4 WBC 5600 PLATELETS 250000",
                        is_table=True,
                        is_handwritten=False,
                        confidence=0.92,
                        script="latin",
                    ),
                    3: PageOCRResult(
                        page_number=3,
                        raw_text=discharge_summary_text + " EXTRA NOISE",
                        is_table=False,
                        is_handwritten=False,
                        confidence=0.85,
                        script="latin",
                    ),
                }
            )

            # Run OCR
            results = ocr_all_pages(
                patient_id="patient_test",
                pages_dir=pages_dir,
                ocr_dir=ocr_dir,
                provider=mock_provider,
            )

            self.assertEqual(len(results), 3)
            self.assertEqual(mock_provider.call_count, 3)

            # Verify JSON files exist
            json_1 = ocr_dir / "patient_test" / "page_1.json"
            json_2 = ocr_dir / "patient_test" / "page_2.json"
            json_3 = ocr_dir / "patient_test" / "page_3.json"
            self.assertTrue(json_1.exists())
            self.assertTrue(json_2.exists())
            self.assertTrue(json_3.exists())

            # Verify Page 3 is marked as duplicate_of Page 1
            with open(json_3, "r", encoding="utf-8") as f:
                p3_data = json.load(f)
            self.assertEqual(p3_data.get("duplicate_of"), 1)

            # Verify idempotency: running again should not call provider
            mock_provider.call_count = 0
            cached_results = ocr_all_pages(
                patient_id="patient_test",
                pages_dir=pages_dir,
                ocr_dir=ocr_dir,
                provider=mock_provider,
            )
            self.assertEqual(len(cached_results), 3)
            self.assertEqual(mock_provider.call_count, 0)


if __name__ == "__main__":
    unittest.main()
