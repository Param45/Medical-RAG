"""
Medical Records RAG Demo — OCR Provider Interface & MinerU Implementation (SRS §5.2)

Provides an OCRProvider interface and MinerU implementation.
Handles bilingual (English + Hindi) OCR with confidence scoring and near-duplicate detection.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import difflib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda *args, **kwargs: None

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure magic-pdf config points to CPU and local models directory
os.environ.setdefault("MINERU_TOOLS_CONFIG_JSON", str(Path(__file__).parent / "magic-pdf.json"))

try:
    from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.pytorch_paddle import PytorchPaddleOCR
except ImportError:
    PytorchPaddleOCR = None


@dataclass
class PageOCRResult:
    """
    Structured result of OCR for a single page (SRS §5.2.2).
    """
    page_number: int
    raw_text: str
    is_table: bool
    is_handwritten: bool
    confidence: float
    script: str  # "latin" | "devanagari" | "mixed"
    duplicate_of: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.duplicate_of is None:
            data.pop("duplicate_of", None)
        return data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PageOCRResult":
        return cls(
            page_number=int(d.get("page_number", 0)),
            raw_text=str(d.get("raw_text", "")),
            is_table=bool(d.get("is_table", False)),
            is_handwritten=bool(d.get("is_handwritten", False)),
            confidence=float(d.get("confidence", 0.0)),
            script=str(d.get("script", "latin")),
            duplicate_of=d.get("duplicate_of"),
        )


class OCRProvider(ABC):
    """Abstract interface for OCR providers (SRS §5.2.1)."""

    @abstractmethod
    def ocr_page(self, image_path: Path | str, page_number: Optional[int] = None) -> PageOCRResult:
        """Process a single page image and return PageOCRResult."""
        pass


def is_devanagari(char: str) -> bool:
    """Check if a Unicode character is in the Devanagari block."""
    return "\u0900" <= char <= "\u097F"


def score_paddle_ocr_result(ocr_res) -> Tuple[float, int, int, List[Tuple[str, float]]]:
    """
    Calculate average confidence, total characters, Devanagari characters, and text lines.
    """
    if not ocr_res or not ocr_res[0]:
        return 0.0, 0, 0, []

    scores: List[float] = []
    total_chars = 0
    devanagari_chars = 0
    lines: List[Tuple[str, float]] = []

    for item in ocr_res[0]:
        # item format: [box, (text, score)]
        if len(item) >= 2 and isinstance(item[1], (list, tuple)):
            text, score = item[1]
            score_val = float(score)
            scores.append(score_val)
            total_chars += len(text)
            devanagari_chars += sum(1 for c in text if is_devanagari(c))
            lines.append((text, score_val))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score, total_chars, devanagari_chars, lines


def detect_table_heuristic(lines: List[Tuple[str, float]], raw_text: str) -> bool:
    """
    Simple visual/token density heuristic for tabular flowsheets (SRS FR-5.2.3).
    Flowsheets contain high count of short numeric tokens, dates, and column headers.
    """
    flowsheet_keywords = ["FLOWSHEET", "HB/PCV", "PLATELATES", "WBC", "ANC", "ESR", "BILIRUBIN", "SGOT", "SGPT", "CREATININE"]
    text_upper = raw_text.upper()
    if any(kw in text_upper for kw in flowsheet_keywords):
        return True

    # Check for dense short numeric tokens / grid lines
    tokens = raw_text.split()
    if len(tokens) >= 20:
        numeric_count = sum(1 for t in tokens if re.match(r"^\d+(\.\d+)?%?$", t))
        if numeric_count / len(tokens) > 0.35:
            return True

    return False


class MinerUProvider(OCRProvider):
    """
    Local MinerU OCR provider supporting bilingual English + Hindi (Devanagari) on CPU.
    """

    def __init__(self):
        if PytorchPaddleOCR is None:
            raise ImportError("magic_pdf is not available in the current environment.")
        self._engines: Dict[str, PytorchPaddleOCR] = {}

    def _get_engine(self, lang: str = "en") -> PytorchPaddleOCR:
        lang_key = "hi" if lang in ("hi", "devanagari", "hindi") else "en"
        if lang_key not in self._engines:
            self._engines[lang_key] = PytorchPaddleOCR(lang=lang_key)
        return self._engines[lang_key]

    def ocr_page(self, image_path: Path | str, page_number: Optional[int] = None) -> PageOCRResult:
        img_path = Path(image_path)
        if page_number is None:
            match = re.search(r"page_(\d+)", img_path.name)
            page_number = int(match.group(1)) if match else 1

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load image at {img_path}")

        # Run dual-pass strategy: English + Hindi
        eng_engine = self._get_engine("en")
        res_en = eng_engine.ocr(img_bgr)
        score_en, chars_en, dev_en, lines_en = score_paddle_ocr_result(res_en)

        hi_engine = self._get_engine("hi")
        res_hi = hi_engine.ocr(img_bgr)
        score_hi, chars_hi, dev_hi, lines_hi = score_paddle_ocr_result(res_hi)

        # Decide which pass to keep
        is_hindi = (dev_hi >= 10 and dev_hi / max(chars_hi, 1) > 0.15) or (score_hi > score_en and dev_hi > 5)

        if is_hindi:
            chosen_lines = lines_hi
            confidence = score_hi
            script = "devanagari" if (dev_hi / max(chars_hi, 1) > 0.60) else "mixed"
        else:
            chosen_lines = lines_en
            confidence = score_en
            script = "mixed" if dev_en > 5 else "latin"

        raw_text = "\n".join(text for text, _ in chosen_lines)
        is_table = detect_table_heuristic(chosen_lines, raw_text)
        
        # is_handwritten defaults to False per SRS FR-5.2.2 / FR-5.5.4
        # MinerU paddleocr does not provide a separate cursive handwriting classifier flag.
        is_handwritten = False

        return PageOCRResult(
            page_number=page_number,
            raw_text=raw_text,
            is_table=is_table,
            is_handwritten=is_handwritten,
            confidence=round(confidence, 4),
            script=script,
        )


def get_ocr_provider() -> OCRProvider:
    """
    Factory function to get configured OCRProvider from .env (SRS §5.2.1, §11).
    """
    load_dotenv()
    engine = os.getenv("OCR_ENGINE", "mineru").lower().strip()
    if engine == "mineru":
        return MinerUProvider()
    raise ValueError(f"Unsupported OCR_ENGINE: '{engine}'. Supported engines: ['mineru']")


def normalize_text_for_dedup(text: str) -> str:
    """Normalize text for near-duplicate comparison."""
    return " ".join(text.lower().split())


def ocr_all_pages(
    patient_id: str,
    pages_dir: str | Path = "data/pages",
    ocr_dir: str | Path = "data/ocr",
    provider: Optional[OCRProvider] = None,
) -> List[PageOCRResult]:
    """
    OCR all page images for a patient sequentially (SRS FR-5.1.4, FR-5.2.4, FR-5.2.5).
    Saves JSON results to data/ocr/{patient_id}/page_{n}.json.
    Skips processing if OCR JSON is newer than page PNG (idempotency).
    Identifies and tags near-duplicate pages (>90% similarity).
    """
    pages_path = Path(pages_dir) / patient_id
    out_dir = Path(ocr_dir) / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pages_path.exists():
        print(f"[*] No pages directory found at {pages_path}")
        return []

    # Collect page files in page_number order
    page_files: List[Tuple[int, Path]] = []
    for file_p in pages_path.glob("page_*.png"):
        m = re.search(r"page_(\d+)\.png$", file_p.name)
        if m:
            page_files.append((int(m.group(1)), file_p))
    page_files.sort(key=lambda x: x[0])

    results: List[PageOCRResult] = []
    seen_pages: List[Tuple[int, str, float]] = []  # (page_number, normalized_text, confidence)

    def _get_provider():
        nonlocal provider
        if provider is None:
            provider = get_ocr_provider()
        return provider

    for page_num, img_path in page_files:
        json_path = out_dir / f"page_{page_num}.json"
        
        # Idempotency check: load existing if newer than image
        if json_path.exists() and json_path.stat().st_mtime >= img_path.stat().st_mtime:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    page_res = PageOCRResult.from_dict(json.load(f))
                print(f"[{patient_id}] page {page_num} loaded from cache")
            except Exception:
                page_res = _get_provider().ocr_page(img_path, page_number=page_num)
        else:
            print(f"[{patient_id}] OCR processing page {page_num}/{len(page_files)}...")
            page_res = _get_provider().ocr_page(img_path, page_number=page_num)

        norm_text = normalize_text_for_dedup(page_res.raw_text)

        # Near-duplicate detection (SRS FR-5.2.5)
        if len(norm_text) > 50:
            for prev_page_num, prev_norm_text, prev_conf in seen_pages:
                ratio = difflib.SequenceMatcher(None, norm_text, prev_norm_text).ratio()
                if ratio > 0.90:
                    if page_res.confidence <= prev_conf:
                        page_res.duplicate_of = prev_page_num
                        print(f"[{patient_id}] page {page_num} is duplicate of page {prev_page_num} (similarity: {ratio:.1%}, conf: {page_res.confidence:.2f} <= {prev_conf:.2f})")
                    else:
                        # Previous page is lower confidence duplicate
                        for r in results:
                            if r.page_number == prev_page_num:
                                r.duplicate_of = page_num
                                prev_json = out_dir / f"page_{prev_page_num}.json"
                                with open(prev_json, "w", encoding="utf-8") as f:
                                    json.dump(r.to_dict(), f, indent=2, ensure_ascii=False)
                                print(f"[{patient_id}] page {prev_page_num} marked duplicate of higher-conf page {page_num} (similarity: {ratio:.1%})")
                    break

        # Save to JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(page_res.to_dict(), f, indent=2, ensure_ascii=False)

        results.append(page_res)
        seen_pages.append((page_num, norm_text, page_res.confidence))

    return results


if __name__ == "__main__":
    from patients import all_patient_ids

    for pid in all_patient_ids():
        print(f"\n{'='*50}\nStarting OCR for patient: {pid}\n{'='*50}")
        ocr_all_pages(pid)
