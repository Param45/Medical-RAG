"""
OCR Smoke Test for Medical RAG (Task 0.4)
Validates local MinerU (PytorchPaddleOCR) execution on CPU with bilingual (English + Hindi/Devanagari) support.

Usage:
    python ocr_smoketest.py                          # Tests Page 0 of data/raw/patient_a.pdf
    python ocr_smoketest.py --pdf data/raw/patient_a.pdf --page 0
    python ocr_smoketest.py --pdf data/raw/patient_a.pdf --page 0 --lang en
    python ocr_smoketest.py --test-hindi             # Tests a synthetic Devanagari/Hindi consent sample
    python ocr_smoketest.py --image path/to/page.png
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

# Ensure magic-pdf config points to cpu and local models directory
os.environ.setdefault("MINERU_TOOLS_CONFIG_JSON", str(Path(__file__).parent / "magic-pdf.json"))

try:
    from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.pytorch_paddle import PytorchPaddleOCR
except ImportError as err:
    print(f"[ERROR] Failed to import MinerU / magic_pdf: {err}", file=sys.stderr)
    print("Ensure the virtual environment is activated and 'magic-pdf[full]' is installed.", file=sys.stderr)
    sys.exit(1)


# Global cached OCR instances to avoid reloading weights repeatedly
_OCR_INSTANCES = {}


def get_ocr_engine(lang: str = "en") -> PytorchPaddleOCR:
    """
    Get or instantiate a local PytorchPaddleOCR engine for the given language.
    Runs on CPU. Model weights are loaded from ./models/OCR/paddleocr_torch/.
    """
    lang_key = "hi" if lang in ("hi", "devanagari", "hindi") else "en"
    if lang_key not in _OCR_INSTANCES:
        print(f"[*] Initializing local MinerU OCR engine (lang='{lang_key}', device='cpu')...")
        t0 = time.time()
        _OCR_INSTANCES[lang_key] = PytorchPaddleOCR(lang=lang_key)
        print(f"[*] Engine loaded in {time.time() - t0:.2f}s (cached in-memory)")
    return _OCR_INSTANCES[lang_key]


def is_devanagari(char: str) -> bool:
    """Check if a Unicode character is in the Devanagari block."""
    return "\u0900" <= char <= "\u097F"


def score_ocr_result(ocr_res) -> tuple[float, int, int]:
    """
    Calculate average confidence, total character count, and Devanagari character count.
    """
    if not ocr_res or not ocr_res[0]:
        return 0.0, 0, 0

    scores = []
    total_chars = 0
    devanagari_chars = 0

    for item in ocr_res[0]:
        # item format: [box, (text, score)]
        if len(item) >= 2 and isinstance(item[1], (list, tuple)):
            text, score = item[1]
            scores.append(float(score))
            total_chars += len(text)
            devanagari_chars += sum(1 for c in text if is_devanagari(c))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score, total_chars, devanagari_chars


def run_bilingual_ocr(image: np.ndarray, strategy: str = "auto") -> dict:
    """
    Run OCR on an image (BGR numpy array).
    
    Strategies:
      - 'en': Run English OCR only.
      - 'hi': Run Hindi / Devanagari OCR only.
      - 'auto': Run bilingual comparison (two-pass) and select the best matching pass.
    """
    if strategy in ("en", "english"):
        engine = get_ocr_engine("en")
        t0 = time.time()
        res = engine.ocr(image)
        elapsed = time.time() - t0
        avg_score, total_chars, dev_chars = score_ocr_result(res)
        return {
            "lang": "en",
            "script": "latin",
            "result": res,
            "avg_confidence": avg_score,
            "elapsed_seconds": elapsed,
            "total_chars": total_chars,
            "devanagari_chars": dev_chars,
        }

    if strategy in ("hi", "hindi", "devanagari"):
        engine = get_ocr_engine("hi")
        t0 = time.time()
        res = engine.ocr(image)
        elapsed = time.time() - t0
        avg_score, total_chars, dev_chars = score_ocr_result(res)
        return {
            "lang": "hi",
            "script": "devanagari",
            "result": res,
            "avg_confidence": avg_score,
            "elapsed_seconds": elapsed,
            "total_chars": total_chars,
            "devanagari_chars": dev_chars,
        }

    # Auto strategy: Run both passes and compare
    print("[*] Running bilingual two-pass OCR strategy (English + Hindi)...")
    
    # 1. English pass
    eng_engine = get_ocr_engine("en")
    t0 = time.time()
    res_en = eng_engine.ocr(image)
    t_en = time.time() - t0
    score_en, chars_en, dev_en = score_ocr_result(res_en)

    # 2. Hindi pass
    hi_engine = get_ocr_engine("hi")
    t0 = time.time()
    res_hi = hi_engine.ocr(image)
    t_hi = time.time() - t0
    score_hi, chars_hi, dev_hi = score_ocr_result(res_hi)

    # Decision logic:
    # If the Hindi pass yielded a significant number of valid Devanagari characters, select Hindi
    is_hindi = (dev_hi >= 10 and dev_hi / max(chars_hi, 1) > 0.15) or (score_hi > score_en and dev_hi > 5)

    if is_hindi:
        return {
            "lang": "hi",
            "script": "devanagari",
            "result": res_hi,
            "avg_confidence": score_hi,
            "elapsed_seconds": t_en + t_hi,
            "total_chars": chars_hi,
            "devanagari_chars": dev_hi,
            "comparison": {
                "en": {"avg_confidence": score_en, "chars": chars_en},
                "hi": {"avg_confidence": score_hi, "chars": chars_hi, "devanagari_chars": dev_hi},
            },
        }
    else:
        return {
            "lang": "en",
            "script": "latin",
            "result": res_en,
            "avg_confidence": score_en,
            "elapsed_seconds": t_en + t_hi,
            "total_chars": chars_en,
            "devanagari_chars": dev_en,
            "comparison": {
                "en": {"avg_confidence": score_en, "chars": chars_en},
                "hi": {"avg_confidence": score_hi, "chars": chars_hi, "devanagari_chars": dev_hi},
            },
        }


def render_pdf_page(pdf_path: str, page_num: int = 0, dpi: int = 150) -> np.ndarray:
    """Render a single PDF page into an OpenCV BGR numpy array using PyMuPDF (fitz)."""
    import fitz

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    if page_num < 0 or page_num >= len(doc):
        raise IndexError(f"Page index {page_num} out of range (PDF has {len(doc)} pages: 0 to {len(doc)-1})")

    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def create_synthetic_hindi_sample() -> np.ndarray:
    """Create a synthetic bilingual consent form image for testing."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (900, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_paths = [
        "C:/Windows/Fonts/Nirmala.ttc",
        "C:/Windows/Fonts/mangal.ttf",
        "C:/Windows/Fonts/aparaj.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 32, index=0 if fp.endswith(".ttc") else None)
                break
            except Exception:
                pass

    if font is None:
        font = ImageFont.load_default()

    draw.text((40, 30), "सहमति पत्र (Informed Consent Form)", fill=(0, 0, 0), font=font)
    draw.text((40, 90), "रोगी का नाम: श्रीमती सूरज मुखी (Mrs. Suraj Mukhi)", fill=(0, 0, 0), font=font)
    draw.text((40, 150), "अस्पताल: अखिल भारतीय आयुर्विज्ञान संस्थान, नई दिल्ली", fill=(0, 0, 0), font=font)
    draw.text((40, 210), "उपचार: कीमोथेरेपी और रेडियोथेरेपी (Chemotherapy)", fill=(0, 0, 0), font=font)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def print_ocr_output(output: dict, max_lines: int = 25):
    """Format and print OCR output to console."""
    print("=" * 70)
    print(f"OCR RESULT | Script: {output['script'].upper()} (lang={output['lang']})")
    print(f"Average Confidence: {output['avg_confidence']:.2%}")
    print(f"Execution Time: {output['elapsed_seconds']:.2f}s (CPU)")
    print(f"Total Characters: {output['total_chars']} (Devanagari: {output['devanagari_chars']})")
    if "comparison" in output:
        comp = output["comparison"]
        print(f"Strategy Comparison -> English: {comp['en']['avg_confidence']:.2%} ({comp['en']['chars']} chars) | Hindi: {comp['hi']['avg_confidence']:.2%} ({comp['hi']['chars']} chars, {comp['hi']['devanagari_chars']} devanagari)")
    print("-" * 70)

    res = output["result"]
    if not res or not res[0]:
        print("[No text recognized on this page]")
        print("=" * 70)
        return

    lines = res[0]
    print(f"Recognized Lines ({len(lines)} total, showing first {min(len(lines), max_lines)}):")
    for i, item in enumerate(lines[:max_lines]):
        if len(item) >= 2 and isinstance(item[1], (list, tuple)):
            text, score = item[1]
            print(f"  [{i+1:02d}] (conf: {float(score):.2f})  {text}")

    if len(lines) > max_lines:
        print(f"  ... and {len(lines) - max_lines} more lines.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Medical RAG - MinerU OCR Smoke Test")
    parser.add_argument("--pdf", type=str, default="data/raw/patient_a.pdf", help="Path to PDF file")
    parser.add_argument("--page", type=int, default=0, help="Page index to test (0-indexed)")
    parser.add_argument("--image", type=str, default=None, help="Path to image file (overrides --pdf)")
    parser.add_argument("--lang", type=str, default="auto", choices=["auto", "en", "hi"], help="OCR language strategy")
    parser.add_argument("--test-hindi", action="store_true", help="Run smoke test on synthetic Hindi consent form sample")
    parser.add_argument("--max-lines", type=int, default=25, help="Maximum lines to display in output")

    args = parser.parse_args()

    print("=== MinerU Local OCR Smoke Test (Task 0.4) ===")
    
    # 1. Load image
    if args.test_hindi:
        print("[*] Generating synthetic Hindi consent form page for testing...")
        image = create_synthetic_hindi_sample()
        source_label = "Synthetic Hindi Consent Sample"
    elif args.image:
        print(f"[*] Loading image: {args.image}")
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        image = cv2.imread(args.image)
        source_label = args.image
    else:
        pdf_path = args.pdf
        print(f"[*] Rendering page {args.page} of {pdf_path}...")
        try:
            image = render_pdf_page(pdf_path, args.page)
            source_label = f"{pdf_path} (Page {args.page})"
        except Exception as e:
            print(f"[ERROR] Failed to load/render PDF: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"[*] Source: {source_label} (dimensions: {image.shape[1]}x{image.shape[0]})")
    
    # 2. Run OCR
    output = run_bilingual_ocr(image, strategy=args.lang)

    # 3. Print results
    print_ocr_output(output, max_lines=args.max_lines)
    print("[SUCCESS] Local MinerU OCR smoke test completed successfully!")


if __name__ == "__main__":
    main()
