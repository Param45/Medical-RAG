"""
Download required MinerU / PDF-Extract-Kit models and local LLM GGUF models.
Caches models under ./models directory (idempotent, skips if already downloaded).
"""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path(__file__).parent / "models"

OCR_FILES_TO_DOWNLOAD = [
    # Layout detection (DocLayout-YOLO)
    ("models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt", "Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"),
    
    # OCR models (PaddleOCR PyTorch weights)
    ("models/OCR/paddleocr_torch/Multilingual_PP-OCRv3_det_infer.pth", "OCR/paddleocr_torch/Multilingual_PP-OCRv3_det_infer.pth"),
    ("models/OCR/paddleocr_torch/en_PP-OCRv5_rec_infer.pth", "OCR/paddleocr_torch/en_PP-OCRv5_rec_infer.pth"),
    ("models/OCR/paddleocr_torch/devanagari_PP-OCRv5_rec_infer.pth", "OCR/paddleocr_torch/devanagari_PP-OCRv5_rec_infer.pth"),
    ("models/OCR/paddleocr_torch/latin_PP-OCRv5_rec_infer.pth", "OCR/paddleocr_torch/latin_PP-OCRv5_rec_infer.pth"),
    ("models/OCR/paddleocr_torch/ch_PP-OCRv5_det_infer.pth", "OCR/paddleocr_torch/ch_PP-OCRv5_det_infer.pth"),
    ("models/OCR/paddleocr_torch/ch_ptocr_mobile_v2.0_cls_infer.pth", "OCR/paddleocr_torch/ch_ptocr_mobile_v2.0_cls_infer.pth"),
]

DEFAULT_LLM_REPO = "unsloth/medgemma-1.5-4b-it-GGUF"
DEFAULT_LLM_FILENAME = "medgemma-1.5-4b-it-Q4_K_M.gguf"


def download_ocr_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Checking/downloading OCR models into: {MODELS_DIR} ---")
    
    repo_id = "opendatalab/PDF-Extract-Kit-1.0"
    for remote_path, local_rel_path in OCR_FILES_TO_DOWNLOAD:
        target_path = MODELS_DIR / local_rel_path
        if target_path.exists() and target_path.stat().st_size > 0:
            print(f"[CACHED] {local_rel_path} already exists ({target_path.stat().st_size:,} bytes). Skipping.")
            continue
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[DOWNLOADING] {remote_path} -> {target_path} ...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            local_dir=str(MODELS_DIR.parent),
        )
        print(f"[SAVED] {downloaded}")

    print("OCR model weights ready.\n")


def download_llm_model(
    repo_id: str = DEFAULT_LLM_REPO,
    filename: str = DEFAULT_LLM_FILENAME,
    target_dir: Path = MODELS_DIR,
):
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / filename
    print(f"\n--- Checking/downloading Local LLM: {repo_id}/{filename} ---")
    if target_file.exists() and target_file.stat().st_size > 0:
        print(f"[CACHED] {target_file} already exists ({target_file.stat().st_size:,} bytes). Skipping download.")
        return str(target_file)

    print(f"[DOWNLOADING] {filename} from {repo_id} (~2.5 GB) ...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_dir),
    )
    print(f"[SAVED] Local LLM model saved to: {downloaded_path}\n")
    return downloaded_path


def main():
    parser = argparse.ArgumentParser(description="Download OCR and local LLM models.")
    parser.add_argument("--ocr", action="store_true", help="Download OCR models")
    parser.add_argument("--llm", action="store_true", help="Download local LLM (MedGemma GGUF)")
    parser.add_argument("--all", action="store_true", help="Download both OCR and LLM models")
    args = parser.parse_args()

    # Default if no arguments specified: download both LLM and OCR
    if not args.ocr and not args.llm and not args.all:
        download_llm_model()
        download_ocr_models()
    else:
        if args.llm or args.all:
            download_llm_model()
        if args.ocr or args.all:
            download_ocr_models()


if __name__ == "__main__":
    main()
