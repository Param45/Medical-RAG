"""
Download required MinerU / PDF-Extract-Kit models for local CPU execution
with bilingual (English + Hindi/Devanagari) OCR support.
Caches models under ./models directory (idempotent, skips if already downloaded).
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path(__file__).parent / "models"

FILES_TO_DOWNLOAD = [
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

def download_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checking/downloading models into: {MODELS_DIR}")
    
    repo_id = "opendatalab/PDF-Extract-Kit-1.0"
    for remote_path, local_rel_path in FILES_TO_DOWNLOAD:
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

    print("\nAll required model weights are ready.")

if __name__ == "__main__":
    download_models()
