"""
Measure triple-extraction quality against a hand-annotated gold standard.

    python eval/run_extraction_eval.py --label baseline
    python eval/run_extraction_eval.py --label qwen7b --model qwen2.5-7b-instruct-q4_k_m.gguf

Writes eval/results/<label>.json and prints a one-line summary. Configuration is
read from the environment so a run is reproducible from its recorded config block.

Metrics, and why each one is here:

  entity_recall      fraction of gold facts found (did it see what is there)
  entity_precision   fraction of predictions that are gold facts
  relation_accuracy  of the facts it found, how often the relation is right
  negated_rate       predictions that assert an explicitly negated/absent fact.
                     This is the headline safety number: "No ascites" becoming a
                     diagnosis of ascites is worse than missing it entirely.
  unsupported_rate   predictions whose name shares no significant token with the
                     source text — invented out of nothing
  json_valid_rate    replies that parsed into at least one usable object

Syntactically valid output is never scored as correct: validity is reported
separately from accuracy, and a run can have json_valid_rate 1.0 with
entity_recall 0.0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv()

# Tokens too generic to prove a name came from the text.
_STOPWORDS = {
    "of", "and", "the", "with", "report", "left", "right", "test", "count",
    "blood", "serum", "total", "cell", "disease", "study", "receptor",
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _tokens(text: str) -> Set[str]:
    return {t for t in _norm(text).split() if len(t) > 2 and t not in _STOPWORDS}


def _names_of(fact: Dict[str, Any]) -> List[str]:
    return [fact["object_name"], *fact.get("aliases", [])]


def _matches(pred_name: str, gold_fact: Dict[str, Any]) -> bool:
    p = _norm(pred_name)
    if not p:
        return False
    for candidate in _names_of(gold_fact):
        c = _norm(candidate)
        if not c:
            continue
        if p == c or p in c or c in p:
            return True
    # Fall back to strong token overlap for reworded names.
    pt, gt = _tokens(pred_name), set()
    for candidate in _names_of(gold_fact):
        gt |= _tokens(candidate)
    return bool(pt) and bool(gt) and len(pt & gt) / len(pt | gt) >= 0.5


def _is_unsupported(pred_name: str, source_text: str) -> bool:
    """True when no significant token of the predicted name occurs in the source."""
    pt = _tokens(pred_name)
    if not pt:
        return True
    src = _tokens(source_text)
    return not (pt & src)


def _prompt_fingerprint() -> str:
    import hashlib
    from graph_backend.build import EXTRACTION_SYSTEM_PROMPT
    return hashlib.sha256(EXTRACTION_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


def _prompt_len() -> int:
    from graph_backend.build import EXTRACTION_SYSTEM_PROMPT
    return len(EXTRACTION_SYSTEM_PROMPT)


def evaluate(gold_path: Path, n_chunks: int, chunks_path: Path) -> Dict[str, Any]:
    from graph_backend.build import extract_triples

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    raw_chunks = raw_chunks if isinstance(raw_chunks, list) else raw_chunks["chunks"]

    rows: List[Dict[str, Any]] = []
    t_start = time.time()

    for spec in gold["chunks"][:n_chunks]:
        idx = spec["index"]
        chunk = raw_chunks[idx]
        ev = chunk.get("evidence_record", {})
        source_text = ev.get("raw_text", "") or chunk.get("text", "")

        meta = {
            # A temp patient id forces the local model path regardless of routing.
            "patient_id": "temp_eval_session",
            "report_id": spec["report_id"],
            "report_type": spec.get("report_type", "CLINICAL_REPORT"),
            "report_date": ev.get("report_date", ""),
            # Drives the OCR-confidence-scaled triple budget.
            "confidence": ev.get("confidence"),
        }

        t0 = time.time()
        error = None
        try:
            preds = extract_triples(
                chunk_text=source_text,
                normalized_entities=chunk.get("normalized_entities", []) or [],
                report_meta=meta,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            preds = []
        seconds = time.time() - t0

        expected: List[Dict[str, Any]] = spec["expected"]
        forbidden: List[Dict[str, Any]] = spec["forbidden"]

        # gold index -> was the relation right on at least one prediction matching it.
        # Counting per prediction let two predictions matching one gold fact push
        # relation accuracy above 1.0.
        matched_gold: Dict[int, bool] = {}
        duplicate_predictions = 0
        negated: List[str] = []
        unsupported: List[str] = []
        spurious: List[str] = []

        for pred in preds:
            name = str(pred.get("object_name") or "")
            relation = str(pred.get("relation") or "")

            hit = next(
                (i for i, g in enumerate(expected) if _matches(name, g)),
                None,
            )
            if hit is not None:
                relation_ok = relation.upper() == expected[hit]["relation"].upper()
                if hit in matched_gold:
                    duplicate_predictions += 1
                    matched_gold[hit] = matched_gold[hit] or relation_ok
                else:
                    matched_gold[hit] = relation_ok
                continue

            if any(_matches(name, {"object_name": f["object_name"]}) for f in forbidden):
                negated.append(name)
            elif _is_unsupported(name, source_text):
                unsupported.append(name)
            else:
                spurious.append(name)

        rows.append({
            "report_id": spec["report_id"],
            "report_type": spec.get("report_type"),
            "predicted": len(preds),
            "gold": len(expected),
            "matched": len(matched_gold),
            "relation_correct": sum(1 for ok in matched_gold.values() if ok),
            "duplicate_predictions": duplicate_predictions,
            "negated": negated,
            "unsupported": unsupported,
            "spurious_but_in_text": spurious,
            "json_parsed": bool(preds) or error is None,
            "error": error,
            "seconds": round(seconds, 1),
            "missed": [
                expected[i]["object_name"]
                for i in range(len(expected)) if i not in matched_gold
            ],
            "over_generation": len(preds) - len(expected),
        })
        print(
            f"CHUNK {spec['report_id']:>12} pred={len(preds):>2} gold={len(expected):>2} "
            f"matched={len(matched_gold):>2} negated={len(negated)} "
            f"unsupported={len(unsupported)} {seconds:.0f}s",
            flush=True,
        )

    gold_total = sum(r["gold"] for r in rows)
    pred_total = sum(r["predicted"] for r in rows)
    matched_total = sum(r["matched"] for r in rows)
    relation_total = sum(r["relation_correct"] for r in rows)
    duplicate_total = sum(r.get("duplicate_predictions", 0) for r in rows)
    negated_total = sum(len(r["negated"]) for r in rows)
    unsupported_total = sum(len(r["unsupported"]) for r in rows)
    parsed = sum(1 for r in rows if r["json_parsed"])
    elapsed = time.time() - t_start

    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 3) if b else None

    recall = ratio(matched_total, gold_total)
    precision = ratio(matched_total, pred_total)
    f1 = (
        round(2 * recall * precision / (recall + precision), 3)
        if recall and precision else 0.0
    )

    return {
        "config": {
            "model": os.getenv("LLM_MODEL"),
            "local_model_path": os.getenv("LOCAL_MODEL_PATH"),
            # Hash the prompt actually used, so two runs can be told apart later even
            # if the prompt text has since changed.
            "prompt_sha12": _prompt_fingerprint(),
            "prompt_chars": _prompt_len(),
            "n_threads": os.getenv("LOCAL_LLM_N_THREADS") or "auto",
            "n_ctx": os.getenv("LOCAL_LLM_N_CTX"),
            "max_tokens": os.getenv("LOCAL_LLM_MAX_TOKENS") or "4096",
            "json_grammar": os.getenv("LOCAL_LLM_JSON_GRAMMAR", "true"),
            "temperature": os.getenv("LOCAL_LLM_TEMPERATURE") or "0.0",
            "few_shot": os.getenv("EXTRACTION_FEW_SHOT", "true"),
        },
        "chunks": len(rows),
        "gold_facts": gold_total,
        "predicted_facts": pred_total,
        "matched": matched_total,
        "entity_recall": recall,
        "entity_precision": precision,
        "entity_f1": f1,
        "relation_accuracy": ratio(relation_total, matched_total),
        "duplicate_predictions": duplicate_total,
        "negated_facts_asserted": negated_total,
        "negated_rate": ratio(negated_total, pred_total),
        "unsupported_facts": unsupported_total,
        "unsupported_rate": ratio(unsupported_total, pred_total),
        "json_valid_rate": ratio(parsed, len(rows)),
        "seconds_total": round(elapsed, 1),
        "seconds_per_chunk": round(elapsed / len(rows), 1) if rows else None,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True, help="name for this run's result file")
    ap.add_argument("--gold", default="eval/gold_patient_a.json")
    ap.add_argument("--chunks", default="data/reports/patient_a_chunks.json")
    ap.add_argument("--n", type=int, default=8, help="how many gold chunks to run")
    ap.add_argument("--model", default=None, help="override LLM_MODEL / LOCAL_MODEL_PATH")
    args = ap.parse_args()

    if args.model:
        os.environ["LLM_MODEL"] = args.model
        os.environ["LOCAL_MODEL_PATH"] = args.model

    summary = evaluate(Path(args.gold), args.n, Path(args.chunks))

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(
        f"RESULT label={args.label} "
        f"recall={summary['entity_recall']} precision={summary['entity_precision']} "
        f"f1={summary['entity_f1']} rel_acc={summary['relation_accuracy']} "
        f"negated={summary['negated_facts_asserted']} "
        f"unsupported={summary['unsupported_facts']} "
        f"json_valid={summary['json_valid_rate']} "
        f"{summary['seconds_per_chunk']}s/chunk -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
