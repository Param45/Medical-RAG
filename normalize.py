"""
Medical Records RAG Demo — Terminology Normalization (SRS §5.4)

Normalizes medical abbreviations, Devanagari/Hindi terms, and extracts
structured clinical entities (diagnoses, drugs, procedures, biomarkers, staging)
using exact/fuzzy dictionary matching, structured regex parsers, and LLM fallback.
Maps to BUILD_GUIDE Task 1.6.
"""

from dataclasses import asdict, dataclass
import difflib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class NormalizedEntity:
    """
    Normalized clinical entity representation (SRS §5.4.3).
    """
    raw_text: str
    normalized_term: str
    entity_type: str  # "Procedure" | "Regimen" | "Medication" | "Diagnosis" | "Modifier" | "Biomarker" | "LabTest" | "Finding" | "Staging" | "DocumentSection" | "Organization"
    method: str       # "dictionary" | "regex" | "llm" | "none"
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.metadata is None:
            data.pop("metadata", None)
        return data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormalizedEntity":
        return cls(
            raw_text=str(d.get("raw_text", "")),
            normalized_term=str(d.get("normalized_term", "")),
            entity_type=str(d.get("entity_type", "Finding")),
            method=str(d.get("method", "none")),
            confidence=float(d.get("confidence", 1.0)),
            metadata=d.get("metadata"),
        )


# Global dictionary caches
_ONCOLOGY_TERMS: Optional[Dict[str, Dict[str, str]]] = None
_HINDI_TERMS: Optional[Dict[str, Dict[str, str]]] = None


def load_normalization_dictionaries(
    dict_dir: Optional[Path] = None
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Load oncology and Hindi abbreviation dictionaries from JSON files.
    """
    global _ONCOLOGY_TERMS, _HINDI_TERMS

    if _ONCOLOGY_TERMS is not None and _HINDI_TERMS is not None and dict_dir is None:
        return _ONCOLOGY_TERMS, _HINDI_TERMS

    if dict_dir is None:
        dict_dir = Path(__file__).parent / "data" / "normalization"
    else:
        dict_dir = Path(dict_dir)

    oncology_file = dict_dir / "oncology_terms.json"
    hindi_file = dict_dir / "hindi_terms.json"

    oncology_dict: Dict[str, Dict[str, str]] = {}
    if oncology_file.exists():
        try:
            data = json.loads(oncology_file.read_text(encoding="utf-8"))
            oncology_dict = {k: v for k, v in data.items() if not k.startswith("_")}
        except Exception as e:
            print(f"Warning: Failed to load {oncology_file}: {e}")

    hindi_dict: Dict[str, Dict[str, str]] = {}
    if hindi_file.exists():
        try:
            data = json.loads(hindi_file.read_text(encoding="utf-8"))
            hindi_dict = {k: v for k, v in data.items() if not k.startswith("_")}
        except Exception as e:
            print(f"Warning: Failed to load {hindi_file}: {e}")

    _ONCOLOGY_TERMS = oncology_dict
    _HINDI_TERMS = hindi_dict
    return _ONCOLOGY_TERMS, _HINDI_TERMS


def dictionary_match(
    term: str,
    min_similarity: float = 0.85,
    base_confidence: float = 1.0,
    dict_dir: Optional[Path] = None,
) -> Optional[NormalizedEntity]:
    """
    Match a term/phrase against oncology and Hindi dictionaries (SRS §5.4.2 step 1).
    Supports exact case-insensitive lookup and fuzzy matching.
    """
    if not term or not term.strip():
        return None

    cleaned_term = term.strip()
    term_upper = cleaned_term.upper()
    oncology_dict, hindi_dict = load_normalization_dictionaries(dict_dir)

    # 1. Exact match (case-insensitive in oncology terms)
    for k, v in oncology_dict.items():
        if k.upper() == term_upper:
            return NormalizedEntity(
                raw_text=cleaned_term,
                normalized_term=v["canonical"],
                entity_type=v["type"],
                method="dictionary",
                confidence=base_confidence,
            )

    # 2. Exact match in Hindi terms (exact unicode match)
    if cleaned_term in hindi_dict:
        v = hindi_dict[cleaned_term]
        return NormalizedEntity(
            raw_text=cleaned_term,
            normalized_term=v["canonical"],
            entity_type=v["type"],
            method="dictionary",
            confidence=base_confidence,
        )

    # 3. Fuzzy matching for multi-character terms (min length 4, not pure numbers, high similarity)
    # 3-letter abbreviations (e.g. LFT, RFT, CBC) must match exactly to avoid false positives with words like 'left'.
    if len(cleaned_term) >= 4 and not cleaned_term.isdigit():
        keys_upper = {k.upper(): k for k in oncology_dict.keys() if len(k) >= 4}
        matches = difflib.get_close_matches(term_upper, keys_upper.keys(), n=1, cutoff=max(min_similarity, 0.88))
        if matches:
            matched_key = keys_upper[matches[0]]
            v = oncology_dict[matched_key]
            # Adjust confidence by similarity score
            ratio = difflib.SequenceMatcher(None, term_upper, matches[0]).ratio()
            adjusted_conf = round(base_confidence * ratio, 2)
            return NormalizedEntity(
                raw_text=cleaned_term,
                normalized_term=v["canonical"],
                entity_type=v["type"],
                method="dictionary",
                confidence=adjusted_conf,
                metadata={"matched_key": matched_key, "similarity": ratio},
            )

    return None


def regex_parsers(text: str, base_confidence: float = 1.0) -> List[NormalizedEntity]:
    """
    Extract structured entities via regular expressions (SRS §5.4.2 step 2).
    
    Parses:
    1. TNM Staging: e.g. T2N0M0, rT2N1M0, pT3N1aM0, T4bN2M1
    2. Hormone Receptor & Biomarker Scores: ER, PR, HER2, Ki67
    3. Nottingham Histological Score / Bloom-Richardson Grade
    """
    if not text:
        return []

    entities: List[NormalizedEntity] = []

    # -------------------------------------------------------------
    # 1. TNM Staging Regex
    # Matches prefix (p|y|r)? followed by T... N... M...
    # Examples: T3N0M0, rT2N1M0, pT3N1aM0, T2N0Mx, T4N2M0, T3NoMo
    # -------------------------------------------------------------
    tnm_pattern = re.compile(
        r"\b((?:[pyrPYR])?\s*T(?:[0-4]|is|[a-dIS])[a-dAD]?\s*(?:[pyrPYR])?\s*N(?:[0-3oO]|[a-cAC]|x|X)[a-cAC]?\s*(?:[pyrPYR])?\s*M(?:[01xXoO]))(?:\b|\s|$|[,;.\)])"
    )
    for match in tnm_pattern.finditer(text):
        raw_match = match.group(1).strip()
        # Clean internal whitespace e.g. 'T3 N0 M0' -> 'T3N0M0' and normalize 'o'/'O' to '0'
        cleaned = re.sub(r"\s+", "", raw_match)
        cleaned = re.sub(r"(?<=[TNMtnm])o", "0", cleaned, flags=re.IGNORECASE)
        entities.append(
            NormalizedEntity(
                raw_text=raw_match,
                normalized_term=f"TNM Staging: {cleaned.upper()}",
                entity_type="Staging",
                method="regex",
                confidence=base_confidence,
                metadata={"staging_code": cleaned.upper()},
            )
        )

    # -------------------------------------------------------------
    # 2. Biomarkers: ER, PR, HER2, Ki67
    # -------------------------------------------------------------
    # ER (Estrogen Receptor)
    # Examples: ER 8/8, ER+ (8/8), ER+, ER-, ER positive, ER 0/8, ER: +ve
    er_pattern = re.compile(
        r"\bER\s*[:=\-]?\s*(\+ve|\-ve|positive|negative|[+\-]|\d+\s*/\s*\d+|\d+\+?)",
        re.IGNORECASE,
    )
    for match in er_pattern.finditer(text):
        val = match.group(1).strip()
        entities.append(
            NormalizedEntity(
                raw_text=match.group(0).strip(),
                normalized_term=f"Estrogen Receptor: {val}",
                entity_type="Biomarker",
                method="regex",
                confidence=base_confidence,
                metadata={"marker": "ER", "value": val},
            )
        )

    # PR (Progesterone Receptor)
    # Examples: PR 0/8, PR 8/8, PR+, PR-, PR positive, PR negative
    pr_pattern = re.compile(
        r"\bPR\s*[:=\-]?\s*(\+ve|\-ve|positive|negative|[+\-]|\d+\s*/\s*\d+|\d+\+?)",
        re.IGNORECASE,
    )
    for match in pr_pattern.finditer(text):
        val = match.group(1).strip()
        entities.append(
            NormalizedEntity(
                raw_text=match.group(0).strip(),
                normalized_term=f"Progesterone Receptor: {val}",
                entity_type="Biomarker",
                method="regex",
                confidence=base_confidence,
                metadata={"marker": "PR", "value": val},
            )
        )

    # HER2 / Her2neu
    # Examples: HER2 3+, Her2neu 1+, HER-2/neu 3+, HER2+, HER2 negative
    her2_pattern = re.compile(
        r"\b(?:HER-?2(?:/neu)?|Her2neu)\s*[:=\-]?\s*(\+ve|\-ve|positive|negative|[+\-]|\d+\+?)",
        re.IGNORECASE,
    )
    for match in her2_pattern.finditer(text):
        val = match.group(1).strip()
        entities.append(
            NormalizedEntity(
                raw_text=match.group(0).strip(),
                normalized_term=f"HER2: {val}",
                entity_type="Biomarker",
                method="regex",
                confidence=base_confidence,
                metadata={"marker": "HER2", "value": val},
            )
        )

    # Ki-67 Proliferation Index
    # Examples: Ki67 12-14%, Ki-67: 15%, Ki67: 20%
    ki67_pattern = re.compile(
        r"\bKi-?67\s*[:=\-]?\s*(\d+(?:\s*-\s*\d+)?\s*%?)",
        re.IGNORECASE,
    )
    for match in ki67_pattern.finditer(text):
        val = match.group(1).strip()
        entities.append(
            NormalizedEntity(
                raw_text=match.group(0).strip(),
                normalized_term=f"Ki-67: {val}",
                entity_type="Biomarker",
                method="regex",
                confidence=base_confidence,
                metadata={"marker": "Ki-67", "value": val},
            )
        )

    # -------------------------------------------------------------
    # 3. Nottingham Score / Histologic Grade
    # Examples: Total Score - 8, Grade 3 | Grade III | Grade 2
    # -------------------------------------------------------------
    nottingham_pattern = re.compile(
        r"(?:(?:Total\s+Score\s*[-:]?\s*(\d+)[,\s]*)?(?:Grade|Nottingham\s+Grade)\s*[-:]?\s*([1-3]|I{1,3}))\b",
        re.IGNORECASE,
    )
    for match in nottingham_pattern.finditer(text):
        score = match.group(1)
        grade = match.group(2)
        desc = f"Histologic Grade: Grade {grade}" + (f" (Score {score})" if score else "")
        entities.append(
            NormalizedEntity(
                raw_text=match.group(0).strip(),
                normalized_term=desc,
                entity_type="Staging",
                method="regex",
                confidence=base_confidence,
                metadata={"grade": grade, "score": score},
            )
        )

    return entities


def llm_normalize(
    term: str,
    context: str = "",
    base_confidence: float = 1.0,
) -> Optional[NormalizedEntity]:
    """
    LLM fallback for terms that did not match dictionary or regex (SRS §5.4.2 step 3).
    
    Prompts the LLM with the fixed dictionary types/options and instructs it
    to return a JSON object or null if none apply.
    """
    try:
        from llm_client import chat
    except ImportError:
        return None

    # Penalty for LLM method per SRS §5.4.3
    llm_confidence = max(0.0, round(base_confidence - 0.1, 2))

    prompt = (
        f"You are a medical oncology normalization assistant.\\n"
        f"Given the abbreviation/term: '{term}' extracted from context: '{context[:200]}'\\n"
        f"Map it to a canonical medical oncology term and entity type.\\n"
        f"Allowed entity types: Procedure, Regimen, Medication, Diagnosis, Modifier, Biomarker, LabTest, Finding, Staging, DocumentSection, Organization.\\n"
        f"If the term is not a medical oncology term or cannot be confidently mapped, return null.\\n"
        f"Respond ONLY in valid JSON: {{\"canonical\": \"...\", \"type\": \"...\"}} or null."
    )

    try:
        response = chat([{"role": "user", "content": prompt}])
        if not response or response.strip() == "null":
            return None
        data = json.loads(response.strip())
        if isinstance(data, dict) and "canonical" in data and "type" in data:
            return NormalizedEntity(
                raw_text=term,
                normalized_term=data["canonical"],
                entity_type=data["type"],
                method="llm",
                confidence=llm_confidence,
            )
    except (NotImplementedError, Exception):
        # LLM client not configured or errored -> graceful fallback to None
        return None

    return None


def normalize_date(raw: str) -> Optional[str]:
    """
    Normalize raw date string to ISO format YYYY-MM-DD (SRS §5.4.4).
    Reuses date extractor from split_reports.py.
    """
    from split_reports import extract_report_date
    return extract_report_date(raw)


def normalize_chunk_text(
    raw_text: str,
    script: str = "latin",
    base_confidence: float = 1.0,
    use_llm_fallback: bool = False,
    dict_dir: Optional[Path] = None,
) -> List[NormalizedEntity]:
    """
    Tokenize chunk text and extract normalized entities via:
    1. Structured regex parsers (TNM staging, biomarkers, grades)
    2. Dictionary matching (exact & fuzzy) across oncology and Hindi terms
    3. Optional LLM fallback
    
    Returns deduplicated list of NormalizedEntity objects.
    """
    if not raw_text:
        return []

    entities: List[NormalizedEntity] = []
    seen_keys: Set[Tuple[str, str]] = set()

    def add_entity(ent: NormalizedEntity) -> None:
        key = (ent.normalized_term.upper(), ent.entity_type.upper())
        if key not in seen_keys:
            seen_keys.add(key)
            entities.append(ent)

    # 1. Regex parsers
    regex_matches = regex_parsers(raw_text, base_confidence=base_confidence)
    for ent in regex_matches:
        add_entity(ent)

    # 2. Dictionary matching on tokens and n-grams (1, 2, 3 words)
    # Split text into tokens keeping alphanumeric, hyphens, and slashes
    lines = raw_text.splitlines()
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Extract words/tokens
        tokens = re.findall(r"[\w\u0900-\u097F\-+/]+", line_clean)
        num_tokens = len(tokens)

        # Multi-word phrases (3-gram, 2-gram, 1-gram)
        for n in range(3, 0, -1):
            for i in range(num_tokens - n + 1):
                phrase = " ".join(tokens[i : i + n])
                # Skip if already part of a matched staging/biomarker
                if any(phrase.upper() in ent.raw_text.upper() for ent in entities if ent.method == "regex"):
                    continue

                matched = dictionary_match(phrase, base_confidence=base_confidence, dict_dir=dict_dir)
                if matched is not None:
                    add_entity(matched)

    # 3. Optional LLM fallback for unresolved potential medical abbreviations
    if use_llm_fallback:
        candidate_words = re.findall(r"\b[A-Z0-9\-]{2,8}\b", raw_text)
        for w in candidate_words:
            if not any(w.upper() in ent.raw_text.upper() for ent in entities):
                llm_ent = llm_normalize(w, context=raw_text, base_confidence=base_confidence)
                if llm_ent is not None:
                    add_entity(llm_ent)

    return entities

