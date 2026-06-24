"""
Phase 1–2: Tamil Text Acquisition and Unicode Normalisation
TaGI Paper — Eq. (1):  NS_i = N(S_i)

Applies NFC Unicode normalisation, removes zero-width joiners/non-joiners,
and standardises punctuation so downstream morphological tools receive
a consistent byte sequence.
"""

import unicodedata
import re
from pathlib import Path
from typing import List, Dict
import json


# ---------------------------------------------------------------------------
# Unicode normalisation (Eq. 1)
# ---------------------------------------------------------------------------

ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")   # ZWS, ZWNJ, ZWJ, BOM
TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")             # Tamil Unicode block

PUNCT_MAP = {
    "\u201c": '"', "\u201d": '"',   # curly double quotes → straight
    "\u2018": "'", "\u2019": "'",   # curly single quotes → straight
    "\u2014": "-", "\u2013": "-",   # em/en dash → hyphen
    "\u2026": "...",                 # ellipsis → three dots
}


def normalise(sentence: str) -> str:
    """
    N(·) from Eq. (1):
      1. NFC canonical composition
      2. Remove zero-width control characters
      3. Standardise punctuation to ASCII equivalents
      4. Collapse multiple whitespace to single space
    """
    # Step 1 — NFC
    s = unicodedata.normalize("NFC", sentence)
    # Step 2 — zero-width characters
    s = ZERO_WIDTH.sub("", s)
    # Step 3 — punctuation normalisation
    for src, tgt in PUNCT_MAP.items():
        s = s.replace(src, tgt)
    # Step 4 — whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Corpus loader  (D = {S_1, S_2, …, S_n})
# ---------------------------------------------------------------------------

DOMAIN_LABELS = ["news", "literary", "social_media", "educational", "spoken"]


def load_corpus(paths: Dict[str, str]) -> List[Dict]:
    """
    Load raw Tamil sentences from five source domains.

    Args:
        paths: {domain_label: file_path}  — one plain-text file per domain,
               one sentence per line.

    Returns:
        List of dicts: {"sentence": str, "domain": str}
    """
    corpus = []
    for domain, path in paths.items():
        p = Path(path)
        if not p.exists():
            print(f"[WARN] {path} not found — skipping domain '{domain}'")
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    corpus.append({"sentence": line, "domain": domain})
    return corpus


def length_filter(corpus: List[Dict], min_tok: int = 8, max_tok: int = 60) -> List[Dict]:
    """Retain sentences within the prescribed token-count window."""
    return [
        item for item in corpus
        if min_tok <= len(item["sentence"].split()) <= max_tok
    ]


def dedup(corpus: List[Dict]) -> List[Dict]:
    """Remove duplicate sentences (by normalised form)."""
    seen = set()
    out = []
    for item in corpus:
        key = normalise(item["sentence"])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Main preprocessing pipeline for Phase 1-2
# ---------------------------------------------------------------------------

def preprocess_corpus(
    paths: Dict[str, str],
    output_path: str = "data/normalised_corpus.jsonl",
    min_tok: int = 8,
    max_tok: int = 60,
) -> List[Dict]:
    """
    Full Phase 1-2 pipeline:
      load → length-filter → dedup → normalise

    Returns:
        D_N = [{sentence, domain, normalised}, …]
    """
    print("[Phase 1] Loading corpus from domains …")
    corpus = load_corpus(paths)
    print(f"  Raw sentences      : {len(corpus):,}")

    corpus = length_filter(corpus, min_tok, max_tok)
    print(f"  After length filter: {len(corpus):,}")

    corpus = dedup(corpus)
    print(f"  After dedup        : {len(corpus):,}")

    print("[Phase 2] Applying Unicode normalisation (NFC) …")
    normalised = []
    for item in corpus:
        ns = normalise(item["sentence"])
        normalised.append({**item, "normalised": ns})

    # Persist
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in normalised:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved to {output_path}")

    return normalised


# ---------------------------------------------------------------------------
# Demo / unit test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "அவர்கள்\u200c வருகிறான்.",          # ZWNJ inside word
        "\u201cதமிழ்\u201d மொழி அழகானது.",    # curly quotes
        "அவன்    பள்ளிக்கு    போகிறான்.",     # extra spaces
        "படித்துக்கொண்டிருந்தான் மாணவன்.",
    ]
    print("=== Phase 1-2 Demo ===")
    for s in samples:
        ns = normalise(s)
        print(f"  IN : {repr(s)}")
        print(f"  OUT: {repr(ns)}")
        print()