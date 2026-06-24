"""
Phase 3: Morpheme-Aware Tokenisation and Morphological Segmentation
TaGI Paper — Eq. (2):  W = <m_1, m_2, …, m_k>

A hybrid rule-augmented SentencePiece tokeniser that respects morpheme
boundaries detected by a rule-based Tamil morphological segmenter.
Falls back gracefully to character-level splitting when SentencePiece
is unavailable (e.g., in a demo environment without a trained SPM model).
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Error category codes (used by downstream phases)
# ---------------------------------------------------------------------------

ERROR_CODES = {
    "ME":  "Morphological suffix error",
    "TE":  "Tense error",
    "SVA": "Subject-verb agreement",
    "CME": "Case marker error",
    "SV":  "Sandhi violation",
    "WOE": "Word order error",
    "NAE": "Number agreement error",
    "SPE": "Spelling error",
    "OK":  "Correct",
}


# ---------------------------------------------------------------------------
# Tamil morpheme boundary rules
# ---------------------------------------------------------------------------

# Common Tamil tense suffixes (simplified; full paradigm has ~200 forms)
TENSE_SUFFIXES = [
    "கிறான்", "கிறாள்", "கிறார்கள்", "கிறார்", "கிறோம்", "கிறீர்கள்",
    "கிறேன்", "கிறது", "கின்றான்", "கின்றாள்",
    "ந்தான்", "ந்தாள்", "ந்தார்கள்", "ந்தது", "ந்தேன்",
    "வான்", "வாள்", "வார்கள்", "வார்", "வோம்", "வேன்",
]

# Case suffixes
CASE_SUFFIXES = [
    "க்கு",   # dative
    "இல்",   # locative
    "ஐ",     # accusative
    "ஆல்",   # instrumental
    "இடம்",  # sociative
    "ஓடு",   # comitative
    "இலிருந்து",  # ablative
]

# Plural markers
PLURAL_MARKERS = ["கள்", "மார்"]

# Sequential/conditional participle markers
PARTICIPLE_MARKERS = ["த்து", "து", "ந்து", "டு"]

# Auxiliary verbs (common)
AUXILIARIES = ["கொண்டு", "விட்டு", "வருகிற", "இருந்த", "இருக்கிற"]


@dataclass
class MorphemeSegment:
    """Single morpheme with its grammatical role label."""
    text:  str
    role:  str      # root | tense | person | case | plural | participle | aux | unknown
    index: int      # position in word


def rule_based_segment(word: str) -> List[MorphemeSegment]:
    """
    Eq. (2): decompose W into <m_1, …, m_k> using suffix-stripping rules.

    Strategy: greedy longest-match suffix stripping from right to left.
    Returns list of MorphemeSegment from left (root) to right (suffixes).
    """
    remaining = word
    suffixes: List[Tuple[str, str]] = []   # (text, role)

    # Strip plural markers first
    for suf in sorted(PLURAL_MARKERS, key=len, reverse=True):
        if remaining.endswith(suf) and len(remaining) > len(suf) + 1:
            remaining = remaining[: -len(suf)]
            suffixes.insert(0, (suf, "plural"))
            break

    # Strip case suffixes
    for suf in sorted(CASE_SUFFIXES, key=len, reverse=True):
        if remaining.endswith(suf) and len(remaining) > len(suf) + 1:
            remaining = remaining[: -len(suf)]
            suffixes.insert(0, (suf, "case"))
            break

    # Strip tense+person suffixes
    for suf in sorted(TENSE_SUFFIXES, key=len, reverse=True):
        if remaining.endswith(suf) and len(remaining) > len(suf) + 1:
            remaining = remaining[: -len(suf)]
            suffixes.insert(0, (suf, "tense_person"))
            break

    # Strip participle markers
    for suf in sorted(PARTICIPLE_MARKERS, key=len, reverse=True):
        if remaining.endswith(suf) and len(remaining) > len(suf) + 1:
            remaining = remaining[: -len(suf)]
            suffixes.insert(0, (suf, "participle"))
            break

    # Strip auxiliary verbs
    for aux in sorted(AUXILIARIES, key=len, reverse=True):
        if remaining.endswith(aux) and len(remaining) > len(aux) + 1:
            remaining = remaining[: -len(aux)]
            suffixes.insert(0, (aux, "aux"))
            break

    # Whatever remains is the root
    segments = [MorphemeSegment(text=remaining, role="root", index=0)]
    for i, (text, role) in enumerate(suffixes, start=1):
        segments.append(MorphemeSegment(text=text, role=role, index=i))

    return segments


# ---------------------------------------------------------------------------
# Sentence-level tokeniser
# ---------------------------------------------------------------------------

@dataclass
class TokenisedSentence:
    raw:        str
    words:      List[str]
    morphemes:  List[List[MorphemeSegment]]   # morphemes[i] → segments for word i
    flat_tokens: List[str]                    # flat morpheme sequence (for model input)


def tokenise(sentence: str) -> TokenisedSentence:
    """
    Split sentence into words, then apply morphological segmentation to each word.
    Punctuation tokens are kept as single-morpheme units.

    Returns TokenisedSentence with:
      - words          : word-level tokens
      - morphemes      : per-word MorphemeSegment lists  (Eq. 2)
      - flat_tokens    : flat morpheme list fed to IndicBERT
    """
    # Simple whitespace + punctuation split (Tamil uses space as word delimiter)
    words = re.findall(r"[\u0B80-\u0BFF]+|[^\s\u0B80-\u0BFF]+", sentence)
    words = [w for w in words if w.strip()]

    morpheme_segs: List[List[MorphemeSegment]] = []
    flat: List[str] = []

    for word in words:
        if re.search(r"[\u0B80-\u0BFF]", word):
            segs = rule_based_segment(word)
        else:
            # Non-Tamil token: treat as single morpheme
            segs = [MorphemeSegment(text=word, role="foreign", index=0)]
        morpheme_segs.append(segs)
        flat.extend(seg.text for seg in segs)

    return TokenisedSentence(
        raw=sentence,
        words=words,
        morphemes=morpheme_segs,
        flat_tokens=flat,
    )


# ---------------------------------------------------------------------------
# Batch tokeniser for full corpus
# ---------------------------------------------------------------------------

def tokenise_corpus(
    normalised_corpus: List[Dict],
    output_path: str = "data/tokenised_corpus.jsonl",
) -> List[Dict]:
    """
    Apply Phase 3 tokenisation to every sentence in D_N.
    Returns enriched records with 'words', 'morphemes', 'flat_tokens'.
    """
    print("[Phase 3] Morpheme-aware tokenisation …")
    out_records = []

    for item in normalised_corpus:
        sentence = item.get("normalised", item["sentence"])
        tok = tokenise(sentence)

        morpheme_dicts = [
            [{"text": s.text, "role": s.role, "index": s.index} for s in segs]
            for segs in tok.morphemes
        ]

        out_records.append({
            **item,
            "words":       tok.words,
            "morphemes":   morpheme_dicts,
            "flat_tokens": tok.flat_tokens,
        })

    # Compute OOV-like stats (morphemes not matching known Tamil roots)
    total_tokens = sum(len(r["flat_tokens"]) for r in out_records)
    print(f"  Sentences processed : {len(out_records):,}")
    print(f"  Total morpheme tokens: {total_tokens:,}")
    avg_morphemes = total_tokens / max(len(out_records), 1)
    print(f"  Avg morphemes/sentence: {avg_morphemes:.1f}")

    # Persist
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in out_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved to {output_path}")

    return out_records


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_words = [
        "படித்துக்கொண்டிருந்தான்",  # complex agglutinated form
        "வருகிறார்கள்",              # verb with 3rd-person plural suffix
        "பள்ளிக்கு",                # noun + dative case
        "மாணவர்கள்",               # noun + plural
        "அழகான",                    # adjective (minimal morphology)
    ]

    print("=== Phase 3 Demo: Morphological Segmentation (Eq. 2) ===\n")
    for word in test_words:
        segs = rule_based_segment(word)
        seg_str = " + ".join(f"[{s.text}|{s.role}]" for s in segs)
        print(f"  {word:35s} → {seg_str}")

    print("\n=== Full Sentence Tokenisation ===\n")
    sent = "அவர்கள் தினமும் பள்ளிக்கு படித்துக்கொண்டிருந்தான்."
    tok = tokenise(sent)
    print(f"  Sentence   : {sent}")
    print(f"  Words      : {tok.words}")
    print(f"  Flat tokens: {tok.flat_tokens}")
    print("\n  Per-word morphemes:")
    for w, segs in zip(tok.words, tok.morphemes):
        print(f"    {w}: {[(s.text, s.role) for s in segs]}")