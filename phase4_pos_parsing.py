"""
Phase 4: POS Tagging and Dependency Parsing
TaGI Paper — Eq. (3):  T* = argmax_T P(T | S)

Implements:
  - A feature-rich CRF-based POS tagger with morphological features.
  - A lightweight arc-eager dependency parser using morphological features.
  - Parse graph G = (V, E) construction.
  - Agreement-error detection heuristics using the parse graph.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from phase3_tokenisation import MorphemeSegment, rule_based_segment


# ---------------------------------------------------------------------------
# Tamil POS tag set (simplified UD-Tamil compatible)
# ---------------------------------------------------------------------------

POS_TAGS = {
    "NOUN":  "Noun",
    "VERB":  "Finite verb",
    "AUX":   "Auxiliary verb",
    "ADJ":   "Adjective",
    "ADV":   "Adverb",
    "PRON":  "Pronoun",
    "PART":  "Particle",
    "CONJ":  "Conjunction",
    "NUM":   "Numeral",
    "PUNCT": "Punctuation",
    "X":     "Foreign / unknown",
}

# Pronoun forms and their number-person mapping
PRONOUNS = {
    "நான்":      ("1", "SG"),
    "நாம்":      ("1", "PL"),
    "நாங்கள்":  ("1", "PL"),
    "நீ":        ("2", "SG"),
    "நீங்கள்":  ("2", "PL"),
    "அவன்":     ("3", "SG_MASC"),
    "அவள்":     ("3", "SG_FEM"),
    "அது":       ("3", "SG_NEU"),
    "அவர்":     ("3", "SG_HON"),
    "அவர்கள்":  ("3", "PL"),
    "அவர்களுக்கு": ("3", "PL"),
}

# Verb suffix → (person, number) mapping (simplified)
VERB_PERSON_MAP = {
    "கிறேன்":    ("1", "SG"),
    "கிறோம்":    ("1", "PL"),
    "கிறாய்":    ("2", "SG"),
    "கிறீர்கள்": ("2", "PL"),
    "கிறான்":    ("3", "SG_MASC"),
    "கிறாள்":    ("3", "SG_FEM"),
    "கிறது":     ("3", "SG_NEU"),
    "கிறார்":    ("3", "SG_HON"),
    "கிறார்கள்": ("3", "PL"),
    "கின்றான்":  ("3", "SG_MASC"),
    "ந்தான்":    ("3", "SG_MASC"),
    "ந்தாள்":    ("3", "SG_FEM"),
    "ந்தது":     ("3", "SG_NEU"),
    "ந்தேன்":    ("1", "SG"),
    "ந்தார்கள்": ("3", "PL"),
    "வான்":      ("3", "SG_MASC"),
    "வாள்":      ("3", "SG_FEM"),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Token:
    index:    int
    text:     str
    morphemes: List[MorphemeSegment]
    pos:      str = "X"
    person:   str = ""    # 1 / 2 / 3
    number:   str = ""    # SG / PL / SG_MASC / SG_FEM / SG_NEU / SG_HON


@dataclass
class DependencyArc:
    head:  int    # index of head token
    dep:   int    # index of dependent token
    label: str    # nsubj, obj, obl, amod, advmod, aux, root, …


@dataclass
class ParseGraph:
    tokens: List[Token]
    arcs:   List[DependencyArc]

    def head_of(self, dep_idx: int) -> Optional[Token]:
        for arc in self.arcs:
            if arc.dep == dep_idx:
                return self.tokens[arc.head]
        return None

    def dependents(self, head_idx: int, label: Optional[str] = None) -> List[Token]:
        return [
            self.tokens[arc.dep]
            for arc in self.arcs
            if arc.head == head_idx and (label is None or arc.label == label)
        ]


# ---------------------------------------------------------------------------
# Feature extraction for POS tagger
# ---------------------------------------------------------------------------

def extract_pos_features(words: List[str], i: int) -> Dict[str, bool]:
    """
    Morphological feature dictionary for CRF POS tagger.
    In a production system these feed a sklearn-crfsuite model.
    Here we use them directly in the rule-based fallback tagger.
    """
    w = words[i]
    segs = rule_based_segment(w)
    roles = {s.role for s in segs}
    last_seg = segs[-1].text if segs else w

    feats: Dict[str, bool] = {
        "has_tense_suffix": "tense_person" in roles,
        "has_case_suffix":  "case" in roles,
        "has_plural":       "plural" in roles,
        "has_participle":   "participle" in roles,
        "has_aux":          "aux" in roles,
        "is_pronoun":       w in PRONOUNS,
        "is_punct":         bool(re.fullmatch(r"[.,!?;:\"'()\[\]{}]", w)),
        "is_foreign":       not bool(re.search(r"[\u0B80-\u0BFF]", w)),
        "ends_with_aan":    w.endswith("ான்") or w.endswith("கிறான்"),
        "ends_with_aal":    w.endswith("ாள்"),
        "ends_with_adu":    w.endswith("டு") or w.endswith("து"),
        "ends_with_kku":    w.endswith("க்கு"),
        "ends_with_il":     w.endswith("இல்") or w.endswith("ில்"),
        "ends_with_ai":     w.endswith("ஐ"),
        "ends_with_kal":    w.endswith("கள்"),
    }
    return feats


# ---------------------------------------------------------------------------
# Rule-based POS tagger  (Eq. 3 approximation — CRF fallback)
# ---------------------------------------------------------------------------

def pos_tag(words: List[str], morpheme_lists: List[List[MorphemeSegment]]) -> List[Token]:
    """
    Assign POS tags and morphological features (person, number) to each word.
    In production this would use a trained CRF (sklearn-crfsuite); here we
    implement the same feature logic with deterministic rules to be dependency-free.
    """
    tokens: List[Token] = []

    for i, (word, segs) in enumerate(zip(words, morpheme_lists)):
        feats = extract_pos_features(words, i)
        token = Token(index=i, text=word, morphemes=segs)

        if feats["is_punct"]:
            token.pos = "PUNCT"
        elif feats["is_foreign"]:
            token.pos = "X"
        elif word in PRONOUNS:
            token.pos = "PRON"
            token.person, token.number = PRONOUNS[word]
        elif feats["has_tense_suffix"]:
            token.pos = "VERB"
            # Infer person/number from verb suffix
            for suffix, (person, number) in VERB_PERSON_MAP.items():
                if word.endswith(suffix):
                    token.person = person
                    token.number = number
                    break
        elif feats["has_aux"]:
            token.pos = "AUX"
        elif feats["has_case_suffix"]:
            token.pos = "NOUN"
        elif feats["has_plural"]:
            token.pos = "NOUN"
            token.number = "PL"
        elif feats["has_participle"]:
            token.pos = "VERB"   # non-finite
        elif feats["ends_with_adu"]:
            token.pos = "NOUN"
            token.number = "SG"
        else:
            # Default heuristic: Tamil nouns often end in consonants or ம்/ன்/ல்
            token.pos = "NOUN"

        tokens.append(token)

    return tokens


# ---------------------------------------------------------------------------
# Lightweight dependency parser
# ---------------------------------------------------------------------------

def dependency_parse(tokens: List[Token]) -> ParseGraph:
    """
    Build parse graph G = (V, E) with arc labels.
    Uses a simplified arc-eager algorithm based on POS and position heuristics.
    Tamil is strictly verb-final; the main verb is usually the last VERB token.
    """
    arcs: List[DependencyArc] = []
    n = len(tokens)

    # Find the root (last finite verb or last token)
    root_idx = n - 1
    for i in range(n - 1, -1, -1):
        if tokens[i].pos == "VERB" and tokens[i].person:
            root_idx = i
            break

    arcs.append(DependencyArc(head=root_idx, dep=root_idx, label="root"))

    for i, tok in enumerate(tokens):
        if i == root_idx:
            continue
        if tok.pos == "PUNCT":
            arcs.append(DependencyArc(head=root_idx, dep=i, label="punct"))
        elif tok.pos == "PRON":
            arcs.append(DependencyArc(head=root_idx, dep=i, label="nsubj"))
        elif tok.pos == "NOUN":
            # Case suffix determines the dependency label
            case_label = "obj"
            for seg in tok.morphemes:
                if seg.role == "case":
                    if seg.text in ("க்கு",):
                        case_label = "obl:dative"
                    elif seg.text in ("இல்", "ில்"):
                        case_label = "obl:locative"
                    elif seg.text in ("ஐ",):
                        case_label = "obj"
                    elif seg.text in ("ஆல்",):
                        case_label = "obl:instrument"
            # Noun before root → subject/object; after root → unlikely in Tamil
            if i < root_idx:
                arcs.append(DependencyArc(head=root_idx, dep=i, label=case_label))
            else:
                arcs.append(DependencyArc(head=root_idx, dep=i, label="dep"))
        elif tok.pos in ("ADJ", "ADV"):
            # Attach to following noun/verb
            head = min(i + 1, n - 1)
            arcs.append(DependencyArc(head=head, dep=i, label="amod" if tok.pos == "ADJ" else "advmod"))
        elif tok.pos == "AUX":
            arcs.append(DependencyArc(head=root_idx, dep=i, label="aux"))
        else:
            arcs.append(DependencyArc(head=root_idx, dep=i, label="dep"))

    return ParseGraph(tokens=tokens, arcs=arcs)


# ---------------------------------------------------------------------------
# Agreement-error heuristics (used by Phase 6 GED layer)
# ---------------------------------------------------------------------------

def check_subject_verb_agreement(graph: ParseGraph) -> List[Dict]:
    """
    Detect SVA errors: subject's number/person does not match root verb's suffix.
    Returns list of potential error dicts.
    """
    errors = []
    root = None
    for arc in graph.arcs:
        if arc.label == "root":
            root = graph.tokens[arc.head]
            break
    if root is None or root.pos != "VERB":
        return errors

    subjects = graph.dependents(root.index, label="nsubj")
    for subj in subjects:
        if subj.number and root.number and subj.number != root.number:
            errors.append({
                "type": "SVA",
                "subject_idx": subj.index,
                "subject_text": subj.text,
                "subject_number": subj.number,
                "verb_idx": root.index,
                "verb_text": root.text,
                "verb_number": root.number,
                "description": (
                    f"Subject '{subj.text}' ({subj.number}) disagrees with "
                    f"verb '{root.text}' ({root.number})"
                ),
            })
    return errors


def check_case_marker(graph: ParseGraph) -> List[Dict]:
    """
    Detect CME errors: dative case in a locative context or vice-versa.
    Heuristic: verbs of study/existence (படி, இரு) expect locative, not dative.
    """
    LOCATIVE_VERBS = {"படி", "இரு", "வாழ்", "தங்கு"}
    errors = []

    for arc in graph.arcs:
        if arc.label == "obl:dative":
            dep = graph.tokens[arc.dep]
            head = graph.tokens[arc.head]
            root_text = head.morphemes[0].text if head.morphemes else head.text
            if root_text in LOCATIVE_VERBS:
                errors.append({
                    "type": "CME",
                    "token_idx": dep.index,
                    "token_text": dep.text,
                    "verb_text": head.text,
                    "description": (
                        f"Verb '{head.text}' expects locative (-இல்), "
                        f"but found dative (-க்கு) on '{dep.text}'"
                    ),
                })
    return errors


# ---------------------------------------------------------------------------
# Full Phase 4 pipeline
# ---------------------------------------------------------------------------

@dataclass
class Phase4Output:
    sentence:  str
    tokens:    List[Token]
    graph:     ParseGraph
    pos_tags:  List[Tuple[str, str]]   # (word, POS)
    sva_errors: List[Dict]
    cme_errors: List[Dict]


def run_phase4(sentence: str, words: List[str], morpheme_lists: List[List[MorphemeSegment]]) -> Phase4Output:
    tokens    = pos_tag(words, morpheme_lists)
    graph     = dependency_parse(tokens)
    sva_errs  = check_subject_verb_agreement(graph)
    cme_errs  = check_case_marker(graph)
    return Phase4Output(
        sentence=sentence,
        tokens=tokens,
        graph=graph,
        pos_tags=[(t.text, t.pos) for t in tokens],
        sva_errors=sva_errs,
        cme_errors=cme_errs,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase3_tokenisation import tokenise

    test_sentences = [
        "அவர்கள் வருகிறான்.",     # SVA error: plural subj + masc.sing verb
        "அவன் பள்ளிக்கு படிக்கிறான்.",   # CME error: dative instead of locative
        "அவள் தினமும் பள்ளிக்கு போகிறாள்.",  # Correct
    ]

    print("=== Phase 4 Demo: POS Tagging & Dependency Parsing (Eq. 3) ===\n")
    for sent in test_sentences:
        tok_sent = tokenise(sent)
        out = run_phase4(sent, tok_sent.words, tok_sent.morphemes)
        print(f"Sentence : {sent}")
        print(f"POS tags : {out.pos_tags}")
        if out.sva_errors:
            for e in out.sva_errors:
                print(f"  [SVA ERROR] {e['description']}")
        if out.cme_errors:
            for e in out.cme_errors:
                print(f"  [CME ERROR] {e['description']}")
        if not out.sva_errors and not out.cme_errors:
            print("  No heuristic errors detected.")
        print()