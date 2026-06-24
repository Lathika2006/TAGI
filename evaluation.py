"""
Evaluation Suite for TaGI Framework
Implements all metrics reported in Tables 3–6 of the paper:
  - Token-level detection: Precision, Recall, F1 (binary)
  - Error classification: Per-class and macro F1 (K=8 categories)
  - Correction quality: BLEU-4, ROUGE-L, Cosine similarity
  - Full-sentence accuracy (Sc = 84.7% in paper)
  - Comparative baseline evaluation
"""

from __future__ import annotations
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Detection metrics (Eq. 7 evaluation)
# ---------------------------------------------------------------------------

@dataclass
class DetectionMetrics:
    precision: float
    recall:    float
    f1:        float
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __str__(self) -> str:
        return (f"P={self.precision:.4f}  R={self.recall:.4f}  "
                f"F1={self.f1:.4f}  TP={self.tp}  FP={self.fp}  FN={self.fn}")


def compute_detection_metrics(
    preds:  List[int],    # predicted binary labels (0/1) per token
    labels: List[int],    # gold binary labels
) -> DetectionMetrics:
    """Token-level binary precision, recall, F1 (positive class = ERROR=1)."""
    assert len(preds) == len(labels)
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))

    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    return DetectionMetrics(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn)


# ---------------------------------------------------------------------------
# Classification metrics (Eq. 8 evaluation)
# ---------------------------------------------------------------------------

ERROR_LABEL_NAMES = {
    0: "OK",
    1: "ME  (Morphological)",
    2: "TE  (Tense)",
    3: "SVA (Subj-Verb Agr.)",
    4: "CME (Case Marker)",
    5: "SV  (Sandhi)",
    6: "WOE (Word Order)",
    7: "NAE (Number Agr.)",
    8: "SPE (Spelling)",
}


@dataclass
class ClassificationMetrics:
    per_class: Dict[int, DetectionMetrics]
    macro_precision: float
    macro_recall:    float
    macro_f1:        float

    def __str__(self) -> str:
        lines = ["Per-class metrics (error tokens only):"]
        for k, v in self.per_class.items():
            lines.append(f"  {ERROR_LABEL_NAMES.get(k, str(k)):30s} {v}")
        lines.append(
            f"\nMacro  P={self.macro_precision:.4f}  "
            f"R={self.macro_recall:.4f}  F1={self.macro_f1:.4f}"
        )
        return "\n".join(lines)


def compute_classification_metrics(
    preds:  List[int],    # predicted class ids (0-8)
    labels: List[int],    # gold class ids
    error_classes: List[int] = list(range(1, 9)),
) -> ClassificationMetrics:
    """Macro-averaged precision/recall/F1 over K=8 error classes."""
    per_class: Dict[int, DetectionMetrics] = {}
    for c in error_classes:
        tp = sum(p == c and l == c for p, l in zip(preds, labels))
        fp = sum(p == c and l != c for p, l in zip(preds, labels))
        fn = sum(p != c and l == c for p, l in zip(preds, labels))
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        per_class[c] = DetectionMetrics(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn)

    macro_p  = np.mean([m.precision for m in per_class.values()])
    macro_r  = np.mean([m.recall    for m in per_class.values()])
    macro_f1 = np.mean([m.f1        for m in per_class.values()])
    return ClassificationMetrics(per_class, macro_p, macro_r, macro_f1)


# ---------------------------------------------------------------------------
# BLEU-4
# ---------------------------------------------------------------------------

def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    hypotheses: List[List[str]],
    references: List[List[str]],
    max_n: int = 4,
) -> float:
    """
    Corpus-level BLEU-n with brevity penalty.
    References and hypotheses are lists of token lists.
    """
    clip_counts  = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_ng = _ngram_counts(hyp, n)
            ref_ng = _ngram_counts(ref, n)
            for gram, cnt in hyp_ng.items():
                clip_counts[n-1]  += min(cnt, ref_ng.get(gram, 0))
                total_counts[n-1] += cnt

    log_bleu = 0.0
    for n in range(max_n):
        if total_counts[n] == 0:
            return 0.0
        p_n = clip_counts[n] / total_counts[n]
        log_bleu += math.log(max(p_n, 1e-8)) / max_n

    bp = min(1.0, math.exp(1 - ref_len / max(hyp_len, 1)))
    return bp * math.exp(log_bleu) * 100


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def corpus_rouge_l(
    hypotheses: List[List[str]],
    references: List[List[str]],
) -> float:
    """Corpus-level ROUGE-L F1 averaged over sentence pairs."""
    scores = []
    for hyp, ref in zip(hypotheses, references):
        l   = _lcs_length(hyp, ref)
        prec = l / max(len(hyp), 1)
        rec  = l / max(len(ref),  1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        scores.append(f1)
    return float(np.mean(scores)) * 100


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return num / max(den, 1e-8)


def sentence_cosine_similarity(
    hyp_vecs: List[np.ndarray],
    ref_vecs: List[np.ndarray],
) -> float:
    """Average cosine similarity between hypothesis and reference sentence embeddings."""
    sims = [cosine_similarity(h, r) for h, r in zip(hyp_vecs, ref_vecs)]
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Full-sentence accuracy (Sc)
# ---------------------------------------------------------------------------

def full_sentence_accuracy(
    predicted_sentences: List[List[str]],
    reference_sentences: List[List[str]],
) -> float:
    """
    Sc = proportion of test sentences where predicted output exactly
    matches the reference correction.
    Paper reports Sc = 84.7%.
    """
    correct = sum(p == r for p, r in zip(predicted_sentences, reference_sentences))
    return correct / max(len(predicted_sentences), 1)


# ---------------------------------------------------------------------------
# Aggregate evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    det:  DetectionMetrics
    cls:  ClassificationMetrics
    bleu4:        float
    rouge_l:      float
    cosine_sim:   float
    sentence_acc: float

    def print(self):
        print("=" * 60)
        print("TaGI Evaluation Report")
        print("=" * 60)
        print("\n[Error Detection — Phase 7]")
        print(f"  {self.det}")
        print("\n[Error Classification — Phase 8]")
        print(str(self.cls))
        print("\n[Correction Quality — Phase 9a]")
        print(f"  BLEU-4         : {self.bleu4:.1f}")
        print(f"  ROUGE-L        : {self.rouge_l:.1f}%")
        print(f"  Cosine sim     : {self.cosine_sim:.4f}")
        print(f"\n[Sentence-level Accuracy]")
        print(f"  Sc             : {self.sentence_acc:.1%}")
        print("=" * 60)


def evaluate_full(
    det_preds:    List[int],
    det_labels:   List[int],
    cls_preds:    List[int],
    cls_labels:   List[int],
    hyp_sents:    List[List[str]],
    ref_sents:    List[List[str]],
    hyp_vecs:     Optional[List[np.ndarray]] = None,
    ref_vecs:     Optional[List[np.ndarray]] = None,
) -> EvaluationReport:
    det  = compute_detection_metrics(det_preds, det_labels)
    cls  = compute_classification_metrics(cls_preds, cls_labels)
    bleu = corpus_bleu(hyp_sents, ref_sents)
    rl   = corpus_rouge_l(hyp_sents, ref_sents)
    sa   = full_sentence_accuracy(hyp_sents, ref_sents)

    cos = 0.0
    if hyp_vecs and ref_vecs:
        cos = sentence_cosine_similarity(hyp_vecs, ref_vecs)

    return EvaluationReport(
        det=det, cls=cls, bleu4=bleu, rouge_l=rl,
        cosine_sim=cos, sentence_acc=sa,
    )


# ---------------------------------------------------------------------------
# Baseline comparisons (Table 3 in paper)
# ---------------------------------------------------------------------------

def simulate_paper_results() -> Dict[str, Dict[str, float]]:
    """
    Reproduce Table 3 comparative results from the paper for reference.
    These are the reported numbers, not computed from a live model.
    """
    return {
        "Rule-Based System": {"P": 96.2, "R": 67.8, "F1": 79.6, "BLEU4": 51.4, "ROUGE-L": 62.3, "Sc": 61.2},
        "CNN-GEC":           {"P": 84.3, "R": 81.7, "F1": 83.0, "BLEU4": 58.7, "ROUGE-L": 67.8, "Sc": 69.4},
        "BiLSTM-CRF":        {"P": 86.1, "R": 83.4, "F1": 84.7, "BLEU4": 61.2, "ROUGE-L": 70.4, "Sc": 71.3},
        "mBERT-GEC":         {"P": 89.3, "R": 85.7, "F1": 87.4, "BLEU4": 66.8, "ROUGE-L": 74.1, "Sc": 76.8},
        "TaGI (proposed)":   {"P": 93.2, "R": 91.8, "F1": 92.5, "BLEU4": 72.3, "ROUGE-L": 79.6, "Sc": 84.7},
    }

import json

def load_jsonl_dataset(file_path):
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            attention_mask = [1] * len(row["input_ids"])

            examples.append(
                GECExample(
                    tokens=row["tokens"],
                    input_ids=row["input_ids"],
                    attention_mask=attention_mask,
                    binary_labels=row["binary_labels"],
                    error_labels=row["error_labels"]
                )
            )

    return examples
# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    import json
    import torch
    from torch.utils.data import DataLoader

    from phase5_6_embedding_attention import TaGIConfig
    from phase7_8_detection_classification import (
      TaGIClassificationModel,
      TamilGECDataset,
      evaluate_model,
      GECExample
    )

    print("=== TaGI Real Model Evaluation ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = TaGIConfig()

    import os

    # Load trained Phase 9b model
    model = TaGIClassificationModel(
        config,
        offline=True
    ).to(device)

    model_path = os.path.join(os.path.dirname(__file__), "data", "tagi_phase7_8_model.pt")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"[Info] Checkpoint not found at {model_path}; using randomly initialised weights.")

    model.eval()

    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "data", "train.jsonl")
    examples = load_jsonl_dataset(data_path)

    split_idx = int(0.8 * len(examples))

    val_examples = examples[split_idx:]

    val_set = TamilGECDataset(
        val_examples,
        max_length=config.max_seq_len
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size
    )

    metrics = evaluate_model(
        model,
        val_loader,
        device
    )

    print("\n===== Evaluation Results =====")

    print(
        f"Detection Precision : {metrics['det_precision']:.4f}"
    )

    print(
        f"Detection Recall    : {metrics['det_recall']:.4f}"
    )

    print(
        f"Detection F1        : {metrics['det_f1']:.4f}"
    )

    print(
        f"Classification Precision : {metrics['cls_precision']:.4f}"
    )

    print(
        f"Classification Recall    : {metrics['cls_recall']:.4f}"
    )

    print(
        f"Classification F1        : {metrics['cls_f1']:.4f}"
    )