"""
Phase 9b: Semi-Supervised Learning + Data Augmentation
TaGI Paper — Eq. (11): ỹ = f_θ(x_u)           [pseudo-label generation]
             Eq. (12): ℒ_total = ℒ_supervised + λ · ℒ_unsupervised

Implements:
  - Pseudo-label generation with confidence threshold τ
  - Three augmentation strategies: SR (synonym replacement),
    BT (back-translation via English), MV (morphological variation)
  - Annealed confidence threshold: τ 0.95 → 0.80 over training
  - Semi-supervised training loop combining supervised + consistency loss
"""

from __future__ import annotations
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from phase5_6_embedding_attention import TaGIConfig
from phase7_8_detection_classification import (
    TaGIClassificationModel, TamilGECDataset, GECExample,
    create_dummy_dataset, TaGILoss, evaluate_model,
)


# ---------------------------------------------------------------------------
# Data Augmentation Strategies
# ---------------------------------------------------------------------------

class SynonymReplacement:
    """
    SR: Replace Tamil content words with morphologically equivalent synonyms.
    In production this uses a Tamil WordNet or embedding-based retrieval.
    Here we maintain a small handcrafted synonym dictionary for demo purposes.
    """

    SYNONYM_MAP: Dict[str, List[str]] = {
        "படிக்கிறான்": ["கற்கிறான்", "அறிகிறான்"],
        "வருகிறான்":  ["வருகின்றான்", "வருகிறார்"],
        "பள்ளி":      ["பாடசாலை", "கல்வி நிலையம்"],
        "அழகான":     ["சுந்தரமான", "நேர்த்தியான"],
        "மாணவன்":    ["மாணவர்", "கற்பவன்"],
        "ஆசிரியர்":  ["மாஸ்டர்", "குரு"],
    }

    def augment(self, tokens: List[str], p: float = 0.15) -> List[str]:
        """Replace up to p·|tokens| words with synonyms."""
        out = list(tokens)
        for i, tok in enumerate(out):
            if random.random() < p and tok in self.SYNONYM_MAP:
                out[i] = random.choice(self.SYNONYM_MAP[tok])
        return out


class MorphologicalVariation:
    """
    MV: Introduce controlled morphological mutations.
    Replaces correct suffixes with plausible incorrect ones from the same paradigm,
    creating realistic training errors for under-represented error categories.
    """

    # Tense alternations (correct suffix → plausible wrong suffix)
    TENSE_ALTERNATIONS: List[Tuple[str, str]] = [
        ("கிறான்",    "ந்தான்"),    # present → past
        ("ந்தான்",    "வான்"),       # past → future
        ("கிறாள்",    "ந்தாள்"),
        ("கிறார்கள்", "ந்தார்கள்"),
    ]
    # Case alternations
    CASE_ALTERNATIONS: List[Tuple[str, str]] = [
        ("க்கு", "இல்"),   # dative → locative
        ("இல்",  "க்கு"),  # locative → dative
        ("ஐ",    "க்கு"),  # accusative → dative
    ]

    def _apply_alternation(self, tok: str, alternations: List[Tuple[str, str]]) -> Optional[str]:
        for src, tgt in alternations:
            if tok.endswith(src):
                return tok[:-len(src)] + tgt
        return None

    def augment(self, tokens: List[str], error_rate: float = 0.10) -> Tuple[List[str], List[int], List[int]]:
        """
        Returns (augmented_tokens, binary_labels, error_type_labels).
        binary_labels: 1 for mutated positions, 0 elsewhere
        error_type_labels: error category id for mutated positions
        """
        import re
        out        = list(tokens)
        bin_labels = [0] * len(tokens)
        err_labels = [0] * len(tokens)

        for i, tok in enumerate(out):
            if random.random() >= error_rate:
                continue
            # Try tense mutation first
            mutated = self._apply_alternation(tok, self.TENSE_ALTERNATIONS)
            etype   = 2   # TE
            if mutated is None:
                mutated = self._apply_alternation(tok, self.CASE_ALTERNATIONS)
                etype   = 4   # CME
            if mutated is not None and mutated != tok:
                out[i]        = mutated
                bin_labels[i] = 1
                err_labels[i] = etype

        return out, bin_labels, err_labels


class BackTranslation:
    """
    BT: Tamil → English → Tamil via a pre-trained MT model.
    In a production setting this calls an AI4Bharat IndicTrans2 endpoint.
    Here we simulate the round-trip with a simple word shuffle (as a placeholder)
    that introduces realistic paraphrase variation.
    """

    def augment(self, tokens: List[str]) -> List[str]:
        """
        Simulate back-translation by:
          1. Keeping 90% of tokens
          2. Randomly reordering adjacent tokens (mimics MT reordering)
        A production implementation would call IndicTrans2 MT models.
        """
        out = list(tokens)
        # Simulate MT-induced reordering of adjacent pairs
        i = 0
        while i < len(out) - 1:
            if random.random() < 0.07:
                out[i], out[i+1] = out[i+1], out[i]
                i += 2
            else:
                i += 1
        return out


# ---------------------------------------------------------------------------
# Pseudo-label generation (Eq. 11)
# ---------------------------------------------------------------------------

def generate_pseudo_labels(
    model:      TaGIClassificationModel,
    unlabelled: List[GECExample],
    device:     torch.device,
    confidence_threshold: float = 0.85,
    batch_size: int = 32,
) -> Tuple[List[GECExample], float]:
    """
    Eq. (11): ỹ = f_θ(x_u)

    Generate pseudo-labels for unlabelled examples.
    Retain only those with max class probability ≥ τ.

    Returns:
        (pseudo_labelled_examples, retention_rate)
    """
    model.eval()
    pseudo = []
    retained = 0

    loader = DataLoader(
        TamilGECDataset(unlabelled, max_length=128),
        batch_size=batch_size,
        shuffle=False,
    )

    all_preds: List[Dict] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model.predict(
                batch["input_ids"],
                batch["attention_mask"],
                det_threshold=0.5,
            )
            all_preds.append({
                "det_pred":  preds["detection_predictions"].cpu(),
                "cls_pred":  preds["classification_predictions"].cpu(),
                "det_prob":  preds["detection_probabilities"].cpu(),
                "cls_prob":  preds["classification_probabilities"].cpu(),
            })

    idx = 0
    for pred_batch in all_preds:
        B = pred_batch["det_pred"].shape[0]
        for b in range(B):
            if idx >= len(unlabelled):
                break
            ex = unlabelled[idx]
            T  = len(ex.tokens)

            det_pred = pred_batch["det_pred"][b, :T].tolist()
            cls_pred = pred_batch["cls_pred"][b, :T].tolist()
            det_prob = pred_batch["det_prob"][b, :T]
            cls_prob = pred_batch["cls_prob"][b, :T].max(dim=-1).values

            # Confidence check: token qualifies if its max probability ≥ τ
            confident_mask = []
            for t in range(T):
                p = det_prob[t].item() if det_pred[t] == 1 else (1 - det_prob[t]).item()
                # For classification, also check class confidence
                if det_pred[t] == 1:
                    p = min(p, cls_prob[t].item())
                confident_mask.append(p >= confidence_threshold)

            # Only retain examples where ALL token pseudo-labels are confident
            if all(confident_mask):
                pseudo.append(GECExample(
                    tokens        = ex.tokens,
                    input_ids     = ex.input_ids,
                    attention_mask= ex.attention_mask,
                    binary_labels = det_pred,
                    error_labels  = cls_pred,
                ))
                retained += 1
            idx += 1

    retention_rate = retained / max(len(unlabelled), 1)
    print(f"  Pseudo-labels: {retained}/{len(unlabelled)} retained "
          f"(τ={confidence_threshold:.2f}, retention={retention_rate:.1%})")
    return pseudo, retention_rate


# ---------------------------------------------------------------------------
# Augmented dataset builder
# ---------------------------------------------------------------------------

def build_augmented_dataset(
    labelled:   List[GECExample],
    unlabelled: List[GECExample],
    model:      TaGIClassificationModel,
    device:     torch.device,
    confidence_threshold: float = 0.85,
) -> TamilGECDataset:
    """
    Build augmented training set for semi-supervised step:
      1. Augment labelled examples with SR, BT, MV
      2. Generate pseudo-labels for unlabelled examples (Eq. 11)
      3. Augment pseudo-labelled examples
      4. Combine all
    """
    sr = SynonymReplacement()
    mv = MorphologicalVariation()
    bt = BackTranslation()

    augmented: List[GECExample] = []

    # Augment labelled examples
    for ex in labelled:
        # SR augmentation
        aug_tokens = sr.augment(ex.tokens)
        augmented.append(GECExample(
            tokens=aug_tokens,
            input_ids=ex.input_ids,       # ids unchanged for demo
            attention_mask=ex.attention_mask,
            binary_labels=ex.binary_labels,
            error_labels=ex.error_labels,
        ))

    # Pseudo-label unlabelled examples
    pseudo_labelled, _ = generate_pseudo_labels(
        model, unlabelled, device, confidence_threshold
    )

    # Augment pseudo-labelled with MV and BT
    for ex in pseudo_labelled:
        mv_tokens, mv_blabs, mv_elabs = mv.augment(ex.tokens)
        augmented.append(GECExample(
            tokens=mv_tokens,
            input_ids=ex.input_ids,
            attention_mask=ex.attention_mask,
            binary_labels=mv_blabs if any(mv_blabs) else ex.binary_labels,
            error_labels=mv_elabs  if any(mv_elabs)  else ex.error_labels,
        ))

        bt_tokens = bt.augment(ex.tokens)
        augmented.append(GECExample(
            tokens=bt_tokens,
            input_ids=ex.input_ids,
            attention_mask=ex.attention_mask,
            binary_labels=ex.binary_labels,
            error_labels=ex.error_labels,
        ))

    print(f"  Augmented dataset: {len(labelled)} labelled + {len(augmented)} augmented")
    return TamilGECDataset(labelled + augmented, max_length=128)


# ---------------------------------------------------------------------------
# Semi-Supervised Training Loop  (Eq. 12)
# ---------------------------------------------------------------------------

def train_semi_supervised(
    model:            TaGIClassificationModel,
    labelled_data:    List[GECExample],
    unlabelled_data:  List[GECExample],
    val_data:         List[GECExample],
    config:           TaGIConfig,
    device:           torch.device,
    ssl_epochs:       int = 20,
    initial_tau:      float = 0.95,
    final_tau:        float = 0.80,
    lambda_ssl:       float = 0.5,
) -> List[Dict]:
    """
    Eq. (12): ℒ_total = ℒ_supervised + λ · ℒ_unsupervised

    Semi-supervised training with annealed confidence threshold
    (τ: initial_tau → final_tau over ssl_epochs).
    """
    from transformers import get_linear_schedule_with_warmup

    criterion = TaGILoss(error_weight=6.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    val_loader     = DataLoader(TamilGECDataset(val_data, 128), batch_size=config.batch_size)
    sup_loader     = DataLoader(TamilGECDataset(labelled_data, 128), batch_size=config.batch_size, shuffle=True)
    total_steps    = ssl_epochs * (len(sup_loader) + 1)
    scheduler      = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    history = []

    for epoch in range(ssl_epochs):
        # Anneal τ linearly from initial_tau to final_tau
        tau = initial_tau - (initial_tau - final_tau) * (epoch / max(ssl_epochs - 1, 1))

        # Rebuild augmented dataset with current model and current τ
        aug_dataset = build_augmented_dataset(
            labelled_data, unlabelled_data, model, device, confidence_threshold=tau
        )
        aug_loader = DataLoader(aug_dataset, batch_size=config.batch_size, shuffle=True)

        model.train()
        sup_losses, ssl_losses = [], []

        sup_iter = iter(sup_loader)
        aug_iter = iter(aug_loader)

        steps = max(len(sup_loader), len(aug_loader))
        for _ in range(steps):
            # ── Supervised loss (ℒ_supervised) ──────────────────────────
            try:
                sup_batch = next(sup_iter)
            except StopIteration:
                sup_iter = iter(sup_loader)
                sup_batch = next(sup_iter)
            sup_batch = {k: v.to(device) for k, v in sup_batch.items()}
            sup_out   = model(sup_batch["input_ids"], sup_batch["attention_mask"])
            sup_loss_dict = criterion(
                sup_out["detection_logits"],
                sup_out["classification_logits"],
                sup_batch["binary_labels"],
                sup_batch["error_labels"],
            )
            l_sup = sup_loss_dict["total"]

            # ── Unsupervised loss (ℒ_unsupervised) ───────────────────────
            try:
                aug_batch = next(aug_iter)
            except StopIteration:
                aug_iter = iter(aug_loader)
                aug_batch = next(aug_iter)
            aug_batch = {k: v.to(device) for k, v in aug_batch.items()}
            aug_out   = model(aug_batch["input_ids"], aug_batch["attention_mask"])
            ssl_loss_dict = criterion(
                aug_out["detection_logits"],
                aug_out["classification_logits"],
                aug_batch["binary_labels"],
                aug_batch["error_labels"],
            )
            l_ssl = ssl_loss_dict["total"]

            # Eq. (12): ℒ_total = ℒ_supervised + λ · ℒ_unsupervised
            loss = l_sup + lambda_ssl * l_ssl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            sup_losses.append(l_sup.item())
            ssl_losses.append(l_ssl.item())

        val_metrics = evaluate_model(model, val_loader, device)
        record = {
            "epoch":      epoch + 1,
            "tau":        tau,
            "sup_loss":   np.mean(sup_losses),
            "ssl_loss":   np.mean(ssl_losses),
            **val_metrics,
        }
        history.append(record)
        print(
            f"SSL Epoch {epoch+1:3d}/{ssl_epochs} | "
            f"τ={tau:.3f} | "
            f"SupLoss={record['sup_loss']:.4f} | "
            f"SSLLoss={record['ssl_loss']:.4f} | "
            f"Det F1={val_metrics['det_f1']:.4f} | "
            f"Cls F1={val_metrics['cls_f1']:.4f}"
        )

    return history

import json

def load_jsonl_dataset(file_path):
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            examples.append(
                GECExample(
                    tokens=row["tokens"],
                    input_ids=row["input_ids"],
                    attention_mask=[1] * len(row["input_ids"]),
                    binary_labels=row["binary_labels"],
                    error_labels=row["error_labels"]
                )
            )

    return examples
# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    print("=== Phase 9b Demo: Semi-Supervised Learning (Eq. 11–12) ===")

    config = TaGIConfig(
        batch_size=8,
        learning_rate=2e-4
    )

    device = torch.device("cpu")

    import os

    data_path = os.path.join(os.path.dirname(__file__), "data", "train.jsonl")
    all_examples = load_jsonl_dataset(data_path)

    total = len(all_examples)

    train_end = int(0.8 * total)
    val_end   = int(0.9 * total)

    labelled   = all_examples[:train_end]
    val_data   = all_examples[train_end:val_end]
    unlabelled = all_examples[val_end:]

    print(f"Labelled samples   : {len(labelled)}")
    print(f"Validation samples : {len(val_data)}")
    print(f"Unlabelled samples : {len(unlabelled)}")

    model = TaGIClassificationModel(
        config,
        offline=True
    ).to(device)

    model_path = os.path.join(os.path.dirname(__file__), "data", "tagi_phase7_8_model.pt")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Phase 7-8 model loaded successfully!")
    except FileNotFoundError:
        print(f"[Info] Checkpoint not found at {model_path}; using randomly initialised weights.")

    model.eval()
    history = train_semi_supervised(
      model,
      labelled[:100],
      unlabelled[:20],
      val_data[:20],
      config,
      device,
      ssl_epochs=1,
      initial_tau=0.95,
      final_tau=0.95,
      lambda_ssl=0.5,
    )

    print("\nSSL Training history:")
    for h in history:
        print(
            f"Epoch {h['epoch']} | "
            f"tau={h['tau']:.3f} | "
            f"Det_F1={h['det_f1']:.4f} | "
            f"Cls_F1={h['cls_f1']:.4f}"
        )