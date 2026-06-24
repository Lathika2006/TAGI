"""
Phase 7–8: Grammar Error Detection + Multi-Class Error Classification
TaGI Paper — Eq. (7): y_i = Softmax(W_e h_i + b_e)   [binary detection]
             Eq. (8): P(c|x) = exp(z_c) / Σ exp(z_j)  [8-class classification]

Implements:
  - Token-level binary GED classifier (CORRECT / ERROR)
  - Multi-class GEC layer for K=8 Tamil grammar error types
  - Class-weighted loss for imbalanced data (6:1 correct:error ratio)
  - Full TaGI classification model combining backbone + detection + classification
  - Training loop with AdamW + linear warmup
  - Evaluation: precision, recall, F1 per class and macro-averaged
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from phase5_6_embedding_attention import TaGIConfig, TaGIBackbone


# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

# Binary labels (Phase 7)
BINARY_LABELS  = {0: "CORRECT", 1: "ERROR"}
# Error category labels K=8 (Phase 8)
ERROR_LABELS   = {
    0: "OK",    # no error
    1: "ME",    # Morphological suffix
    2: "TE",    # Tense
    3: "SVA",   # Subject-verb agreement
    4: "CME",   # Case marker
    5: "SV",    # Sandhi violation
    6: "WOE",   # Word order
    7: "NAE",   # Number agreement
    8: "SPE",   # Spelling
}
NUM_ERROR_CLASSES = 9   # 0=OK + 8 error types


# ---------------------------------------------------------------------------
# Grammar Error Detection head  (Eq. 7)
# ---------------------------------------------------------------------------

class GrammarErrorDetector(nn.Module):
    """
    Eq. (7): y_i = Softmax(W_e h_i + b_e)
    Binary token classifier: {CORRECT=0, ERROR=1}
    """
    def __init__(self, config: TaGIConfig):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 2),   # 2 output classes
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args: H — (B, T, d) hidden states from MHA encoder
        Returns: logits (B, T, 2)
        """
        return self.classifier(H)   # W_e h_i + b_e before softmax


# ---------------------------------------------------------------------------
# Grammar Error Classification head  (Eq. 8)
# ---------------------------------------------------------------------------

class GrammarErrorClassifier(nn.Module):
    """
    Eq. (8): P(c|x) = exp(z_c) / Σ_j exp(z_j)
             z_i = W_c h_c + b_c
    Multi-class classifier over K=8 error categories (+ OK = 9 total).
    Applied only to tokens flagged as ERROR by the detection head.
    """
    def __init__(self, config: TaGIConfig):
        super().__init__()
        self.K = NUM_ERROR_CLASSES
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, self.K),   # K classes
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args: H — (B, T, d) or (N_error, d) subset of error tokens
        Returns: logits (B, T, K)
        """
        return self.classifier(H)


# ---------------------------------------------------------------------------
# Full classification model
# ---------------------------------------------------------------------------

class TaGIClassificationModel(nn.Module):
    """
    Phases 5–8 combined:
      IndicBERT → MHA → GED head → GEC head
    """
    def __init__(self, config: TaGIConfig, offline: bool = False):
        super().__init__()
        self.config   = config
        self.backbone = TaGIBackbone(config, offline=offline)
        self.detector = GrammarErrorDetector(config)
        self.classifier = GrammarErrorClassifier(config)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          'detection_logits'     : (B, T, 2)   binary error/correct
          'classification_logits': (B, T, K)   error type
          'hidden_states'        : (B, T, d)
        """
        H = self.backbone.encode(input_ids, attention_mask, token_type_ids)
        det_logits  = self.detector(H)      # Eq. (7)
        cls_logits  = self.classifier(H)    # Eq. (8) — applied to all tokens; loss masked
        return {
            "detection_logits":      det_logits,
            "classification_logits": cls_logits,
            "hidden_states":         H,
        }

    def predict(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        det_threshold:  float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: apply threshold τ=0.5 to detection probabilities.
        Returns predicted binary labels and error-type labels per token.
        """
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask)
        det_probs  = F.softmax(out["detection_logits"], dim=-1)      # (B, T, 2)
        error_prob = det_probs[..., 1]                                 # P(ERROR)
        det_preds  = (error_prob >= det_threshold).long()             # 0/1

        cls_probs  = F.softmax(out["classification_logits"], dim=-1)
        cls_preds  = cls_probs.argmax(dim=-1)                         # (B, T)

        # Only assign error class where detection predicts ERROR
        cls_preds = cls_preds * det_preds

        return {
            "detection_predictions":      det_preds,
            "classification_predictions": cls_preds,
            "detection_probabilities":    error_prob,
            "classification_probabilities": cls_probs,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class GECExample:
    """Single annotated Tamil sentence for GEC training."""
    tokens:         List[str]
    input_ids:      List[int]
    attention_mask: List[int]
    binary_labels:  List[int]    # 0=CORRECT, 1=ERROR per token
    error_labels:   List[int]    # 0=OK, 1-8=error type per token


class TamilGECDataset(Dataset):
    """
    PyTorch Dataset for Tamil GEC.
    Each item provides tensors for one sentence.
    """
    def __init__(
        self,
        examples:   List[GECExample],
        max_length: int = 128,
        pad_id:     int = 0,
    ):
        self.examples   = examples
        self.max_length = max_length
        self.pad_id     = pad_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        L  = min(len(ex.input_ids), self.max_length)

        input_ids      = ex.input_ids[:L]      + [self.pad_id] * (self.max_length - L)
        attention_mask = ex.attention_mask[:L]  + [0]          * (self.max_length - L)
        binary_labels  = ex.binary_labels[:L]  + [-100]        * (self.max_length - L)
        error_labels   = ex.error_labels[:L]   + [-100]        * (self.max_length - L)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "binary_labels":  torch.tensor(binary_labels,  dtype=torch.long),
            "error_labels":   torch.tensor(error_labels,   dtype=torch.long),
        }


def create_dummy_dataset(n: int = 200, seq_len: int = 20, vocab_size: int = 5000) -> List[GECExample]:
    """Generate synthetic examples for demo/testing."""
    import random
    examples = []
    for _ in range(n):
        L      = random.randint(8, seq_len)
        ids    = [random.randint(1, vocab_size - 1) for _ in range(L)]
        mask   = [1] * L
        b_labs = [random.choices([0, 1], weights=[0.85, 0.15])[0] for _ in range(L)]
        e_labs = [random.randint(1, 8) if b == 1 else 0 for b in b_labs]
        examples.append(GECExample(
            tokens=[f"tok_{i}" for i in range(L)],
            input_ids=ids,
            attention_mask=mask,
            binary_labels=b_labs,
            error_labels=e_labs,
        ))
    return examples


# ---------------------------------------------------------------------------
# Loss function with class weights
# ---------------------------------------------------------------------------

class TaGILoss(nn.Module):
    """
    Combined detection + classification loss.
    Detection: binary cross-entropy with 6:1 class weight (correct:error)
    Classification: cross-entropy, computed only on ERROR tokens
    """
    def __init__(self, error_weight: float = 6.0):
        super().__init__()
        # Weight errors more heavily due to 6:1 imbalance
        det_weights = torch.tensor(
          [1.0, error_weight],
          device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
          )
        )
        self.det_loss = nn.CrossEntropyLoss(weight=det_weights, ignore_index=-100)
        self.cls_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        det_logits:  torch.Tensor,    # (B, T, 2)
        cls_logits:  torch.Tensor,    # (B, T, K)
        binary_labels: torch.Tensor, # (B, T)
        error_labels:  torch.Tensor, # (B, T)
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = det_logits.shape

        # Detection loss — Eq. (7)
        det_loss = self.det_loss(
            det_logits.view(B * T, 2),
            binary_labels.view(B * T),
        )

        # Classification loss — Eq. (8), only on ERROR positions
        # Mask non-error positions to ignore_index=-100
        masked_error_labels = error_labels.clone()
        masked_error_labels[binary_labels != 1] = -100

        cls_loss = self.cls_loss(
            cls_logits.view(B * T, NUM_ERROR_CLASSES),
            masked_error_labels.view(B * T),
        )

        total = det_loss + cls_loss
        return {"total": total, "detection": det_loss, "classification": cls_loss}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    det_preds:   np.ndarray,    # (N,) binary
    det_labels:  np.ndarray,
    cls_preds:   np.ndarray,    # (N,) 0-8
    cls_labels:  np.ndarray,
    mask:        np.ndarray,    # valid (non-pad) token mask
) -> Dict[str, float]:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    # Filter to valid (non-padding) positions
    valid = mask.astype(bool)
    dp, dl = det_preds[valid], det_labels[valid]
    cp, cl = cls_preds[valid],  cls_labels[valid]

    # Detection metrics (binary, positive class = ERROR=1)
    det_p, det_r, det_f1, _ = precision_recall_fscore_support(
        dl, dp, average="binary", pos_label=1, zero_division=0
    )

    # Classification metrics (macro over 8 error classes, excluding OK=0)
    err_mask = (cl > 0)
    if err_mask.sum() > 0:
        cls_p, cls_r, cls_f1, _ = precision_recall_fscore_support(
            cl[err_mask], cp[err_mask],
            average="macro", zero_division=0
        )
    else:
        cls_p = cls_r = cls_f1 = 0.0

    return {
        "det_precision": det_p,
        "det_recall":    det_r,
        "det_f1":        det_f1,
        "cls_precision": cls_p,
        "cls_recall":    cls_r,
        "cls_f1":        cls_f1,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_classification_model(
    model:       TaGIClassificationModel,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    config:      TaGIConfig,
    device:      torch.device,
    epochs:      int = None,
) -> List[Dict[str, float]]:
    """
    Training loop for Phases 7–8.
    Uses AdamW with linear warmup (as in the paper).
    """
    from transformers import get_linear_schedule_with_warmup

    epochs = epochs or config.max_epochs
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps   = epochs * len(train_loader)
    warmup_steps  = int(total_steps * config.warmup_ratio)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion     = TaGILoss(error_weight=6.0)

    history = []
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(batch["input_ids"], batch["attention_mask"])
            losses = criterion(
                out["detection_logits"],
                out["classification_logits"],
                batch["binary_labels"],
                batch["error_labels"],
            )
            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        avg_loss    = np.mean(epoch_losses)
        history.append({"epoch": epoch + 1, "loss": avg_loss, **val_metrics})

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Det F1: {val_metrics['det_f1']:.4f} | "
            f"Cls F1: {val_metrics['cls_f1']:.4f}"
        )

    return history


def evaluate_model(
    model: TaGIClassificationModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_det_preds, all_det_labels = [], []
    all_cls_preds, all_cls_labels = [], []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model.predict(batch["input_ids"], batch["attention_mask"])

            mask = (batch["binary_labels"] != -100).cpu().numpy()
            bl   = batch["binary_labels"].cpu().numpy()
            el   = batch["error_labels"].cpu().numpy()
            dp   = preds["detection_predictions"].cpu().numpy()
            cp   = preds["classification_predictions"].cpu().numpy()

            all_det_preds.append(dp.flatten())
            all_det_labels.append(bl.flatten())
            all_cls_preds.append(cp.flatten())
            all_cls_labels.append(el.flatten())
            all_masks.append(mask.flatten())

    det_preds  = np.concatenate(all_det_preds)
    det_labels = np.concatenate(all_det_labels)
    cls_preds  = np.concatenate(all_cls_preds)
    cls_labels = np.concatenate(all_cls_labels)
    masks      = np.concatenate(all_masks)

    return compute_metrics(det_preds, det_labels, cls_preds, cls_labels, masks)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
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
if __name__ == "__main__":

    print("=== Phase 7–8 Tamil Grammar Error Detection ===")

    config = TaGIConfig(max_epochs=10, batch_size=8)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TaGIClassificationModel(
        config,
        offline=True
    )

    import os
    data_path = os.path.join(os.path.dirname(__file__), "data", "train.jsonl")
    examples = load_jsonl_dataset(data_path)

    split_idx = int(0.8 * len(examples))

    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    train_set = TamilGECDataset(
        train_examples,
        max_length=config.max_seq_len
    )

    val_set = TamilGECDataset(
        val_examples,
        max_length=config.max_seq_len
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size
    )

    print(f"Total Samples: {len(examples)}")
    print(f"Train Samples: {len(train_set)}")
    print(f"Validation Samples: {len(val_set)}")

    history = train_classification_model(
        model,
        train_loader,
        val_loader,
        config,
        device,
        epochs=config.max_epochs
    )

    model_out = os.path.join(os.path.dirname(__file__), "data", "tagi_phase7_8_model.pt")
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save(model.state_dict(), model_out)

    print("Model saved successfully!")

    val_set = TamilGECDataset(
      val_examples,
      max_length=config.max_seq_len
    )
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=config.batch_size)

    print(f"Train size: {len(train_set)} | Val size: {len(val_set)}\n")

    history = train_classification_model(model, train_loader, val_loader, config, device, epochs=2)

    print("\nTraining history:")
    for h in history:
        print(f"  Epoch {h['epoch']}: loss={h['loss']:.4f}  det_F1={h['det_f1']:.4f}  cls_F1={h['cls_f1']:.4f}")

    # Single-batch inference demo
    print("\n--- Inference demo ---")
    batch = next(iter(val_loader))
    preds = model.predict(batch["input_ids"].to(device), batch["attention_mask"].to(device))
    det = preds["detection_predictions"][0]
    cls = preds["classification_predictions"][0]
    print(f"  Detection preds (first example): {det.tolist()}")
    print(f"  Error class preds              : {[ERROR_LABELS.get(c.item(), '?') for c in cls]}")