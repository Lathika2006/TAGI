"""
Phase 5–6: IndicBERT Contextual Embedding + Multi-Head Attention Encoder
TaGI Paper — Eq. (4): Attention(Q,K,V) = softmax(QKᵀ/√d_k) V
             Eq. (5): head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
             Eq. (6): MultiHead(Q,K,V) = Concat(head_1,…,head_h)W^O

Implements:
  - IndicBERT fine-tuning wrapper for Tamil GEC
  - Custom Multi-Head Attention analysis layer (h=8 heads)
  - Attention weight extraction for interpretability
  - Head specialisation probing utilities
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TaGIConfig:
    # IndicBERT backbone
    backbone_name:  str   = "ai4bharat/indic-bert"
    hidden_size:    int   = 768
    num_heads:      int   = 8       # h in Eq. (5)
    head_dim:       int   = 96      # d_k = hidden_size / num_heads
    dropout:        float = 0.1
    max_seq_len:    int   = 128
    num_labels:     int   = 2       # binary: CORRECT / ERROR   (Phase 7)
    num_error_types: int  = 8       # K=8 error categories       (Phase 8)
    # Training
    learning_rate:  float = 2e-5
    weight_decay:   float = 0.01
    warmup_ratio:   float = 0.10
    max_epochs:     int   = 30
    batch_size:     int   = 32
    # Semi-supervised (Phase 9b)
    ssl_lambda:     float = 0.5
    ssl_confidence: float = 0.85
    # Correction generator (Phase 9a)
    beam_width:     int   = 5
    max_gen_len:    int   = 64


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention (Eq. 4)
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    """
    Eq. (4): Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,          # (B, h, T, d_k)
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        # Eq. (4) — attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)          # attention weights
        weights = self.dropout(weights)
        output  = torch.matmul(weights, V)            # weighted values
        return output, weights                         # return weights for interpretability


# ---------------------------------------------------------------------------
# Multi-Head Attention Layer  (Eq. 5–6)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Eq. (5): head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    Eq. (6): MultiHead(Q,K,V) = Concat(head_1,…,head_h) W^O
    """
    def __init__(self, config: TaGIConfig):
        super().__init__()
        self.h        = config.num_heads
        self.d_k      = config.head_dim
        self.d_model  = config.hidden_size

        assert self.d_model == self.h * self.d_k, (
            f"hidden_size ({self.d_model}) must equal num_heads ({self.h}) "
            f"× head_dim ({self.d_k})"
        )

        # Per-head projection matrices W_i^Q, W_i^K, W_i^V  (fused for efficiency)
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_O = nn.Linear(self.d_model, self.d_model, bias=False)   # W^O in Eq. (6)

        self.attention = ScaledDotProductAttention(config.dropout)
        self.dropout   = nn.Dropout(config.dropout)
        self.norm      = nn.LayerNorm(self.d_model)

        # Store last attention weights for interpretability
        self.last_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        X: torch.Tensor,                       # (B, T, d_model)  — hidden states H
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = X.shape

        # Linear projections + reshape to (B, h, T, d_k)
        def project_and_split(linear: nn.Linear) -> torch.Tensor:
            return (
                linear(X)
                .view(B, T, self.h, self.d_k)
                .transpose(1, 2)           # → (B, h, T, d_k)
            )

        Q = project_and_split(self.W_Q)
        K = project_and_split(self.W_K)
        V = project_and_split(self.W_V)

        # Eq. (5) — compute each head's attention output
        attn_out, weights = self.attention(Q, K, V, mask)
        self.last_weights = weights.detach()   # (B, h, T, T)

        # Eq. (6) — concatenate heads and project
        attn_out = (
            attn_out
            .transpose(1, 2)               # → (B, T, h, d_k)
            .contiguous()
            .view(B, T, self.d_model)      # → (B, T, d_model)  = Concat(head_1,…,head_h)
        )
        output = self.W_O(attn_out)        # final linear projection W^O
        output = self.dropout(output)
        output = self.norm(output + X)     # residual connection + layer norm
        return output

    def get_head_attention(self, head_idx: int) -> Optional[torch.Tensor]:
        """Return attention weight matrix for a specific head. (B, T, T)"""
        if self.last_weights is None:
            return None
        return self.last_weights[:, head_idx, :, :]


# ---------------------------------------------------------------------------
# IndicBERT-based Tamil GEC Backbone
# ---------------------------------------------------------------------------

class TaGIBackbone(nn.Module):
    """
    IndicBERT encoder + custom MHA layer.
    Wraps the HuggingFace IndicBERT model with a fine-tuning interface.
    Falls back to a randomly-initialised BERT-like config when the
    ai4bharat/indic-bert checkpoint is unavailable (offline / demo mode).
    """

    def __init__(self, config: TaGIConfig, offline: bool = False):
        super().__init__()
        self.config = config

        if not offline:
            try:
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
                self.encoder   = AutoModel.from_pretrained(config.backbone_name)
                print(f"[Phase 5] Loaded IndicBERT from '{config.backbone_name}'")
            except Exception as e:
                print(f"[Phase 5] Could not load '{config.backbone_name}': {e}")
                print("[Phase 5] Falling back to demo BERT encoder …")
                offline = True

        if offline:
            from transformers import BertConfig, BertModel, BertTokenizer
            bert_cfg = BertConfig(
                vocab_size       = 32000,
                hidden_size      = config.hidden_size,
                num_hidden_layers = 4,           # shallow for demo speed
                num_attention_heads = config.num_heads,
                intermediate_size = config.hidden_size * 4,
                max_position_embeddings = config.max_seq_len,
            )
            self.encoder   = BertModel(bert_cfg)
            self.tokenizer = None               # provide raw token ids externally
            print("[Phase 5] Demo mode: using randomly-initialised BERT encoder.")

        # Custom MHA layer on top of IndicBERT (Phase 6)
        self.mha = MultiHeadAttention(config)

    def encode(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through IndicBERT + custom MHA.
        Returns: H ∈ ℝ^{B×T×d} — contextual hidden states
        """
        # IndicBERT encoding — Phase 5
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = enc_out.last_hidden_state    # (B, T, 768)

        # Custom MHA layer — Phase 6  (Eq. 5–6)
        # Use padding mask to prevent attending to [PAD] tokens
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, T)
        H = self.mha(hidden, mask=pad_mask)

        return H


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def batch_encode(
    backbone: TaGIBackbone,
    sentences: List[str],
    device: torch.device,
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Tokenise a batch of Tamil sentences with IndicBERT's tokeniser.
    Returns dict of tensors ready for backbone.encode().
    """
    if backbone.tokenizer is None:
        raise ValueError("Tokenizer not available in offline/demo mode.")

    enc = backbone.tokenizer(
        sentences,
        padding        = True,
        truncation     = True,
        max_length     = max_length,
        return_tensors = "pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


# ---------------------------------------------------------------------------
# Attention head analysis (interpretability)
# ---------------------------------------------------------------------------

def analyse_head_specialisation(
    backbone: TaGIBackbone,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_labels: List[str],       # list of morpheme/token strings for display
) -> Dict[int, float]:
    """
    For each attention head, compute the proportion of attention weight
    concentrated on syntactically relevant pairs (proxy: adjacent tokens
    vs. long-range tokens). Returns dict {head_idx: long_range_ratio}.
    Paper reports 93.2% syntactic attention accuracy for the 8-head config.
    """
    with torch.no_grad():
        backbone.encode(input_ids, attention_mask)

    weights = backbone.mha.last_weights    # (B, h, T, T)
    if weights is None:
        return {}

    T = weights.shape[-1]
    long_range_threshold = 3   # tokens separated by >3 positions = long-range

    head_stats: Dict[int, float] = {}
    for h_idx in range(backbone.config.num_heads):
        w = weights[0, h_idx].cpu()    # (T, T) for first example in batch
        # Long-range attention mass
        long_range_mass = 0.0
        total_mass = 0.0
        for i in range(T):
            for j in range(T):
                if attention_mask[0, j].item() == 0:
                    continue
                mass = w[i, j].item()
                total_mass += mass
                if abs(i - j) > long_range_threshold:
                    long_range_mass += mass
        head_stats[h_idx] = long_range_mass / max(total_mass, 1e-8)

    return head_stats


# ---------------------------------------------------------------------------
# Demo (offline mode — no internet required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    print("=== Phase 5–6 Demo: IndicBERT + Multi-Head Attention (Eq. 4–6) ===\n")
    config = TaGIConfig()
    device = torch.device("cpu")

    # Use offline / demo mode
    backbone = TaGIBackbone(config, offline=True)
    backbone.to(device).eval()

    # Fake token ids (would normally come from IndicBERT tokeniser)
    B, T = 2, 16
    input_ids      = torch.randint(100, 5000, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[0, 14:] = 0    # pad last 2 positions of first example

    H = backbone.encode(input_ids, attention_mask)
    print(f"Output hidden states H shape: {H.shape}")   # (2, 16, 768)

    weights = backbone.mha.last_weights
    print(f"Attention weights shape      : {weights.shape}")   # (2, 8, 16, 16)

    token_labels = [f"tok_{i}" for i in range(T)]
    head_stats = analyse_head_specialisation(
        backbone, input_ids, attention_mask, token_labels
    )
    print("\nHead long-range attention ratios:")
    for h, ratio in head_stats.items():
        bar = "█" * int(ratio * 20)
        print(f"  Head {h}: {ratio:.3f}  {bar}")

    print("\n[Eq. 4] Attention formula: Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V")
    print(f"[Eq. 5] {config.num_heads} heads × d_k={config.head_dim} = d_model={config.hidden_size}")
    print("[Eq. 6] Outputs concatenated and projected through W^O")