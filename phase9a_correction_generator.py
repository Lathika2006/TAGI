"""
Phase 9a: Grammar Correction Generator
TaGI Paper — Eq. (9):  P(y_t | H) = Softmax(W_g h_t + b_g)
             Eq. (10): ŷ = argmax_y Σ_t log P(y_t | y_{<t}, H)

Implements:
  - Autoregressive decoder conditioned on IndicBERT hidden states H
  - Beam search with configurable beam width (b=5 in the paper)
  - Token-level correction: replaces each erroneous position
  - Full sentence reconstruction after correction
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from phase5_6_embedding_attention import TaGIConfig, TaGIBackbone, MultiHeadAttention


# ---------------------------------------------------------------------------
# Special tokens

# ---------------------------------------------------------------------------

START_TOKEN = 0   # <START>
END_TOKEN   = 1   # <END>
PAD_TOKEN   = 2   # <PAD>
UNK_TOKEN   = 3   # <UNK>


# ---------------------------------------------------------------------------
# Transformer Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer:
      1. Masked self-attention over generated tokens
      2. Cross-attention over encoder hidden states H
      3. Feed-forward network
    """
    def __init__(self, config: TaGIConfig):
        super().__init__()
        d = config.hidden_size

        # Masked self-attention (causal)
        self.self_attn   = MultiHeadAttention(config)
        # Cross-attention to encoder H
        self.cross_attn  = nn.MultiheadAttention(d, config.num_heads, dropout=config.dropout, batch_first=True)
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d * 4, d),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.drop  = nn.Dropout(config.dropout)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask for autoregressive decoding."""
        mask = torch.ones(T, T, device=device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)    # True = mask out (attend to past only)
        return mask                             # shape (T, T)

    def forward(
        self,
        tgt:          torch.Tensor,    # (B, T_dec, d)  decoder input
        memory:       torch.Tensor,    # (B, T_enc, d)  encoder hidden states H
        tgt_key_mask: Optional[torch.Tensor] = None,
        mem_key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = tgt.size(1)
        causal = self._causal_mask(T, tgt.device)

        # 1. Masked self-attention
        sa = self.self_attn(tgt, mask=(~causal).unsqueeze(0).unsqueeze(0))
        tgt = self.norm1(tgt + sa)

        # 2. Cross-attention to encoder
        # nn.MultiheadAttention uses key_padding_mask (True = ignore)
        ca, _ = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=mem_key_mask,
        )
        tgt = self.norm2(tgt + self.drop(ca))

        # 3. Feed-forward
        tgt = self.norm3(tgt + self.drop(self.ff(tgt)))
        return tgt


# ---------------------------------------------------------------------------
# Grammar Correction Decoder
# ---------------------------------------------------------------------------

class GrammarCorrectionDecoder(nn.Module):
    """
    6-layer Transformer decoder for correction generation.
    Eq. (9): P(y_t | H) = Softmax(W_g h_t + b_g)
    """
    def __init__(self, config: TaGIConfig, vocab_size: int = 32000, num_layers: int = 6):
        super().__init__()
        d = config.hidden_size
        self.config     = config
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d, padding_idx=PAD_TOKEN)
        self.pos_enc   = PositionalEncoding(d, config.max_seq_len, config.dropout)
        self.layers    = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])

        # Eq. (9): W_g and b_g — output projection
        self.W_g = nn.Linear(d, vocab_size)   # W_g in paper
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.W_g.weight, std=0.02)
        nn.init.zeros_(self.W_g.bias)

    def forward(
        self,
        tgt_ids:   torch.Tensor,    # (B, T_dec)  target token ids
        memory:    torch.Tensor,    # (B, T_enc, d)  encoder H
        mem_mask:  Optional[torch.Tensor] = None,   # (B, T_enc) attention mask
    ) -> torch.Tensor:
        """Returns log-probabilities over vocabulary: (B, T_dec, V)"""
        # Key-padding mask for cross-attention: True = ignore PAD positions
        mem_key_mask = None
        if mem_mask is not None:
            mem_key_mask = (mem_mask == 0)   # (B, T_enc) — True where PAD

        x = self.pos_enc(self.embedding(tgt_ids))   # (B, T, d)
        for layer in self.layers:
            x = layer(x, memory, mem_key_mask=mem_key_mask)

        logits = self.W_g(x)                         # (B, T, V) — Eq. (9)
        return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.drop(x)


# ---------------------------------------------------------------------------
# Beam Search  (Eq. 10)
# ---------------------------------------------------------------------------

@dataclass
class BeamHypothesis:
    token_ids: List[int]
    score:     float        # Σ_t log P(y_t | y_{<t}, H)

    def __lt__(self, other):
        return self.score < other.score


def beam_search(
    decoder:   GrammarCorrectionDecoder,
    memory:    torch.Tensor,    # (1, T_enc, d)  single example
    mem_mask:  Optional[torch.Tensor],
    beam_width: int = 5,
    max_length: int = 64,
    device:    torch.device = torch.device("cpu"),
) -> List[int]:
    """
    Eq. (10): ŷ = argmax_y Σ_t log P(y_t | y_{<t}, H)

    Standard beam search with beam_width hypotheses.
    Returns the token-id sequence of the best hypothesis.
    """
    # Initialise beams with <START>
    beams: List[BeamHypothesis] = [BeamHypothesis([START_TOKEN], 0.0)]
    completed: List[BeamHypothesis] = []

    for _ in range(max_length):
        candidates: List[BeamHypothesis] = []

        for hyp in beams:
            if hyp.token_ids[-1] == END_TOKEN:
                completed.append(hyp)
                continue

            # Decode current sequence
            tgt = torch.tensor([hyp.token_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                log_probs = decoder(tgt, memory, mem_mask)    # (1, T, V)
            step_log_probs = log_probs[0, -1]                 # (V,) — last timestep

            # Top-k expansion
            topk_lp, topk_ids = step_log_probs.topk(beam_width)

            for lp, tok_id in zip(topk_lp.tolist(), topk_ids.tolist()):
                candidates.append(BeamHypothesis(
                    token_ids=hyp.token_ids + [tok_id],
                    score=hyp.score + lp,
                ))

        if not candidates:
            break

        # Keep top beam_width candidates  (Eq. 10)
        candidates.sort(key=lambda h: h.score, reverse=True)
        beams = candidates[:beam_width]

        # Early stop if all beams have finished
        if all(h.token_ids[-1] == END_TOKEN for h in beams):
            completed.extend(beams)
            break

    all_hyps = completed + beams
    best = max(all_hyps, key=lambda h: h.score)
    # Strip <START> and <END>
    ids = best.token_ids
    if ids and ids[0] == START_TOKEN:
        ids = ids[1:]
    if ids and ids[-1] == END_TOKEN:
        ids = ids[:-1]
    return ids


# ---------------------------------------------------------------------------
# Token-level corrector
# ---------------------------------------------------------------------------

class TokenCorrector:
    """
    Applies the GrammarCorrectionDecoder to each flagged error position.
    The correction for each position is generated independently, conditioned
    on the full sentence's encoder representation H.
    """
    def __init__(
        self,
        decoder:    GrammarCorrectionDecoder,
        config:     TaGIConfig,
        device:     torch.device,
    ):
        self.decoder = decoder.to(device).eval()
        self.config  = config
        self.device  = device

    def correct_sentence(
        self,
        tokens:         List[str],
        error_positions: List[int],        # indices of ERROR tokens
        memory:         torch.Tensor,      # (1, T_enc, d)
        mem_mask:       torch.Tensor,      # (1, T_enc)
        id_to_token:    Dict[int, str],    # vocabulary
    ) -> Tuple[List[str], List[str]]:
        """
        Generate corrections for each error position using beam search.
        Returns (corrected_tokens, original_tokens).
        """
        corrected = list(tokens)

        for pos in error_positions:
            # Run beam search to generate the best replacement token
            best_ids = beam_search(
                self.decoder,
                memory,
                mem_mask,
                beam_width=self.config.beam_width,
                max_length=self.config.max_gen_len,
                device=self.device,
            )
            if best_ids:
                # Use the first generated token as the single-token replacement
                corrected[pos] = id_to_token.get(best_ids[0], tokens[pos])

        return corrected, tokens


# ---------------------------------------------------------------------------
# Full correction model (backbone + decoder)
# ---------------------------------------------------------------------------

class TaGICorrectionModel(nn.Module):
    """
    End-to-end correction model.
    Phase 5 backbone encodes the source; Phase 9a decoder generates corrections.
    """
    def __init__(self, config: TaGIConfig, vocab_size: int = 32000, offline: bool = False):
        super().__init__()
        self.backbone = TaGIBackbone(config, offline=offline)
        self.decoder  = GrammarCorrectionDecoder(config, vocab_size)
        self.config   = config

    def forward(
        self,
        src_input_ids:  torch.Tensor,
        src_attn_mask:  torch.Tensor,
        tgt_input_ids:  torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forcing forward for training."""
        H    = self.backbone.encode(src_input_ids, src_attn_mask)
        lp   = self.decoder(tgt_input_ids, H, src_attn_mask)
        return lp    # (B, T_dec, V)

    def compute_generation_loss(
        self,
        src_input_ids:  torch.Tensor,
        src_attn_mask:  torch.Tensor,
        tgt_input_ids:  torch.Tensor,    # shifted right (input to decoder)
        tgt_labels:     torch.Tensor,    # (B, T) target labels
    ) -> torch.Tensor:
        log_probs = self.forward(src_input_ids, src_attn_mask, tgt_input_ids)
        B, T, V   = log_probs.shape
        loss = F.nll_loss(
            log_probs.view(B * T, V),
            tgt_labels.view(B * T),
            ignore_index=PAD_TOKEN,
        )
        return loss


# ---------------------------------------------------------------------------
# BLEU / ROUGE evaluation helpers
# ---------------------------------------------------------------------------

def compute_bleu(hypotheses: List[List[str]], references: List[List[str]]) -> float:
    """Compute corpus BLEU-4."""
    try:
        import sacrebleu
        hyp_str = [" ".join(h) for h in hypotheses]
        ref_str = [[" ".join(r) for r in references]]
        bleu = sacrebleu.corpus_bleu(hyp_str, ref_str)
        return bleu.score
    except ImportError:
        # Manual 4-gram BLEU approximation
        from collections import Counter

        def ngrams(seq, n):
            return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]

        total_clip = total_count = 0
        for hyp, ref in zip(hypotheses, references):
            for n in range(1, 5):
                hyp_ng = Counter(ngrams(hyp, n))
                ref_ng = Counter(ngrams(ref, n))
                clip = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
                total_clip  += clip
                total_count += sum(hyp_ng.values())

        precision = total_clip / max(total_count, 1)
        bp = 1.0   # simplified — no brevity penalty
        return bp * math.exp(math.log(max(precision, 1e-8))) * 100


def compute_rouge_l(hypothesis: List[str], reference: List[str]) -> float:
    """ROUGE-L F1 via LCS."""
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    l = lcs(hypothesis, reference)
    p = l / max(len(hypothesis), 1)
    r = l / max(len(reference),  1)
    f = 2 * p * r / max(p + r, 1e-8)
    return f


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Phase 9a Demo: Grammar Correction Generator (Eq. 9–10) ===\n")

    config  = TaGIConfig()
    device  = torch.device("cpu")
    VOCAB   = 32000

    backbone = TaGIBackbone(config, offline=True)
    decoder  = GrammarCorrectionDecoder(config, vocab_size=VOCAB, num_layers=2)

    backbone.eval(); decoder.eval()

    # Simulate encoder hidden states (batch=1, seq=10)
    memory    = torch.randn(1, 10, config.hidden_size)
    mem_mask  = torch.ones(1, 10, dtype=torch.long)

    print(f"Beam width b={config.beam_width}, max_gen_len={config.max_gen_len}")
    best_ids = beam_search(decoder, memory, mem_mask,
                           beam_width=config.beam_width,
                           max_length=8, device=device)
    print(f"Beam search output token ids: {best_ids}")

    # BLEU / ROUGE demo
    hyps = [["அவர்கள்", "வருகிறார்கள்"]]
    refs = [["அவர்கள்", "வருகிறார்கள்"]]
    bleu = compute_bleu(hyps, refs)
    print(f"\nBLEU-4 (perfect match demo): {bleu:.1f}")

    rl = compute_rouge_l(hyps[0], refs[0])
    print(f"ROUGE-L (perfect match demo): {rl:.4f}")

    print("\n[Eq. 9]  P(y_t|H) = Softmax(W_g h_t + b_g)")
    print(f"[Eq. 10] ŷ = argmax Σ_t log P(y_t|y_{{<t}}, H)  with beam_width={config.beam_width}")