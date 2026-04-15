"""ReasonScore: identify SAE features active during reasoning moments.

Port of the metric from Gao et al., "I Have Covered All the Bases Here:
Interpreting Reasoning Features in Large Language Models via Sparse
Autoencoders" (arxiv:2503.18878), AIRI Institute.

The score is contrastive between a reasoning corpus D_R and a baseline
corpus D_nR, both restricted to a context window around occurrences of
a small reasoning vocabulary (wait, hmm, therefore, ...)::

    ReasonScore_i = p(i | D_R^W) * H_i^alpha  -  p(i | D_nR^W)

where p(i | D^W) = mu(i, D^W) / sum_j mu(j, D^W) and H_i is the entropy
of feature i's activation distribution over the reasoning words. The
entropy exponent (alpha=0.7 in the paper) rewards features that fire
on several reasoning words and penalises features that latch onto a
single word.

Example::

    from mechreward.sae.topk_sae import load_topk_sae
    from mechreward.features.reasonscore import (
        compute_reasonscore, resolve_vocab_token_ids,
    )
    from transformers import AutoTokenizer

    sae = load_topk_sae('sae_final.pt', layer=18)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B')
    vocab = resolve_vocab_token_ids(tok)

    scores = compute_reasonscore(
        sae,
        reasoning_samples=reasoning_activations,  # iterable of (hidden, tokens)
        baseline_samples=baseline_activations,
        reasoning_vocab=vocab,
        top_k=200,
    )
    # scores[0] is the feature with highest ReasonScore.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mechreward.sae.loader import SAEHandle

Sample = tuple[torch.Tensor, torch.Tensor]
"""(hidden_states [T, d_model], token_ids [T]) for a single example."""


DEFAULT_REASONING_VOCAB: tuple[str, ...] = (
    "alternatively",
    "hmm",
    "maybe",
    "wait",
    "perhaps",
    "let me",
    "therefore",
    "however",
    "but",
    "another",
)
"""The 10 reasoning trigger words from arxiv:2503.18878."""


@dataclass
class FeatureScore:
    """ReasonScore components for a single SAE feature."""

    feature_id: int
    reason_score: float
    entropy: float
    prob_reasoning: float
    prob_baseline: float


def _dilate_mask(hit: torch.Tensor, before: int, after: int) -> torch.Tensor:
    """Dilate a [T] bool hit mask by (before, after) positions.

    A position t becomes True iff any hit lies in ``[t - before, t + after]``.
    Equivalently, a hit at position p paints positions ``[p - before, p + after]``.
    Implemented as a 1D max-pool so it runs on GPU.
    """
    T = hit.shape[0]
    kernel = before + after + 1
    # Pad so a hit at original position p ends up at padded position p + after,
    # and max-pool output[t] covers padded[t..t+kernel) = original[t-after..t+before].
    # That makes a hit at p influence outputs t in [p-before, p+after].
    padded = F.pad(hit.float().view(1, 1, T), (after, before), value=0.0)
    pooled = F.max_pool1d(padded, kernel_size=kernel, stride=1)
    return pooled.view(T).bool()


def _validate_sample(hidden: torch.Tensor, tokens: torch.Tensor) -> None:
    if hidden.dim() != 2:
        raise ValueError(f"hidden must be [T, d_model], got {tuple(hidden.shape)}")
    if tokens.dim() != 1:
        raise ValueError(f"tokens must be [T], got {tuple(tokens.shape)}")
    if hidden.shape[0] != tokens.shape[0]:
        raise ValueError(
            f"hidden and tokens length mismatch: "
            f"{hidden.shape[0]} vs {tokens.shape[0]}"
        )


def _encode_sample(sae: SAEHandle, hidden: torch.Tensor) -> torch.Tensor:
    """Encode one [T, d_model] sample and return [T, d_sae] float64."""
    with torch.inference_mode():
        acts = sae.encode(hidden.to(sae.device, torch.float32))
    return acts.to(torch.float64)


def compute_reasonscore(
    sae: SAEHandle,
    reasoning_samples: Iterable[Sample],
    baseline_samples: Iterable[Sample],
    reasoning_vocab: dict[str, Sequence[int]],
    context_before: int = 2,
    context_after: int = 3,
    entropy_alpha: float = 0.7,
    top_k: int | None = None,
    device: torch.device | str | None = None,
) -> list[FeatureScore]:
    """Compute ReasonScore for every SAE feature and return them ranked.

    Args:
        sae: Trained SAE handle (any ``SAEBackend``-compatible).
        reasoning_samples: Iterable of ``(hidden [T, d_model], tokens [T])``
            pairs drawn from reasoning traces (D_R).
        baseline_samples: Same shape, from non-reasoning text (D_nR).
        reasoning_vocab: Dict mapping each reasoning word to a list of token
            ids that count as a hit. For multi-token phrases like "let me",
            pass only the head-subtoken id.
        context_before: Window size before a hit. Paper default: 2.
        context_after: Window size after a hit. Paper default: 3.
        entropy_alpha: Entropy exponent. Paper default: 0.7.
        top_k: If set, return only the top-k ranked features.
        device: Override compute device. Defaults to the SAE's device.

    Returns:
        List of ``FeatureScore`` sorted by ``reason_score`` descending.
    """
    if device is None:
        device = sae.device
    device = torch.device(device)

    words = list(reasoning_vocab.keys())
    n_words = len(words)
    if n_words == 0:
        raise ValueError("reasoning_vocab must contain at least one word")

    word_id_tensors: list[torch.Tensor] = []
    for w in words:
        ids = reasoning_vocab[w]
        if not ids:
            raise ValueError(f"reasoning_vocab['{w}'] is empty")
        word_id_tensors.append(
            torch.tensor(list(ids), dtype=torch.long, device=device)
        )
    all_vocab_ids = torch.cat(word_id_tensors).unique()

    d_sae = sae.d_sae
    sum_R = torch.zeros(d_sae, device=device, dtype=torch.float64)
    sum_NR = torch.zeros(d_sae, device=device, dtype=torch.float64)
    cnt_R = 0
    cnt_NR = 0
    pw_sum = torch.zeros(n_words, d_sae, device=device, dtype=torch.float64)
    pw_cnt = torch.zeros(n_words, device=device, dtype=torch.float64)

    # Reasoning stream — accumulates full window sum AND per-word breakdown.
    for hidden, tokens in reasoning_samples:
        hidden = hidden.to(device)
        tokens = tokens.to(device)
        _validate_sample(hidden, tokens)

        acts = _encode_sample(sae, hidden)
        full_hit = torch.isin(tokens, all_vocab_ids)
        if not full_hit.any():
            continue

        full_mask = _dilate_mask(full_hit, context_before, context_after)
        sum_R += acts[full_mask].sum(dim=0)
        cnt_R += int(full_mask.sum().item())

        for w_idx in range(n_words):
            hit_w = torch.isin(tokens, word_id_tensors[w_idx])
            if not hit_w.any():
                continue
            mask_w = _dilate_mask(hit_w, context_before, context_after)
            pw_sum[w_idx] += acts[mask_w].sum(dim=0)
            pw_cnt[w_idx] += mask_w.sum().to(torch.float64)

    # Baseline stream — only full window sum.
    for hidden, tokens in baseline_samples:
        hidden = hidden.to(device)
        tokens = tokens.to(device)
        _validate_sample(hidden, tokens)

        acts = _encode_sample(sae, hidden)
        full_hit = torch.isin(tokens, all_vocab_ids)
        if not full_hit.any():
            continue
        full_mask = _dilate_mask(full_hit, context_before, context_after)
        sum_NR += acts[full_mask].sum(dim=0)
        cnt_NR += int(full_mask.sum().item())

    if cnt_R == 0:
        raise RuntimeError(
            "No reasoning-vocab hits found in reasoning_samples. "
            "Verify the reasoning_vocab token ids match your tokenizer."
        )

    mu_R = sum_R / cnt_R
    mu_NR = sum_NR / cnt_NR if cnt_NR > 0 else torch.zeros_like(sum_NR)

    eps = 1e-12
    p_R = mu_R / (mu_R.sum() + eps)
    p_NR = mu_NR / (mu_NR.sum() + eps)

    # Per-feature distribution over the n_words reasoning words.
    safe_cnt = pw_cnt.clamp_min(1.0).unsqueeze(-1)
    pw_mu = pw_sum / safe_cnt
    pw_col_sum = pw_mu.sum(dim=0, keepdim=True)
    pw_dist = pw_mu / (pw_col_sum + eps)
    log_term = (pw_dist + eps).log()
    H = -(pw_dist * log_term).sum(dim=0)
    H = H.clamp_min(0.0)

    reasonscore = p_R * H.pow(entropy_alpha) - p_NR

    rs = reasonscore.cpu()
    H_cpu = H.cpu()
    pR_cpu = p_R.cpu()
    pNR_cpu = p_NR.cpu()

    scores = [
        FeatureScore(
            feature_id=i,
            reason_score=float(rs[i].item()),
            entropy=float(H_cpu[i].item()),
            prob_reasoning=float(pR_cpu[i].item()),
            prob_baseline=float(pNR_cpu[i].item()),
        )
        for i in range(d_sae)
    ]
    scores.sort(key=lambda s: s.reason_score, reverse=True)
    if top_k is not None:
        scores = scores[:top_k]
    return scores


def resolve_vocab_token_ids(
    tokenizer,
    words: Sequence[str] = DEFAULT_REASONING_VOCAB,
    add_leading_space: bool = True,
) -> dict[str, list[int]]:
    """Resolve a word list to tokenizer-specific token ids.

    Most BPE tokenizers emit different ids for "wait" vs " wait" (with the
    space). We try both forms and keep the head-token id of each. For
    multi-token phrases like "let me", only the head token is kept -- the
    paper matches on head tokens as well.

    Args:
        tokenizer: A HuggingFace tokenizer.
        words: Word list. Defaults to the paper's 10 words.
        add_leading_space: Also try the space-prefixed form.

    Returns:
        Dict word -> sorted list of unique head token ids. Words that the
        tokenizer rejects map to an empty list (callers should drop them).
    """
    out: dict[str, list[int]] = {}
    for word in words:
        seen: set[int] = set()
        variants: list[str] = [word]
        if add_leading_space and not word.startswith(" "):
            variants.append(" " + word)
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                seen.add(int(ids[0]))
        out[word] = sorted(seen)
    return out
