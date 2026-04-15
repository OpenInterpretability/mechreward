"""Feature faithfulness validation.

Before trusting a feature as a reward component, we need evidence that it
actually does what its label claims. This module runs a contrastive test:

1. Generate N "positive" text examples that should activate the feature.
2. Generate N "negative" examples that should not.
3. Encode both through the SAE at the target layer.
4. Check that positives activate >> negatives on this specific feature.

If the AUC of positives vs negatives on the feature activation is below a
threshold, the feature is marked unvalidated and downweighted/dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mechreward.sae.batched_encode import batched_encode_selective
from mechreward.sae.loader import SAEHandle


@dataclass
class ValidationResult:
    feature_id: int
    auc: float
    positive_mean: float
    negative_mean: float
    validated: bool
    threshold: float
    n_pos: int
    n_neg: int


def _activate_layer(
    model,
    tokenizer,
    texts: list[str],
    layer_idx: int,
    max_length: int = 256,
    batch_size: int = 4,
) -> torch.Tensor:
    """Run a HF model on a list of texts and return hidden states at a layer.

    Returns:
        Tensor of shape ``[N, T, d_model]`` padded to the longest sequence.
    """
    device = next(model.parameters()).device
    model.eval()

    all_hidden = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.inference_mode():
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        # hidden_states: tuple of (num_layers+1) tensors, each [B, T, d_model]
        hs = outputs.hidden_states[layer_idx]  # +1 for embed
        all_hidden.append(hs.float().cpu())

    # Pad to max T
    max_t = max(h.shape[1] for h in all_hidden)
    padded = []
    for h in all_hidden:
        if h.shape[1] < max_t:
            pad = torch.zeros(h.shape[0], max_t - h.shape[1], h.shape[2])
            h = torch.cat([h, pad], dim=1)
        padded.append(h)
    return torch.cat(padded, dim=0)


def _compute_auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Approximate AUC between positive and negative score distributions."""
    # Simple Mann-Whitney based AUC
    pos_flat = pos.flatten()
    neg_flat = neg.flatten()
    n_pos = pos_flat.numel()
    n_neg = neg_flat.numel()
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Faster approximation: mean(pos) > mean(neg) scaled into AUC
    # For exact Mann-Whitney on small samples this is O(n*m); we cap at 10k pairs.
    max_pairs = 10_000
    if n_pos * n_neg <= max_pairs:
        wins = 0.0
        total = 0.0
        for p in pos_flat.tolist():
            for n in neg_flat.tolist():
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5
                total += 1.0
        return wins / total if total > 0 else 0.5

    # Fall back to a threshold-at-midpoint heuristic
    combined = torch.cat([pos_flat, neg_flat])
    median = combined.median().item()
    tp = (pos_flat > median).float().mean().item()
    fp = (neg_flat > median).float().mean().item()
    # AUC ≈ 0.5 + 0.5 * (tp - fp)
    return max(0.0, min(1.0, 0.5 + 0.5 * (tp - fp)))


def validate_feature(
    sae: SAEHandle,
    model,
    tokenizer,
    feature_id: int,
    positive_examples: list[str],
    negative_examples: list[str],
    auc_threshold: float = 0.75,
    layer_idx: int | None = None,
) -> ValidationResult:
    """Test whether a feature discriminates positive from negative examples.

    Args:
        sae: The SAE to test.
        model: The base HF model the SAE was trained on.
        tokenizer: Matching tokenizer.
        feature_id: Which feature to test.
        positive_examples: Texts expected to activate the feature.
        negative_examples: Texts expected not to activate.
        auc_threshold: Minimum AUC to mark the feature validated.
        layer_idx: Override the SAE's default layer. Defaults to sae.layer.

    Returns:
        A ``ValidationResult`` with AUC and pass/fail.
    """
    if layer_idx is None:
        layer_idx = sae.layer
    if layer_idx is None or layer_idx < 0:
        raise ValueError(
            f"SAE layer is unknown ({sae.layer}); pass layer_idx explicitly."
        )

    pos_hidden = _activate_layer(model, tokenizer, positive_examples, layer_idx)
    neg_hidden = _activate_layer(model, tokenizer, negative_examples, layer_idx)

    # Take last token of each (most informative for completion-style tasks)
    pos_last = pos_hidden[:, -1, :]
    neg_last = neg_hidden[:, -1, :]

    pos_act = batched_encode_selective(sae, pos_last, [feature_id]).squeeze(-1)
    neg_act = batched_encode_selective(sae, neg_last, [feature_id]).squeeze(-1)

    auc = _compute_auc(pos_act, neg_act)
    pos_mean = float(pos_act.mean().item())
    neg_mean = float(neg_act.mean().item())

    return ValidationResult(
        feature_id=feature_id,
        auc=auc,
        positive_mean=pos_mean,
        negative_mean=neg_mean,
        validated=auc >= auc_threshold,
        threshold=auc_threshold,
        n_pos=len(positive_examples),
        n_neg=len(negative_examples),
    )
