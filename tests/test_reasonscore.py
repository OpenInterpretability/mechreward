"""Tests for the ReasonScore feature-discovery metric."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mechreward.features.reasonscore import (
    DEFAULT_REASONING_VOCAB,
    _dilate_mask,
    compute_reasonscore,
    resolve_vocab_token_ids,
)
from mechreward.sae.loader import SAEHandle


class _IdentitySAE(nn.Module):
    """Mock SAE where encode is the identity.

    ``encode(x)`` returns ``x`` padded/truncated to ``d_sae``, so feature i
    activation equals ``x[..., i]``. This lets tests construct deterministic
    activation patterns without training anything.
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        assert d_sae >= d_model, "identity mock needs d_sae >= d_model"
        self.d_model = d_model
        self.d_sae = d_sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.d_sae:
            pad_shape = list(x.shape)
            pad_shape[-1] = self.d_sae - x.shape[-1]
            pad = torch.zeros(*pad_shape, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=-1)
        return x[..., : self.d_sae]

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return y[..., : self.d_model]

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


def _mock_handle(d_model: int = 8, d_sae: int = 8) -> SAEHandle:
    return SAEHandle(
        backend=_IdentitySAE(d_model, d_sae),
        release="mock",
        sae_id="mock",
        hook_name="mock.hook",
        layer=0,
        d_model=d_model,
        d_sae=d_sae,
        model_name="mock",
    )


def test_dilate_mask_symmetric():
    hit = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.bool)
    dilated = _dilate_mask(hit, before=2, after=2)
    assert dilated.tolist() == [0, 1, 1, 1, 1, 1, 0]


def test_dilate_mask_paper_defaults():
    # paper uses before=2, after=3 (asymmetric)
    hit = torch.zeros(10, dtype=torch.bool)
    hit[5] = True
    dilated = _dilate_mask(hit, before=2, after=3)
    # window around position 5: [3, 4, 5, 6, 7, 8]
    expected = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    assert dilated.tolist() == expected


def test_dilate_mask_boundary():
    hit = torch.tensor([1, 0, 0, 0, 1], dtype=torch.bool)
    dilated = _dilate_mask(hit, before=1, after=1)
    # hit at 0 -> [0, 1]; hit at 4 -> [3, 4]; union = [0, 1, 3, 4]
    assert dilated.tolist() == [1, 1, 0, 1, 1]


def test_dilate_mask_multiple_hits_merge():
    hit = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool)
    dilated = _dilate_mask(hit, before=1, after=1)
    # windows [0,1,2] and [3,4,5] merge into full mask
    assert dilated.tolist() == [1, 1, 1, 1, 1, 1]


def test_compute_reasonscore_ranks_multi_word_feature_above_single_word():
    """Rig a scenario where feat0 fires on both reasoning words (should win)
    and feat3 fires on only one (should be penalised by the entropy term).
    """
    torch.manual_seed(0)
    d_model = 8
    d_sae = 8
    sae = _mock_handle(d_model, d_sae)

    vocab = {"word_a": [100], "word_b": [101]}
    T = 20
    VOCAB_POS = 10

    def sample_with_vocab(vocab_id: int, active_feat: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.arange(T) + 1
        tokens[VOCAB_POS] = vocab_id
        hidden = torch.zeros(T, d_model)
        if active_feat is not None:
            hidden[:, active_feat] = 1.0
        return hidden, tokens

    reasoning = []
    # feat0 fires on samples with word_a AND word_b
    for _ in range(5):
        reasoning.append(sample_with_vocab(100, active_feat=0))
    for _ in range(5):
        reasoning.append(sample_with_vocab(101, active_feat=0))
    # feat3 fires only on samples with word_a
    for _ in range(5):
        reasoning.append(sample_with_vocab(100, active_feat=3))

    # Baseline: no vocab tokens at all
    baseline = []
    for _ in range(5):
        tokens = torch.arange(T) + 1
        hidden = torch.zeros(T, d_model)
        baseline.append((hidden, tokens))

    scores = compute_reasonscore(
        sae,
        reasoning_samples=reasoning,
        baseline_samples=baseline,
        reasoning_vocab=vocab,
        context_before=2,
        context_after=3,
        entropy_alpha=0.7,
    )

    assert len(scores) == d_sae

    feat0 = next(s for s in scores if s.feature_id == 0)
    feat3 = next(s for s in scores if s.feature_id == 3)

    assert feat0.reason_score > feat3.reason_score, (
        f"feat0={feat0.reason_score:.4f}, feat3={feat3.reason_score:.4f}"
    )
    # feat0 activates on both words -> entropy close to log(2) ≈ 0.693
    assert feat0.entropy > 0.55, f"feat0.entropy={feat0.entropy:.4f}"
    # feat3 activates on one word only -> entropy ≈ 0
    assert feat3.entropy < 0.05, f"feat3.entropy={feat3.entropy:.4f}"
    # feat0 should be at the top of the ranking
    assert scores[0].feature_id == 0


def test_compute_reasonscore_raises_on_no_vocab_hits():
    sae = _mock_handle()
    tokens = torch.arange(20) + 1000
    hidden = torch.zeros(20, 8)
    with pytest.raises(RuntimeError, match="No reasoning-vocab hits"):
        compute_reasonscore(
            sae,
            reasoning_samples=[(hidden, tokens)],
            baseline_samples=[(hidden, tokens)],
            reasoning_vocab={"word_a": [100]},
        )


def test_compute_reasonscore_top_k_respected():
    torch.manual_seed(0)
    sae = _mock_handle(d_model=8, d_sae=16)
    T = 20
    tokens = torch.arange(T) + 1
    tokens[10] = 100
    tokens[15] = 101
    hidden = torch.randn(T, 8).abs()
    scores = compute_reasonscore(
        sae,
        reasoning_samples=[(hidden, tokens)],
        baseline_samples=[(torch.zeros(T, 8), torch.arange(T) + 1)],
        reasoning_vocab={"a": [100], "b": [101]},
        top_k=3,
    )
    assert len(scores) == 3
    assert scores[0].reason_score >= scores[1].reason_score >= scores[2].reason_score


def test_compute_reasonscore_rejects_empty_vocab():
    sae = _mock_handle()
    tokens = torch.arange(10) + 1
    hidden = torch.zeros(10, 8)
    with pytest.raises(ValueError, match="at least one word"):
        compute_reasonscore(
            sae,
            reasoning_samples=[(hidden, tokens)],
            baseline_samples=[(hidden, tokens)],
            reasoning_vocab={},
        )


def test_compute_reasonscore_rejects_word_with_empty_ids():
    sae = _mock_handle()
    tokens = torch.arange(10) + 1
    hidden = torch.zeros(10, 8)
    with pytest.raises(ValueError, match="is empty"):
        compute_reasonscore(
            sae,
            reasoning_samples=[(hidden, tokens)],
            baseline_samples=[(hidden, tokens)],
            reasoning_vocab={"word_a": []},
        )


def test_resolve_vocab_token_ids_keeps_head_subtokens():
    class MockTok:
        def encode(self, text, add_special_tokens=False):
            mapping = {
                "wait": [1],
                " wait": [2],
                "hmm": [3],
                " hmm": [3],  # same id for both variants
            }
            return mapping.get(text, [])

    vocab = resolve_vocab_token_ids(MockTok(), words=["wait", "hmm", "unknown"])
    assert vocab["wait"] == [1, 2]
    assert vocab["hmm"] == [3]
    assert vocab["unknown"] == []


def test_default_reasoning_vocab_has_ten_words():
    assert len(DEFAULT_REASONING_VOCAB) == 10
    # sanity: the paper's key trigger words should be present
    assert "wait" in DEFAULT_REASONING_VOCAB
    assert "hmm" in DEFAULT_REASONING_VOCAB
    assert "therefore" in DEFAULT_REASONING_VOCAB
