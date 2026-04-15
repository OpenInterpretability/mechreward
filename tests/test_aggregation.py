"""Tests for token→trajectory aggregation strategies."""
import torch

from mechreward.reward.aggregation import (
    LastKAggregation,
    MaxAggregation,
    MeanAggregation,
    WeightedMeanAggregation,
    build_aggregation,
)


def test_mean_aggregation_no_mask():
    acts = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    agg = MeanAggregation()
    assert torch.allclose(agg(acts), torch.tensor([2.5]))


def test_mean_aggregation_with_mask():
    acts = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1, 1, 0, 0]])
    agg = MeanAggregation()
    assert torch.allclose(agg(acts, mask), torch.tensor([1.5]))


def test_last_k_aggregation_simple():
    acts = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    agg = LastKAggregation(k=2)
    assert torch.allclose(agg(acts), torch.tensor([4.5]))


def test_last_k_clips_at_sequence_length():
    acts = torch.tensor([[1.0, 2.0]])
    agg = LastKAggregation(k=100)
    assert torch.allclose(agg(acts), torch.tensor([1.5]))


def test_max_aggregation():
    acts = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
    agg = MaxAggregation()
    assert torch.allclose(agg(acts), torch.tensor([5.0]))


def test_max_aggregation_with_mask():
    acts = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
    mask = torch.tensor([[1, 0, 1, 1]])
    agg = MaxAggregation()
    # Position 1 is masked, so max of [1, 2, 3] = 3
    assert torch.allclose(agg(acts, mask), torch.tensor([3.0]))


def test_weighted_mean_later_tokens_matter_more():
    acts = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    agg = WeightedMeanAggregation()
    out = agg(acts)
    # Uniform activations → weighted mean ≈ mean ≈ 1.0
    assert torch.allclose(out, torch.tensor([1.0]))


def test_build_aggregation_string_dispatch():
    assert isinstance(build_aggregation("mean"), MeanAggregation)
    assert isinstance(build_aggregation("max"), MaxAggregation)
    assert isinstance(build_aggregation("last_k", k=7), LastKAggregation)
    assert build_aggregation("mean_last_32_tokens").k == 32
    assert build_aggregation("last_5").k == 5


def test_build_aggregation_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        build_aggregation("nonsense_aggregation")
