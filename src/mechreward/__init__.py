"""mechreward — mechanistic interpretability as reward signal for RL training of LLMs."""

from mechreward.__version__ import __version__
from mechreward.features.catalog import FeaturePack, load_pack, save_pack
from mechreward.probes.linear_probe import LinearProbe, load_probe
from mechreward.reward.aggregation import (
    LastKAggregation,
    MaxAggregation,
    MeanAggregation,
)
from mechreward.reward.composition import CompositeReward
from mechreward.reward.feature_reward import FeatureReward
from mechreward.sae.loader import SAEHandle, load_sae

__all__ = [
    "__version__",
    "load_sae",
    "SAEHandle",
    "FeaturePack",
    "load_pack",
    "save_pack",
    "FeatureReward",
    "CompositeReward",
    "MeanAggregation",
    "LastKAggregation",
    "MaxAggregation",
    "LinearProbe",
    "load_probe",
]
