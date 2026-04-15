"""Reward computation: FeatureReward, aggregation, normalization, composition."""

from mechreward.reward.aggregation import (
    Aggregation,
    LastKAggregation,
    MaxAggregation,
    MeanAggregation,
    WeightedMeanAggregation,
    build_aggregation,
)
from mechreward.reward.composition import CompositeReward
from mechreward.reward.feature_reward import FeatureReward
from mechreward.reward.normalization import (
    Normalization,
    RankNormalization,
    SigmoidSquash,
    ZScoreNormalization,
    build_normalization,
)
from mechreward.reward.scheduler import (
    ConstantSchedule,
    CosineSchedule,
    LinearAnnealSchedule,
    WeightSchedule,
)

__all__ = [
    "FeatureReward",
    "Aggregation",
    "MeanAggregation",
    "LastKAggregation",
    "MaxAggregation",
    "WeightedMeanAggregation",
    "build_aggregation",
    "Normalization",
    "ZScoreNormalization",
    "RankNormalization",
    "SigmoidSquash",
    "build_normalization",
    "CompositeReward",
    "WeightSchedule",
    "ConstantSchedule",
    "LinearAnnealSchedule",
    "CosineSchedule",
]
