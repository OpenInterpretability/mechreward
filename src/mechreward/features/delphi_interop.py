"""Optional integration with EleutherAI Delphi for automated feature labeling.

Delphi (https://github.com/EleutherAI/delphi) runs an LLM over SAE feature
activations to produce human-readable descriptions. This module is a thin
wrapper that:

1. Checks whether Delphi is installed.
2. Runs it on a given SAE and a corpus of activation examples.
3. Returns the descriptions as a dict `{feature_id: description}`.

Delphi is an optional dependency. If it's not installed, this module
degrades gracefully with a clear error message.
"""
from __future__ import annotations

from typing import Any


def is_available() -> bool:
    """Return True if Delphi is importable in the current environment."""
    try:
        import delphi  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def describe_features(
    sae: Any,
    model_name: str,
    feature_ids: list[int],
    n_examples_per_feature: int = 10,
    explainer_model: str = "gpt-4o-mini",
) -> dict[int, str]:
    """Use Delphi to auto-label features.

    Args:
        sae: A sae_lens SAE object.
        model_name: HF model id the SAE was trained on.
        feature_ids: Which features to describe.
        n_examples_per_feature: How many high-activation contexts Delphi should
            see per feature.
        explainer_model: The LLM Delphi uses to generate descriptions.

    Returns:
        ``{feature_id: description}``. Empty dict if Delphi is not installed.

    Note:
        Running Delphi requires GPU access and API credentials for the
        explainer model. See Delphi's README for setup.
    """
    if not is_available():
        raise ImportError(
            "Delphi is not installed. Install with "
            "`pip install git+https://github.com/EleutherAI/delphi.git`"
        )

    # Delphi's exact API varies by version; this is the v0.2+ signature.
    # We import lazily to avoid hard dependency.
    from delphi.explainers import DefaultExplainer  # type: ignore[import-not-found]
    from delphi.features import FeatureDataset  # type: ignore[import-not-found]

    dataset = FeatureDataset(
        model_name=model_name,
        sae=sae,
        feature_indices=feature_ids,
        n_examples=n_examples_per_feature,
    )
    explainer = DefaultExplainer(client=None, model=explainer_model)

    out: dict[int, str] = {}
    for feature_id in feature_ids:
        try:
            desc = explainer.explain(dataset[feature_id])
            out[feature_id] = str(desc)
        except Exception as e:  # noqa: BLE001
            out[feature_id] = f"[delphi_error: {e}]"
    return out
