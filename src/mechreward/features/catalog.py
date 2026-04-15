"""FeaturePack: a JSON-backed, versioned bundle of SAE features with metadata."""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any


@dataclass
class Feature:
    """A single SAE feature with metadata needed to use it as a reward component.

    Attributes:
        feature_id: Integer index into the SAE latent space.
        name: Human-readable label (e.g. "confident_assertion").
        description: Sentence-level description produced by auto-interp.
        weight: Contribution to the reward (positive = reward, negative = penalty).
        release: SAE release this feature is defined against.
        sae_id: Specific SAE within the release.
        activation_threshold: Minimum activation to count as "on".
        validated: Whether this feature passed a faithfulness test.
        metadata: Free-form extra info (auto-interp confidence, source, etc.).
    """

    feature_id: int
    name: str
    description: str
    weight: float = 1.0
    release: str = ""
    sae_id: str = ""
    activation_threshold: float = 0.0
    validated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Feature:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FeaturePack:
    """A named bundle of features, version-stamped and targeting a specific model.

    Feature packs are the unit that end-users consume. A user picks a pack by
    name (e.g. "gemma-2-9b/reasoning_pack"), and mechreward loads the SAE,
    resolves the features, and plugs everything into the reward computation.
    """

    name: str
    version: str = "0.1.0"
    model_name: str = ""
    release: str = ""
    sae_id: str = ""
    description: str = ""
    features: list[Feature] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def feature_ids(self) -> list[int]:
        return [f.feature_id for f in self.features]

    def feature_weights(self) -> dict[int, float]:
        return {f.feature_id: f.weight for f in self.features}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model_name": self.model_name,
            "release": self.release,
            "sae_id": self.sae_id,
            "description": self.description,
            "features": [f.to_dict() for f in self.features],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeaturePack:
        features = [Feature.from_dict(f) for f in d.get("features", [])]
        return cls(
            name=d["name"],
            version=d.get("version", "0.1.0"),
            model_name=d.get("model_name", ""),
            release=d.get("release", ""),
            sae_id=d.get("sae_id", ""),
            description=d.get("description", ""),
            features=features,
            metadata=d.get("metadata", {}),
        )


def _resolve_pack_path(name: str) -> Path:
    """Resolve a pack name to an on-disk path.

    Search order:
    1. Bundled catalogs/ directory (via importlib.resources when installed)
    2. Project-root catalogs/ (when running from source)
    3. Absolute path if `name` is already a path to a .json file
    """
    # Case 3: literal path
    literal = Path(name)
    if literal.suffix == ".json" and literal.exists():
        return literal

    # Normalize: allow "gemma-2-9b/reasoning_pack" or "gemma-2-9b/reasoning_pack.json"
    if not name.endswith(".json"):
        name = name + ".json"

    # Case 1: package-bundled catalogs (post-install)
    try:
        pkg_root = files("mechreward")
        # Walk up to project root: mechreward/__init__.py → ../../../catalogs/
        # Not reliable once installed, so fall through gracefully.
        _ = pkg_root
    except Exception:
        pass

    # Case 2: source layout
    here = Path(__file__).resolve()
    project_root = here.parents[3]  # src/mechreward/features/catalog.py → project
    candidate = project_root / "catalogs" / name
    if candidate.exists():
        return candidate

    # Case 2b: installed layout — catalogs shipped inside the package
    pkg_catalogs = here.parents[1] / "catalogs" / name
    if pkg_catalogs.exists():
        return pkg_catalogs

    raise FileNotFoundError(
        f"Could not resolve feature pack '{name}'. Searched: "
        f"{candidate}, {pkg_catalogs}."
    )


def load_pack(name: str) -> FeaturePack:
    """Load a feature pack by name.

    Args:
        name: Either a short name like ``"gemma-2-9b/reasoning_pack"`` or a
            direct path to a .json file.

    Returns:
        A populated ``FeaturePack``.

    Raises:
        FileNotFoundError: If the pack cannot be located.
        json.JSONDecodeError: If the file exists but is malformed.
    """
    path = _resolve_pack_path(name)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    pack = FeaturePack.from_dict(data)

    if not pack.features:
        warnings.warn(f"Feature pack '{name}' has zero features.", stacklevel=2)
    return pack


def save_pack(pack: FeaturePack, path: str | Path) -> None:
    """Save a feature pack to disk as JSON (pretty-printed)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(pack.to_dict(), f, indent=2, ensure_ascii=False)
