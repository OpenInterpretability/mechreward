"""Minimal client for the Neuronpedia public API.

Neuronpedia (https://neuronpedia.org) hosts interactive visualizations and
feature-search APIs for published SAEs. We use it for two things:

1. Looking up human-readable descriptions of a feature by id.
2. Finding features that light up strongly on a given prompt ("search").

The API is rate-limited; respect their server by caching results locally.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from mechreward.sae.cache import cache_dir

NEURONPEDIA_BASE = "https://www.neuronpedia.org/api"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 2


@dataclass
class NeuronpediaFeature:
    """A feature record as returned by Neuronpedia."""

    model_id: str
    layer: str  # Neuronpedia's "layer" is actually sae_id, e.g. "22-gemmascope-res-16k"
    index: int  # feature id within the SAE
    description: str
    max_activation: float
    positive_examples: list[str]
    raw: dict[str, Any]

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> NeuronpediaFeature:
        # Neuronpedia returns a "frac_nonzero", "freq" etc. We keep the raw dict.
        desc = ""
        explanations = d.get("explanations") or []
        if explanations:
            desc = explanations[0].get("description", "") or ""
        activations = d.get("activations") or []
        pos_examples = []
        for a in activations[:5]:
            tokens = a.get("tokens") or []
            if tokens:
                pos_examples.append("".join(tokens))
        return cls(
            model_id=d.get("modelId", ""),
            layer=d.get("layer", ""),
            index=int(d.get("index", 0)),
            description=desc,
            max_activation=float(d.get("maxActApprox", 0.0)),
            positive_examples=pos_examples,
            raw=d,
        )


class NeuronpediaClient:
    """Tiny client with local caching.

    Usage:
        >>> client = NeuronpediaClient()
        >>> feat = client.get_feature("gemma-2-9b", "22-gemmascope-res-16k", 2817)
        >>> print(feat.description)
    """

    def __init__(
        self,
        base_url: str = NEURONPEDIA_BASE,
        api_key: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        cache_enabled: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        self._cache_root = cache_dir() / "neuronpedia"
        self._cache_root.mkdir(parents=True, exist_ok=True)

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json", "User-Agent": "mechreward/0.1"}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self._cache_root / f"{safe}.json"

    def _cached_get(self, key: str, url: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if self.cache_enabled and path.exists():
            with path.open() as f:
                return json.load(f)

        last_err: Exception | None = None
        for attempt in range(DEFAULT_RETRIES + 1):
            try:
                resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code >= 400:
                    return None
                data = resp.json()
                if self.cache_enabled:
                    with path.open("w") as f:
                        json.dump(data, f)
                return data
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        if last_err:
            return None
        return None

    def get_feature(
        self,
        model_id: str,
        sae_id: str,
        feature_index: int,
    ) -> NeuronpediaFeature | None:
        """Fetch a single feature's metadata."""
        url = f"{self.base_url}/feature/{model_id}/{sae_id}/{feature_index}"
        key = f"feature::{model_id}::{sae_id}::{feature_index}"
        data = self._cached_get(key, url)
        if data is None:
            return None
        return NeuronpediaFeature.from_api(data)

    def search(
        self,
        model_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[NeuronpediaFeature]:
        """Search for features whose descriptions match a query string."""
        url = f"{self.base_url}/explanation/search"
        key = f"search::{model_id}::{query}::{top_k}"
        path = self._cache_path(key)
        if self.cache_enabled and path.exists():
            with path.open() as f:
                data = json.load(f)
        else:
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    json={
                        "modelId": model_id,
                        "query": query,
                        "numResults": top_k,
                    },
                    timeout=self.timeout,
                )
                if resp.status_code >= 400:
                    return []
                data = resp.json()
                if self.cache_enabled:
                    with path.open("w") as f:
                        json.dump(data, f)
            except Exception:  # noqa: BLE001
                return []

        results = data.get("results") or []
        out: list[NeuronpediaFeature] = []
        for r in results[:top_k]:
            try:
                out.append(NeuronpediaFeature.from_api(r))
            except Exception:  # noqa: BLE001
                continue
        return out
