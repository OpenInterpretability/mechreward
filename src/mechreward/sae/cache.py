"""Local filesystem cache for downloaded SAEs, keyed by (release, sae_id)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def cache_dir() -> Path:
    """Return the mechreward cache directory, creating it if needed."""
    env_path = os.environ.get("MECHREWARD_CACHE_DIR")
    if env_path:
        path = Path(env_path)
    else:
        path = Path.home() / ".cache" / "mechreward"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sae_cache_key(release: str, sae_id: str) -> str:
    """Stable filesystem-safe identifier for a (release, sae_id) pair."""
    payload = f"{release}::{sae_id}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    safe_release = release.replace("/", "_")
    return f"{safe_release}__{digest}"


def ensure_cached(release: str, sae_id: str) -> Path:
    """Return the cache subdirectory for a SAE, creating it if absent."""
    sub = cache_dir() / "sae" / sae_cache_key(release, sae_id)
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def clear_cache() -> None:
    """Remove the entire SAE cache. Intended for tests and manual maintenance."""
    import shutil

    target = cache_dir() / "sae"
    if target.exists():
        shutil.rmtree(target)
