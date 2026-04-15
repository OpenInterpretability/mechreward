"""SAE loading, caching, and batched encoding utilities."""

from mechreward.sae.batched_encode import batched_encode
from mechreward.sae.cache import cache_dir, ensure_cached
from mechreward.sae.loader import SAEHandle, load_sae

__all__ = ["SAEHandle", "load_sae", "cache_dir", "ensure_cached", "batched_encode"]
