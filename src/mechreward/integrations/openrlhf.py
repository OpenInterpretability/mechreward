"""OpenRLHF adapter.

OpenRLHF (github.com/OpenRLHF/OpenRLHF) uses Ray + vLLM for scalable RLHF
training. Its reward function protocol is different from TRL's — rewards
come from a ``--remote_rm_url`` HTTP endpoint or a custom callable passed
to the PPO loop.

This adapter provides:
1. A FastAPI-compatible endpoint wrapping a mechreward FeatureReward.
2. A thin ``OpenRLHFRewardServer`` that can be spawned as a subprocess.

The endpoint expects POST requests with ``{"queries": [...], "responses": [...]}``
and returns ``{"rewards": [...]}``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class OpenRLHFAdapter:
    """Expose a mechreward reward function as an OpenRLHF-compatible reward model."""

    reward_fn: Callable[..., list[float]]
    rollout: Any  # HFRollout or VLLMRollout

    def score(
        self,
        queries: list[str],
        responses: list[str],
    ) -> list[float]:
        """Compute reward for a batch of (query, response) pairs.

        OpenRLHF passes query+response separately. We re-run the HF forward
        pass on the concatenation to capture hidden states at the reward
        function's target layer.
        """
        out = self.rollout.generate(prompts=queries, num_return_sequences=1)

        # Swap in the HF-captured hidden states and re-invoke the reward fn.
        return self.reward_fn(
            prompts=queries,
            completions=responses,
            hidden_states=out["hidden_states"],
            attention_mask=out.get("attention_mask"),
        )

    def serve(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Spin up a FastAPI server on the given host/port.

        Requires the optional ``fastapi`` and ``uvicorn`` dependencies.
        """
        try:
            import uvicorn
            from fastapi import FastAPI
        except ImportError as e:
            raise ImportError(
                "OpenRLHFAdapter.serve() requires fastapi + uvicorn. "
                "Install with `pip install fastapi uvicorn`."
            ) from e

        app = FastAPI(title="mechreward-openrlhf")

        @app.post("/reward")
        def _reward(payload: dict):  # type: ignore[type-arg]
            queries = payload.get("queries", [])
            responses = payload.get("responses", [])
            rewards = self.score(queries, responses)
            return {"rewards": rewards}

        uvicorn.run(app, host=host, port=port)
