# Contributing to mechreward

Thanks for your interest. This is an alpha research project — expect rapid breakage, expect strong opinions in code review, expect reviewers to ask for empirical evidence.

## Scope

We welcome contributions in:

- **New feature packs** for new models (follow the Gemma-2-9B template in `catalogs/`)
- **New hacking attacks** for the adversarial suite
- **New aggregation / normalization strategies**
- **Integration adapters** for more RL frameworks (Axolotl, trlx, LlamaFactory, etc.)
- **Reproductions** of published methods for the comparison suite
- **Documentation and examples**
- **Bug fixes** — obviously

Out of scope for this repo (but welcome as separate projects):

- Training new SAEs at scale (use `sae_lens`)
- Fine-tuning model weights directly via activation editing (use `ActivationAddition`, `RepE`)
- Generic RLHF infrastructure (use `trl`, `OpenRLHF`, `verl`)

## Development setup

```bash
git clone https://github.com/caiovicentino/mechreward
cd mechreward
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,sae,trl]"
pre-commit install
```

## Running the test suite

```bash
pytest
```

The default suite runs in under 60 seconds on CPU. Integration tests that require an SAE and GPU are marked with `@pytest.mark.gpu` and skipped by default.

```bash
pytest -m gpu  # run GPU-dependent tests
```

## Code style

- **ruff** (line length 100, target py310)
- **mypy** (permissive — just catch obvious shape/type errors)
- Docstrings in Google style where they help; not where they don't

```bash
ruff check src tests
ruff format src tests
mypy src
```

## Commit messages

Short, specific, first-person-plural is fine. Examples:
- `Add confident_assertion feature to Gemma-2-9B reasoning pack`
- `Fix off-by-one in LastKAggregation when T < k`
- `Reproduce SARM training loop (exp04)`

## Pull requests

1. Open an issue first for anything beyond a 10-line fix — we want to agree on direction before reviewing code.
2. Add tests for new code paths. "It runs on my machine" is not a test.
3. For reward-related changes, add at least one entry to the adversarial suite that exercises the new behavior.
4. For catalog additions, include the validation AUC numbers in the PR description.
5. Keep PRs focused. One conceptual change per PR.

## Research contributions

If you're submitting results (e.g. "I trained Gemma-2-9B with mechreward and got X on GSM8K"), please include:

- Exact command line used
- Commit hash you ran against
- Wall-clock time and approximate cost
- Eval methodology (seed, temperature, sampling)
- A JSON dump of raw numbers, not just screenshots

## Citations

If you publish work using mechreward, cite the repo (see README). If your work builds on specific prior art (SARM, CRL, etc.), cite those too — we take academic credit seriously here.
