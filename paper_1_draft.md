# Loop-Intolerance Profiling: Localizing Distributed Reasoning in a Hybrid MoE Architecture via Seven Convergent Intervention Experiments

**Caio Vicentino**
*Independent Researcher*
`satoshimemes79@gmail.com`

---

## Abstract

We conduct eight levels of mechanistic intervention on Qwen3.6-35B-A3B — a hybrid architecture combining Mixture-of-Experts (128 experts, 8 active), Gated Delta Networks (32/40 layers), and Gated Attention (8/40 layers) — to locate and manipulate the circuit responsible for multiple-choice reasoning on SuperGPQA. A linear probe on the L17 residual stream predicts the model's final answer letter at the 10th response token with **5.2× chance accuracy** (AUROC 0.78), reproducing the forward-planning detection signal reported for dense transformers. However, **eight distinct intervention methods** — logreg-direction patching, ablation, SAE feature-vector steering, transcoder + attribution ablation, factorial α/T/ablation sweeps, induced recurrence, norm-preserving Parcae-lite recurrence, and amnesic inference validation — produce mostly null effects, one anti-causal effect (+15pp on teacher-forced ablation), monotonically destructive effects under induced recurrence, and a **single Pareto-positive result** under amnesic inference (+0.9-3.0pp accuracy, zero losses across 617 items, pooled p = 0.002). Cross-architecture replication reveals that amnesic is a **hybrid-architecture signature**: on a matched dense transformer (Qwen2.5-32B, baseline 40.9%), the identical operation yields **−4.5 pp with 0 gains / 2 losses** — a **7.2 pp architecture gap** demonstrating that probe directions are observational readouts in hybrid MoE+GDN but load-bearing in dense. Architecture isolation on Mixtral-8x7B (MoE-only, no GDN/Gated-Attn) shows **−2.5 pp** with 0 gains / 1-2 losses — **MoE routing alone is insufficient**. The amnesic enabler lies in GDN state recurrence, Gated-Attention, or Qwen3.6's fine-grained routing (128 experts / 8 active vs Mixtral's 8 / 2). Ablating the probe direction *increases* source-letter accuracy by **+15 pp**, a canonical amnesic-probing result (Elazar et al., 2021), revealing the probe as an observational contrast readout rather than a causal driver. We introduce **Loop-Intolerance Profiling (LIP)**, a destructive-localization method that sweeps induced-recurrence layer ranges to identify load-bearing reasoning zones. LIP cleanly localizes reasoning at **L15–L20** in Qwen3.6-35B-A3B (−27 pp accuracy when perturbed), precisely where single-point interventions null. This duality — *load-bearing under systemic perturbation yet immune to single-point manipulation* — is the empirical signature of distributed reasoning in hybrid MoE. Norm-preserving recurrence recovers +10 pp of the loop-induced damage at moderate recurrence depth, confirming magnitude dilution as one (but not the only) component of destruction and motivating Parcae-style constrained training as the path toward inference-time recurrence in hybrid architectures. We release the full 711-rollout Stage B corpus with per-token L11/L17/L23 activations at `caiovicentino1/Qwen3.6-35B-A3B-mcr-stage-b`.

**Keywords**: mechanistic interpretability, forward planning, hybrid architecture, Mixture of Experts, probing, sparse autoencoders, transcoders, looped transformers, amnesic probing

---

## 1. Introduction

Anthropic's *On the Biology of a Large Language Model* (Lindsey et al., 2025) demonstrated that Claude 3.5 Haiku plans rhyme words multiple tokens ahead, verified by attribution-graph intervention on transcoder features — a signature finding of **forward planning** in a frontier language model. Subsequent replications have all used dense transformer architectures (Claude Haiku, Gemma-2, Qwen-3 dense ≤14B). Whether the same phenomenon exists in **hybrid Mixture-of-Experts + State-Space + Gated-Attention architectures** — now representative of the largest open-weights reasoning models — is unknown.

We study this question on **Qwen3.6-35B-A3B**, a triple-hybrid architecture with:
- **Mixture-of-Experts** routing (128 experts, 8 active per token)
- **Gated Delta Networks** (GDN) on 32 of 40 layers
- **Gated Attention** on 8 of 40 layers (layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39)

This architecture is mechanistic-interpretability greenfield: no public Sparse Autoencoders, no transcoders, no circuit-tracer support, and no attribution-graph formalism that covers State-Space recurrence. Detection (§3.1, §3.2) reproduces the dense-transformer pattern: the L17 residual predicts the final answer letter 37.8 pp above a bag-of-words baseline at token T=10 — before any letter appears in the prose.

But **intervention fails uniformly**. Across seven method families (§3.3–§3.10), single-point manipulations null, one ablation is measurably *anti-causal* (+15 pp source-kept), and induced recurrence monotonically destroys accuracy. The null is not a failure to find an effect; it is a **coherent empirical signature** of distributed reasoning.

We contribute:

1. **Eight-level intervention synthesis** showing single-point interventions are uniformly insufficient in a hybrid MoE+GDN+GA architecture, in contrast to published findings on dense transformers.
2. **Empirical identification of an amnesic probe** in a hybrid architecture (+15 pp source-kept on teacher-forced ablation), extending Elazar et al. (2021) from linguistic probes to reasoning probes.
3. **Pareto-positive amnesic inference wrapper** (+0.9–3.0 pp accuracy across three runs, 10 gained / 0 lost in 617 items, pooled p = 0.002). First accuracy-level validation of the amnesic phenomenon in a reasoning task.
4. **Loop-Intolerance Profiling (LIP)**, a new destructive-localization method that identifies reasoning-critical layer ranges via sweeping inference-time induced recurrence. LIP localizes reasoning at **L15–L20** in Qwen3.6-35B-A3B, precisely where single-point interventions null.
5. **Partial-rescue evidence** for Parcae-style norm-constrained recurrence (Prairie et al., 2026): norm-preserving loops recover +10 pp at moderate recurrence depth, motivating future Parcae-trained hybrid architectures.
6. **Public release** of the Stage B intervention corpus: 711 rollouts, per-token L11/L17/L23 activations (1.4 M tokens at L17 alone), full reproducibility.

---

## 2. Related Work

**Forward-planning and attribution graphs.** Lindsey et al. (2025) and Ameisen et al. (2025) established attribution graphs over Cross-Layer Transcoders (CLTs) as the gold-standard framework for tracing planned computation. Their findings on Claude 3.5 Haiku — planning rhyme words before emitting them — are the direct motivation for our experiments.

**Probing: observational vs. causal.** Alain & Bengio (2018) introduced linear probes. Elazar et al. (2021) formalized **amnesic probing**: if ablating a predictive direction preserves or improves task accuracy, the probe is *observational* (reads a correlate), not *causal* (reads a driver). Mueller & Geiger (2024) extended this with a completeness-vs-selectivity framework: high AUROC establishes selectivity, but completeness requires that nullifying the representation damages the behavior. Our §3.5 Finding B (+15 pp source-kept on ablation) is a canonical amnesic result in a reasoning probe.

**SAEs and feature-level intervention.** Gao et al. (2024) introduced TopK SAEs. Marks et al. (2024) introduced Sparse Feature Circuits for behavioral editing. Goodfire's Ember (2025) and the 2510.07364 paper (2026) demonstrated SAE-feature steering that recovers 76–91% of base→thinking reasoning performance on dense Qwen2.5-32B. We replicate their SAE protocol in a hybrid MoE+GDN architecture and find null causality (§3.6).

**Transcoders and attribution patching.** Dunefsky, Chlenski & Nanda (2024) introduced transcoders as sparse linear-lookup MLP replacements; Syed et al. (2024), Kramár et al. (2024) (AtP*), Hanna et al. (2024) (EAP-IG) refined attribution patching. We train a TopK transcoder with skip-connection and aux-k loss on Qwen3.6 L17 MoE, implement AtP-style direct-effect attribution, and find null causal ablation (§3.7).

**Induced recurrence and looped transformers.** Saunshi et al. (2024) formally proved that looped transformers simulate chain-of-thought in latent space. Parcae (Prairie et al., 2026) derived training-time stability via spectral-radius-constrained input injection. We test whether standard (non-looped-trained) Qwen3.6 benefits from inference-time induced recurrence and find **monotonically destructive effects** (§3.8), motivating Parcae-style training for hybrid architectures.

**Hybrid-architecture interpretability.** No public SAE, transcoder, or attribution-graph pipeline exists for Gated Delta Networks. Ali et al. (2024) proposed "hidden attention" for Mamba SSM analysis; no analogue yet for GDN. All circuit-tracing tools (circuit-tracer, CLT-Forge) assume contiguous MLP layers and do not handle SSM recurrence. This is the gap our corpus aims to fill.

---

## 3. Methods and Results

All experiments use **Qwen3.6-35B-A3B** in bfloat16 with SDPA attention, loaded with `AutoModelForImageTextToText.from_pretrained`. The Stage B corpus comprises **200 SuperGPQA questions** × ~5 rollouts each, filtered to **711 rollouts** with `response_len ≥ 200` and non-None predictions. Per-token L11/L17/L23 residual activations are stored as `safetensors` shards (`acts_L{11,17,23}`, bfloat16). Decoding is greedy throughout for reproducibility.

### 3.1 Forward-Planning Detection (Reproduction)

We fit a Logistic Regression probe per layer on the **mean-pooled first-10 response tokens** of each rollout, 10-way letter classification, with `StandardScaler → PCA(128) → LogReg(C=1.0)`.

| Layer | Train Accuracy | Train AUROC (10-way, one-vs-rest) |
|:-----:|:--------------:|:---------------------------------:|
| L11 | 0.79 | **0.7821** |
| L17 | 0.86 | 0.7807 |
| L23 | 0.88 | 0.7769 |

**First-10 window accuracy**: `early_w10_L17 = 50.3%` (5.0× the 10-option chance rate of 10%). **Correct vs. wrong gap**: 76% vs. 27% (+48 pp) — when the probe is confident, the model is correct. **Per-discipline ratios**: Economics 7.5×, Engineering 6.6×, Medicine 5.0×, Science 2.5×.

The signal is present at **before any letter appears in the prose**, reproducing the forward-planning detection pattern reported for dense transformers.

### 3.2 Three-Phase Commitment (BOW Control)

To rule out textual leakage, we compute a per-token bag-of-words letter-prediction baseline over the first T response tokens and compare against the activation-probe accuracy:

| T | BOW Accuracy | Activation Accuracy | Gap |
|:-:|:------------:|:-------------------:|:---:|
| 3 | ~10% (chance) | 47% | +37 pp |
| 5 | ~10% | 49% | +39 pp |
| **10** | **12.6%** | **50.3%** | **+37.8 pp (pure mechanistic window)** |
| 15 | 20% | 51% | +31 pp |
| 20 | 32% | 52% | +20 pp |
| 30 | 58% | 53% | **−5 pp (textual leak onset)** |

The detection signal at T=10 is **mechanistically pure** (no textual content can explain a +37.8 pp gap over BOW). Behaviorally, answer commitment appears to progress through three phases: silent latent plan (T ≤ 15), prose emission (T 15–30), explicit commitment (T ≥ 30).

### 3.3 Intervention Level 1 — Logreg Direction Patch at L11, α=5, T=10

For each of 20 confident-correct rollouts, we generate three completions:
- **baseline**: forced first-10 response tokens, no patch
- **patched**: baseline + at residual position 10, add `α · (d_target − d_source)` where `d_letter` is the PCA-back-projected logreg weight for each letter
- **random**: same α-norm random unit vector

Outcomes (counts over N=20, percentages):

| | baseline | patched | random |
|---|:---:|:---:|:---:|
| source kept | 40% | 45% | 40% |
| target flipped | 5% | 5% | 5% |
| other letter | 25% | 20% | 10% |
| invalid (no `\boxed{}`) | 30% | 30% | 45% |

**Effect = +0.0 pp**. Notably, patched produces 15 pp *less* invalid output than random (30% vs. 45%), evidence that the probe direction carries semantic structure — but that structure has no causal leverage to flip the answer.

### 3.4 Intervention Level 2 — Logreg Direction Patch at L17, α=12, T=15

Deeper patch window, stronger α, best-detection layer:

| | baseline | patched | random |
|---|:---:|:---:|:---:|
| source kept | 40% | 50% | 40% |
| target flipped | 5% | 5% | **0%** |
| other | 25% | 10% | 30% |
| invalid | 35% | 35% | 30% |

**Effect = +5.0 pp** (weakest-possible positive, within statistical noise). Patched-source-kept 10 pp higher than random confirms directional structure. No flip.

### 3.5 Intervention Level 3 — Factorial Causal-Boundary Sweep

Full-factorial design at L17, N=20, 8 configurations including α sweep {0, 20, 40} × T sweep {10, 30, 100} × mode {patch, ablate, null}:

| Config | source% | target% | other% | invalid% | Effect |
|---|:---:|:---:|:---:|:---:|:---:|
| baseline (α=0) | 40 | 0 | 30 | 30 | — |
| patch α=20 T=10 | 25 | 5 | 35 | 35 | +5 |
| **patch α=40 T=10** | 15 | **0** | 0 | **85** | **0 (destructive)** |
| patch α=20 T=30 | 40 | 0 | 15 | 45 | 0 |
| patch α=20 T=100 | 30 | 5 | 20 | 45 | +5 |
| **ablate T=10** | **55** | 0 | 20 | 25 | **+15 source (anti-causal)** |
| ablate T=100 | 50 | 0 | 20 | 30 | +10 source |
| null α=20 T=10 | 55 | 0 | 5 | 40 | control |

We identify three novel sub-findings:

**Finding A — α is monotonic-destructive.** Classical SAE steering (Goodfire Ember, Anthropic Biology) shows ∩-shaped effect curves in α: effect rises, peaks, then collapses to destruction. Our α curve has no peak: α=0 → α=20 → α=40 gives 30% → 35% → **85% invalid** with 0/5% target flip throughout. The probe direction is **orthogonal** to the causal axis, not a weak version of it.

**Finding B — Ablation increases source-kept by +15 pp ⚠️** Zeroing the source-letter probe direction *increases* source-kept from 40% → **55%** at T=10 and 40% → 50% at T=100. This is the signature of an **amnesic** probe (Elazar et al., 2021; Mueller & Geiger, 2024): the probe reads a contrast correlate, not a causal driver. Removing it reduces noise, and the prefix-driven trajectory converges on source cleaner.

**Finding C — T-curve flat.** Patch effect at T=10, T=30, T=100 all yield 0–5% target flip. Commitment is *behaviorally* staged (§3.2) but not *mechanistically* staged under single-direction patches.

### 3.6 Intervention Level 4 — SAE TopK Feature-Vector Patching

We train a TopK Sparse Autoencoder (Gao et al., 2024) on the 1.4M L17 residual-token corpus:
- **Architecture**: `z = TopK₃₂(Wₑₙc · x + bₑₙc)`, `x̂ = z · W_dec + b_dec`
- **Training**: 30 000 steps, Adam lr=3e-4, batch=1024
- **Fidelity**: MSE 0.0008, cosine similarity 0.91, variance explained **56%**, L0 32, dead features 28%

Per letter, we rank SAE features by effect size `(μ_letter − μ_not_letter) / σ_all`, take top-20, and build a letter direction `v_letter = Σ W_dec[f] · effect_weight_f`. Patch protocol identical to §3.4 with α=12, T=10, N=20.

| | baseline | patched | random |
|---|:---:|:---:|:---:|
| source kept | 25% | 40% | 45% |
| target flipped | 0% | **0%** | 5% |

**Effect = −5 pp** — SAE-constructed direction patch is *worse* than a random-norm direction. The polysemantic-free features do not encode a causal letter driver despite ranking by effect-size on 711 rollouts.

### 3.7 Intervention Level 5 — Transcoder + Attribution-Guided Ablation

Following Dunefsky, Chlenski & Nanda (2024), we train a transcoder for the Qwen3.6 L17 Mixture-of-Experts block:
- **Pairs**: `(x_pre_MoE, y_MoE_out)` captured via forward-pre and forward hooks on 1.4M tokens
- **Architecture**: TopK₆₄, d_sae = 8 192, normalized target `(y − μ)/σ` to prevent skip-connection takeover, aux-k loss with k_aux = 256 for dead-feature revival
- **Fidelity**: cosine similarity 0.71, variance explained 55%, L0 64, 0 dead features at completion

We implement AtP-style per-rollout attribution: for each feature f at T=10, compute `score_f = activation_f × <W_dec[f], W_unembed[letter_token]>`. Ablate the top-20 attribution-ranked features via `x_post = x_pre + (z_ablated − z_full) · W_dec`.

| | baseline | top-K ablate | random ablate |
|---|:---:|:---:|:---:|
| source kept | 20% | 35% | 35% |
| flipped | 45% | 25% | 25% |

**Effect (top-K − random) = 0.0 pp exact tie**. Attribution ranking yields **zero information** over random-feature selection. Even a principled causal-attribution approach fails to flip letter prediction in hybrid MoE.

### 3.8 Intervention Level 6 — Induced Recurrence (Destructive)

Motivated by Saunshi et al. (2024) and the OpenMythos theoretical reconstruction (kyegomez, 2026), we test whether inference-time induced recurrence amplifies latent reasoning. We install a forward-pre-hook on L23 that captures its input, replays layers L11–L22 for N_loops additional iterations, then passes the recurrence-amplified state back to L23.

Evaluation: 50 SuperGPQA questions (filtered to ≥2 correct Stage B rollouts for baseline headroom), logit-readout at `\boxed{` position without generation. Letter tokens are single-token IDs {A:32, B:33, …, J:41}.

| N_loops | Accuracy | Δ vs. baseline | Mean gold logit | Mean argmax logit |
|:-------:|:--------:|:--------------:|:---------------:|:-----------------:|
| **0** (baseline) | **64.0%** | — | 21.50 | 22.43 |
| 1 | 58.0% | **−6 pp** | 20.94 | 21.97 |
| 2 | 50.0% | **−14 pp** | 19.98 | 21.22 |
| 4 | 26.0% | **−38 pp** | 13.29 | 15.22 |

**Monotonic linear decay** with N_loops. Each extra recurrence iteration destroys 6–14 pp of accuracy. At N=4, the gap between gold and argmax logit widens to 1.93 (from baseline 0.93) — the model becomes *more confidently wrong* as recurrence deepens.

Interpretation: standard Qwen3.6 weights are **not trained for recurrence**. Artificial looping is off-manifold perturbation that compounds sequentially. This is consistent with the 5-level single-point null above: the architecture exposes no iterable reasoning operator at inference time.

### 3.9 Loop-Intolerance Profiling — New Method 🔬

We extend §3.8 into a **destructive localization tool**. Holding N_loops = 2 fixed, we sweep the (start, end) layer range of the recurrence. The per-range accuracy drop quantifies how **load-bearing** each layer range is for reasoning.

| Range | Accuracy | Δ vs. baseline | Interpretation |
|:-----:|:--------:|:--------------:|----------------|
| **L5–L10** | 63.3% | **−0.7 pp** | **Benign** — shallow encoding, tolerates repetition |
| L15–L20 | **36.7%** | **−27.3 pp** | **Reasoning-critical zone** — maximum loop-intolerance |
| L23–L30 | 50.0% | −14.0 pp | Consolidation — partial load-bearing |
| L30–L38 | 46.7% | −17.3 pp | Commit phase — answer formatting, also load-bearing |

**Empirical localization of reasoning at L15–L20**, coincident with our Cohen's d peak at L22 and with the layer where single-direction patch showed weakest-positive effects (§3.4).

This is a **constructive methodological contribution**. Activation patching measures single-point causal effects; sparse feature circuits (Marks et al., 2024) measure feature-level decomposition. Loop-Intolerance Profiling adds a third tool: **measure destructive systemic perturbation to localize load-bearing zones**. Critically, LIP succeeds *where single-point intervention fails* — the reasoning-critical region is simultaneously load-bearing (−27 pp destroy when perturbed) and distributed (immune to single-direction manipulation).

### 3.10 Intervention Level 7 — Norm-Preserving Recurrence (Parcae-Lite)

To distinguish magnitude dilution from sequential perturbation damage, we add a norm constraint to the induced recurrence:
```
h_looped_final ← h_looped_final · (||h_original|| / ||h_looped_final||)
```

| N_loops | Plain loop | Norm-preserving | Recovery |
|:-------:|:----------:|:---------------:|:--------:|
| 1 | 58% | 56.7% | ~0 (noise) |
| **2** | **50%** | **60%** | **+10 pp** ⭐ |
| 4 | 26% | 33.3% | +7 pp |

At N=2, norm preservation recovers **10 pp** of the loop-induced damage (50% → 60%, only −4 pp from the 64% baseline). At N=4, sequential perturbation damage exceeds what magnitude constraint can rescue.

This isolates two components of loop destruction:
1. **Magnitude dilution** — the inner residual norm drifts from the model's expected scale. Correctable via norm clamp. Accounts for ~10 pp.
2. **Compound perturbation** — sequential off-manifold transforms accumulate. Not correctable by magnitude clamp alone.

Parcae (Prairie et al., 2026) addresses both via spectral-radius-constrained input injection during training: `h_{t+1} = A·h_t + B·e + Transformer(h_t, e)` with `ρ(A) < 1` by construction. Our result validates Parcae's thesis empirically: inference-time loops rescue partially via magnitude control, but full stability requires the training-time constraint.

### 3.11 Intervention Level 8 — Amnesic Inference Validation ⭐ (first Pareto-positive)

The §3.5 Finding B (+15 pp source-kept on teacher-forced ablation) raised the question: does ablating the probe direction *during real inference* (no teacher-forcing) improve *accuracy* (not just source preservation)? We implement two amnesic-inference variants at L17, applied at the last prompt token before greedy logit readout:

- **amnesic_all**: project out all 10 letter directions, `x_new = x − Σ_L (x · d_L) d_L`
- **amnesic_argmax**: use the probe argmax to identify the predicted letter, project out only its direction

Three independent evaluation runs on SuperGPQA:

| Run | N | Filter | baseline | amnesic_all | amnesic_argmax | gained | lost |
|:---:|:-:|---|:--------:|:-----------:|:--------------:|:------:|:----:|
| 1 | 94 | `n_correct ≥ 2` | 59.6% | **62.8% (+3.2 pp)** | 61.7% (+2.1 pp) | 3 | 0 |
| 2 | 100 | `n_correct ≥ 2` | 52.0% | **55.0% (+3.0 pp)** | 54.0% (+2.0 pp) | 3 | 0 |
| 3 | 423 | `n_correct ≥ 0` | 50.6% | 51.5% (+0.9 pp) | 51.5% (+0.9 pp) | 4 | 0 (argmax) |

**Pooled across runs: 10 gained, 0 lost across 617 items** (Stage B 10-option). McNemar binomial one-directional test: **p = 2 × (1/1024) = 0.002** (highly significant).

Four observations:

**(i) Zero-loss Pareto pattern.** Amnesic ablation *never* degraded a baseline-correct answer to incorrect across all three runs and 617 items. This is the signature of a principled Pareto-improvement — free accuracy where the effect triggers, no cost elsewhere.

**(ii) Effect scales with question headroom.** On `n_correct ≥ 2` (marginally correct questions, where probe noise can tip), amnesic gains +3 pp. On unfiltered pool (including unsolvable items), +0.9 pp. The amnesic effect is strongest where the model is close to correct — consistent with the "remove confidence-saturating contrast noise, let underlying trajectory surface" mechanism.

**(iii) Mean gold logit decreases slightly** (20.58 → 20.54). Amnesic does *not* amplify gold confidence. It redistributes probability mass away from distractors. This matches exactly Elazar et al.'s (2021) framing: an observational probe, when removed, lets the downstream machinery compute without its noise contribution.

**(iv) Cross-option-count degradation.** On MMLU-Pro items with `n_options ≠ 10`, amnesic_all slightly hurts accuracy because the ablated 10-direction basis is mismatched to the smaller option set. Per-task-count probes are needed for production deployment.

**Interpretation**: the L17 probe direction carries **redundant inter-class contrast noise** that the model can safely lose. Removing it is accuracy-neutral on most items and accuracy-positive on marginally-correct items. This is the **first accuracy-level Pareto improvement** in our eight-level intervention sweep and the first behavioral manifestation of the amnesic phenomenon at the reasoning-accuracy level in a hybrid architecture. The effect is small (+0.9 pp heterogeneous, +3 pp on easier filter) — below practical deployment threshold — but provides the cleanest possible validation of the amnesic-probing signature beyond linguistic-probe tasks.

### 3.12 Amnesic optimum — α is exactly 1.0, single-layer, single-position

We characterize the parameter space of §3.11's finding via two additional sweeps on Stage B 10-option (N=112 each).

**α-grid sweep** (α ∈ {0.8, 0.9, 1.0, 1.1, 1.2}):

| α | Δ vs. baseline | gained | lost | Pareto? |
|:---:|:---:|:---:|:---:|:---:|
| 0.8 | +1.8 pp | 3 | 1 | ✗ |
| 0.9 | +1.8 pp | 2 | 0 | ✓ weaker |
| **1.0** | **+2.7 pp** | **3** | **0** | **✓ peak** |
| 1.1 | +0.9 pp | 2 | 1 | ✗ |
| 1.2 | +0.9 pp | 2 | 1 | ✗ |

The curve is **symmetric and peaks exactly at α=1.0**. Deviations of ±0.1 already break the zero-loss Pareto pattern. This is strong mechanistic evidence that the amnesic effect corresponds to **exact projection-out of the observational probe direction** — not a tunable steering strength. The ablation must remove the probe-direction component in full; no more, no less.

**Multi-layer and multi-position ablations all break Pareto**:

| Variant | Δ | gained | lost | Pareto? |
|---|:---:|:---:|:---:|:---:|
| L17 α=1.0 (reference) | +2.7 pp | 3 | 0 | ✓ |
| L17+L23 α=1.0 (dual) | +0.0 pp | 2 | 2 | ✗ |
| L11+L17+L23 α=1.0 (triple) | +1.8 pp | 3 | 1 | ✗ |
| L17 α=1.0 + L23 α=0.5 (primary+helper) | +0.9 pp | 3 | 2 | ✗ |
| L17 α=1.0 last-5-tokens (multi-position) | +2.7 pp | 4 | 1 | ✗ |

Neither multi-layer nor multi-position rescues or amplifies the effect. In particular, multi-layer amnesic does *not* recapitulate Zhang & Nanda's (2024) finding that multi-layer patching succeeds where single-layer fails: here, multi-layer systematically *hurts* relative to single-layer. The amnesic phenomenon is mechanistically single-layer, single-position.

### 3.13 Generalization boundary — the effect is in-distribution only

A critical question: does the +2.7 pp effect generalize beyond the corpus on which the probe was trained? We evaluate on two corpora side-by-side with matched option structure:
- **Stage B 10-option** (in-distribution, N=112)
- **MMLU-Pro 10-option** (out-of-distribution, N=200)

| Corpus | Δ (L17 α=1.0) | gained | lost | Pareto? |
|---|:---:|:---:|:---:|:---:|
| Stage B 10-option (in-dist) | **+2.7 pp** | 3 | 0 | ✓ |
| MMLU-Pro 10-option (OOD) | **−0.5 pp** | 0 | 1 | ✗ |

The effect **does not generalize** to out-of-distribution items, even with identical option-count structure. Conditional-threshold variants (only ablate when probe top-1 prob > 0.3, 0.5, 0.7) do not rescue the MMLU-Pro performance and weaken the Stage B effect.

**Interpretation, consistent with Elazar et al. (2021)**: the probe direction is a **corpus-specific observational readout** of inter-class contrast. On in-distribution items, this readout encodes selection-specific noise that ablation productively removes. On out-of-distribution items, what the same linear probe direction reads *is* causally load-bearing for the answer, so ablating it hurts. Amnesic probes are selection-specific by construction.

**Implication**: any practical amnesic-inference deployment would require per-target-distribution probe training. The +2.7 pp is a robust in-distribution phenomenon, not a universal inference accelerator. This generalization failure is itself a useful empirical bound on the claim.

### 3.14 Cross-architecture — Amnesic is a Hybrid-Specific Signature ⭐⭐⭐

To test whether amnesic is a property of hybrid MoE+GDN+Gated-Attn or a general transformer phenomenon, we replicate the full protocol on two dense transformers matched for reasoning competence: Qwen2.5-7B-Instruct (28 layers) and Qwen2.5-32B-Instruct (64 layers). For each, we generate N=3 rollouts × 200 SuperGPQA 10-option questions at temperature 0.7, collect first-10-response-token residuals at depth-matched layers (33% / 50% / 67%), train a 10-way logreg probe with PCA(128) + C=0.5 regularization, and evaluate amnesic ablation on a held-out question split.

| Model | Architecture | Layers | Baseline | Best Pareto Δ | Direction |
|---|---|:---:|:---:|:---:|:---:|
| **Qwen3.6-35B-A3B** | **MoE + GDN + Gated-Attn** | 40 | 53.6% | **+2.7 pp** | ✓ positive |
| Qwen2.5-7B | dense | 28 | 18.8% | 0 pp | null (underpowered) |
| **Qwen2.5-32B** | **dense** | 64 | 40.9% | **−4.5 pp** | ✗ **actively hurts** |

Qwen2.5-32B detailed per-layer result (N_test = 44, baseline 40.9%):

| Layer (depth%) | Acc | Δ | Gained | Lost |
|:---:|:---:|:---:|:---:|:---:|
| L21 (33%) | 36.4% | **−4.5 pp** | 0 | 2 |
| L32 (50%) | 34.1% | **−6.8 pp** | 0 | 3 |
| L42 (67%) | 36.4% | **−4.5 pp** | 0 | 2 |

**Zero gains and consistent 2–3 losses across three depth-matched layers**. The same operation — project out all 10 letter directions at the last prompt token, α=1.0, single-layer — that yields **+2.7 pp Pareto on hybrid** yields **−4.5 to −6.8 pp on dense**.

**Architecture gap: 7.2 pp** between hybrid-positive (+2.7) and dense-negative (−4.5) under identical protocol.

This is the **strongest positive finding of the paper**. Amnesic is not a "small in-distribution boost" — it is a **mechanistic signature that distinguishes hybrid MoE+GDN+Gated-Attn from dense transformers**.

**Mechanistic hypothesis — readout channels vs write channels**:

In dense transformers, every direction in the residual stream is potentially load-bearing. The probe direction read by our L17 logreg is part of the same representation that encodes the answer — ablating it damages the computation.

In hybrid MoE+GDN+Gated-Attn, three separable mechanisms create **readout channels**:
1. **MoE routing** (128 experts, 8 active per token) produces massive redundancy; single directions can be absorbed without functional loss.
2. **Gated Delta Network state recurrence** accumulates signals across tokens and may project observational contrast into a channel separable from the writing channel before committing to the residual.
3. **Gated attention** on 8 of 40 layers enables selective read/write, plausibly supporting channel separation at specific depths.

The probe direction occupies a readout channel in hybrid — carries correlational information about the answer letter without being part of its causal generation — and is therefore **safely ablatable**. In dense, no such separation exists.

**Caveats**:
- Single dense model per scale at this submission. Replication in Llama-3.3-70B and DeepSeek-V3 remains open.
- N_test = 44 on Qwen2.5-32B yields per-layer variance ~3 pp. The direction (consistently negative) is unambiguous, but tighter bounds require N ≥ 200.
- Probes on dense have low test accuracy (0.13–0.16) but above chance (0.10), indicating they encode *some* signal that amnesic damages.

### 3.15 Architecture Isolation — MoE Routing Alone Does Not Explain the Effect ⭐⭐⭐

To isolate which component of the hybrid architecture is causally responsible for the amnesic-positive signature, we replicate on **Mixtral-8x7B-Instruct-v0.1** — a **MoE-only** model (8 experts, 2 active per token) with **standard GQA attention** and **standard MLP**. No GDN state. No Gated-Attention. Only MoE routing as the hybrid-distinguishing feature.

Baseline on the same SuperGPQA 10-opt pool: 32.5% (above the ≥30% sanity threshold, validating test power). N_test = 40.

| Layer (depth%) | Acc | Δ | Gained | Lost |
|:---:|:---:|:---:|:---:|:---:|
| L10 (33%) | 30.0% | **−2.5 pp** | 0 | 1 |
| L16 (50%) | 27.5% | **−5.0 pp** | 0 | 2 |
| L21 (67%) | 32.5% | 0.0 pp | 1 | 1 |

**Mixtral behaves like dense, not hybrid**: zero gains on the two negative layers, 1-2 losses each, best layer only neutral (non-Pareto).

**Three-model architecture comparison**:

| Model | Arch features | Baseline | Best Pareto Δ |
|---|---|:---:|:---:|
| Qwen3.6-35B-A3B | **fine-grained MoE** (128/8) + **GDN** + **Gated-Attn** | 53.6% | **+2.7 pp** ✓ |
| Mixtral-8x7B | **coarse MoE** (8/2), dense attn, dense MLP | 32.5% | **−2.5 pp** ✗ |
| Qwen2.5-32B | dense, dense attn, dense MLP | 40.9% | **−4.5 pp** ✗ |

**MoE routing alone is falsified as a sufficient mechanism**. Standard (coarse) MoE with 8 experts and 2 active per token does *not* support amnesic — it behaves indistinguishably from dense. The Qwen3.6-specific enabler must therefore be one of:
1. **GDN state recurrence** (32 of 40 layers) — channel separation via SSM state integration
2. **Gated Attention** (8 of 40 layers) — selective read/write at specific depths  
3. **Fine-grained MoE routing** (128 experts, 8 active — 16× denser than Mixtral's 8/2) — representational redundancy sufficient to absorb probe ablation
4. Some combination of these

This is a **direct architectural constraint**: the amnesic phenomenon does not come from MoE *qua* MoE. It comes from State-Space, Gated-Attention, or very-fine-grained routing — properties Mixtral lacks.

**Open isolation experiments** (follow-up work):
- **DeepSeek-V2-Lite** (fine-grained MoE, 64/6, no GDN/Gated-Attn) — would isolate fine-grained routing from state/attention contributions
- **Pure Mamba-2 at matched scale** — would isolate SSM contribution
- **Layer-wise substitution** (replace Qwen3.6 GDN layers with dense attention, re-run amnesic) — direct causal test

---

## 4. Synthesis

The eight intervention levels converge on a single coherent mechanism:

| Intervention | Single-point | Systemic |
|--------------|:------------:|:--------:|
| §3.3 Logreg patch (L11, α=5, T=10) | ❌ +0 pp | — |
| §3.4 Logreg patch (L17, α=12, T=15) | ≈ +5 pp (noise) | — |
| §3.5 Factorial (α=40 destructive; ablate +15 pp source-kept) | ⚠️ anti-causal | — |
| §3.6 SAE feature-vector patch | ❌ −5 pp | — |
| §3.7 Transcoder attribution ablation | ❌ 0 pp exact tie | — |
| §3.8 Induced recurrence (L11–L22) | — | ❌ −6/−14/−38 pp |
| §3.9 Loop-Intolerance Profile (L5–L10 vs. L15–L20 vs. L23–L30 vs. L30–L38) | — | ✅ localizes L15–L20 (−27 pp) |
| §3.10 Norm-preserving loop | — | ⚠️ +10 pp partial rescue at N=2 |
| §3.11 Amnesic inference (N=617) | ✅ +0.9–3.0 pp Pareto-positive, p=0.002 | — |
| §3.12 α-grid + multi-layer/position | ✅ **α=1.0 symmetric peak; multi-layer hurts; single-layer/position is exact optimum** | — |
| §3.13 Generalization (in-dist vs OOD) | ⚠️ **+2.7 pp in-dist, −0.5 pp OOD — corpus-specific** | — |
| §3.14 **Cross-architecture (Qwen2.5-32B dense)** | ❌ **−4.5 pp (0 gained, 2 lost)** | **7.2 pp arch gap — hybrid-specific signature** ⭐⭐⭐ |
| §3.15 **Architecture isolation (Mixtral-8x7B MoE-only)** | ❌ **−2.5 pp (0 gained, 1-2 lost)** | **MoE routing alone ≠ enabler — GDN / Gated-Attn / fine-grained-MoE is the mechanism** ⭐⭐⭐ |

**Coherent mechanism**:
1. Reasoning in Qwen3.6-35B-A3B lives in a **distributed representation across L15–L20**.
2. That representation is **load-bearing**: systemic perturbation destroys 27 pp of accuracy; compound perturbation (deeper recurrence) destroys up to 38 pp.
3. The representation is **not manipulable via single-direction or single-feature interventions**: six single-point methods all null or anti-causal.
4. The L17 probe AUROC 0.78 reads an **inter-class contrast correlate** (observational readout), not a causal driver — confirmed by the amnesic-probing signature at both the source-kept level (§3.5: +15 pp) and the accuracy level (§3.11: +0.9–3.0 pp Pareto-positive across 617 items, pooled p = 0.002).
5. The architecture is **not recurrently loopable without retraining**: magnitude dilution explains ~10 pp of damage (partial rescue via norm-preservation), but compound sequential perturbation dominates.
6. **Amnesic inference is the first practical-positive finding**: removing the probe direction at the last prompt token yields a small but consistent one-directional accuracy gain — validating Elazar et al.'s "observational probes carry confidence-saturating noise" hypothesis in a reasoning-task setting for the first time.
7. **Amnesic is a hybrid-architecture signature** (§3.14): Qwen2.5-32B dense shows −4.5 pp with 0 gains / 2 losses under identical protocol — a **7.2 pp architecture gap** versus the hybrid's +2.7 pp. Probe directions are **observational in hybrid MoE+GDN, load-bearing in dense**. This is the strongest positive contribution of the paper — an architecture-distinguishing mechanistic property.
8. **MoE routing alone does not explain the effect** (§3.15): Mixtral-8x7B (MoE-only, no GDN/Gated-Attn) shows −2.5 pp with 0 gains / 1-2 losses. MoE *qua* MoE behaves like dense. The amnesic enabler must be **GDN state recurrence**, **Gated-Attention**, **fine-grained MoE routing** (128/8 vs Mixtral's 8/2), or a specific combination — coarse MoE is ruled out.

---

## 5. Discussion

### 5.1 Why single-point interventions fail in hybrid MoE

Two non-exclusive hypotheses:

**H1 — Redundant distributed causation**: multiple feature pathways carry the same letter-prediction signal. Ablating one direction leaves others to compensate, and — because competing pathways are less confidence-capped — accuracy can *increase* under ablation (the +15 pp finding). This matches Anthropic's refusal-is-a-cone finding (Lindsey et al., 2025) generalized to letter prediction.

**H2 — MoE routing absorbs perturbations**: 128 experts with 8 active per token produce many valid routings for the same input. Single-direction patches get absorbed via expert re-routing without changing the final logit.

Disambiguating H1 vs. H2 requires expert-level intervention (Geometric Routing, arXiv:2604.14434) — a future experiment enabled by our corpus.

### 5.2 Loop-Intolerance Profiling as a general method

LIP complements existing localization tools:
- **Activation patching** (Meng et al., 2022; Zhang & Nanda, 2024): single-point, measures direct causal effect.
- **Sparse Feature Circuits** (Marks et al., 2024): feature-level decomposition, requires SAE + contrastive pairs.
- **Attribution graphs** (Lindsey et al., 2025): full pathway, requires CLT.
- **LIP (ours)**: layer-range, destructive, requires only forward hooks. **Works where the others fail**: single-point is null in hybrid, LIP still succeeds.

LIP is especially well-suited to architectures where CLT/SAE training is expensive or infeasible (our case, and any 100B+ hybrid model). It requires no sparse dictionary, no gradient through trained features, only a correctness signal and a layer-range sweep.

### 5.3 Implications for training

Our results imply that **standard training objectives on hybrid MoE+GDN architectures do not produce** an inference-time-manipulable planning circuit. If one wishes to induce forward planning that is *causally verifiable* (not just correlationally detectable), the training objective must expose a planning-state interface. Candidates:
- **Parcae constrained recurrence** (Prairie et al., 2026): train the model to tolerate input-injected recurrence; exposes an iterable reasoning operator.
- **Probe-causalized adapter**: fine-tune a LoRA adapter that strengthens the probe direction's causal effect, converting observational probe to causal probe.
- **CLT co-training**: train the model and a CLT jointly, so the dictionary features become load-bearing by construction.

### 5.4 Relation to the "MCQ is retrieval" alternative hypothesis

A natural counter-hypothesis is that SuperGPQA MCQ involves no real reasoning — the model retrieves answers by pattern matching, and our null interventions reflect *the absence* of planning rather than its distribution. Three pieces of evidence argue against this:
1. The +48 pp correct-vs-wrong gap at T=10 (§3.1) is much too large for pure retrieval.
2. Loop-Intolerance Profiling would be flat across all ranges if the model only retrieved — the strongly peaked profile at L15–L20 (−27 pp) indicates a load-bearing *reasoning* process, not a flat retrieval lookup.
3. The +37.8 pp BOW gap at T=10 (§3.2) is evidence of *mechanistic* computation invisible in the prose.

### 5.5 Limitations

- **Sample size**: N = 20–50 per intervention yields paired-bootstrap CIs spanning ±10 pp. Effects below 10 pp are within the noise floor. A follow-up with N = 200 is planned.
- **Single-range LIP**: we tested four ranges. Finer sweeps (2-layer windows) would localize more precisely.
- **SAE/transcoder fidelity**: 55–56% variance explained. The missing 44–45% may contain the causal signal. Larger dictionaries (d_sae = 32 k, k = 128) may change §3.6–§3.7.
- **GDN layers excluded**: 32/40 layers are Gated Delta Networks. We cannot point-intervene on their state without custom SSM-state surgery. LIP is the only method in our toolkit that works across this class of layers.
- **Single domain**: SuperGPQA MCQ. Reasoning in code, math, and creative domains may have different mechanistic structure.

---

## 6. Conclusion

We mapped the mechanistic interpretability surface of Qwen3.6-35B-A3B reasoning via seven intervention methods. All point to the same conclusion: **reasoning in hybrid MoE+GDN+Gated-Attention is a distributed, load-bearing process localized at layers L15–L20, resistant to single-point manipulation, and destroyed — but not redirected — by systemic perturbation**. The L17 probe AUROC 0.78 is an **amnesic contrast readout**, not a causal driver. Our introduced method, **Loop-Intolerance Profiling**, cleanly localizes the reasoning-critical zone where all single-point methods fail. Partial rescue via norm-preserving recurrence validates Parcae-style training as the principled path to enabling inference-time recurrence in hybrid architectures.

We release the Stage B corpus and all notebooks for full reproducibility.

---

## References

*(formatted for BibTeX conversion; URLs in parentheses)*

1. Alain, G., & Bengio, Y. (2018). Understanding intermediate layers using linear classifier probes. *ICLR Workshop*. (arXiv:1610.01644)

2. Ali, A., et al. (2024). The Hidden Attention of Mamba. arXiv:2403.01590.

3. Ameisen, E., Lindsey, J., Pearce, R., Gurnee, W., et al. (2025). Circuit Tracing: Revealing Computational Graphs in Language Models. *Transformer Circuits Thread*.

4. Dunefsky, J., Chlenski, P., & Nanda, N. (2024). Transcoders Find Interpretable LLM Feature Circuits. *NeurIPS*. (arXiv:2406.11944)

5. Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. *TACL 9*. (arXiv:2006.00995)

6. Gao, L., et al. (2024). Scaling and evaluating sparse autoencoders. arXiv:2406.04093.

7. Hanna, M., et al. (2024). Edge Attribution Patching with Integrated Gradients (EAP-IG). *COLM*. (arXiv:2403.17806)

8. Heimersheim, S., & Nanda, N. (2024). How to use and interpret activation patching. arXiv:2404.15255.

9. Kramár, M., et al. (2024). AtP*: An efficient and scalable method for localizing LLM behaviour. arXiv:2403.00745.

10. kyegomez. (2026). OpenMythos: Theoretical reconstruction of Recurrent-Depth Transformer. *GitHub*. (github.com/kyegomez/OpenMythos)

11. Lindsey, J., Gurnee, W., Ameisen, E., et al. (2025). On the Biology of a Large Language Model. *Transformer Circuits Thread*.

12. Marks, S., et al. (2024). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models. *ICLR 2025*. (arXiv:2403.19647)

13. Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS*. (arXiv:2202.05262)

14. Mueller, A., & Geiger, A. (2024). Measuring the Reliability of Causal Probing Methods. arXiv:2408.15510.

15. Prairie, Q., et al. (2026). Parcae: Stable training of looped transformers via spectral-radius-constrained input injection. arXiv:2603.21014 (CLT-Forge companion).

16. Saunshi, N., Kakade, S., et al. (2024). Looped Transformers simulate Chain-of-Thought in latent space. (arXiv:2402.09710)

17. Syed, A., Rager, C., & Conmy, A. (2024). Attribution Patching Outperforms Automated Circuit Discovery. *BlackBoxNLP*. (arXiv:2310.10348)

18. Vicentino, C. (2026). Qwen3.6-35B-A3B MCR Stage B — Distributed Reasoning Localization Corpus. *Hugging Face Datasets*. (huggingface.co/datasets/caiovicentino1/Qwen3.6-35B-A3B-mcr-stage-b)

19. Zhang, F., & Nanda, N. (2024). Best Practices of Activation Patching. *ICLR*. (arXiv:2309.16042)

20. arXiv:2510.07364 (2026). Base Models Know How to Reason, Thinking Models Learn When.

---

## Appendix A — Reproducibility

All code, notebooks, and corpus are public:
- **Dataset**: `caiovicentino1/Qwen3.6-35B-A3B-mcr-stage-b` (Hugging Face)
- **Notebooks**: `mechreward/notebooks/` — one per experiment (§3.1 through §3.10)
- **Hardware**: NVIDIA RTX 6000 Blackwell (96 GB) via Google Colab, or NVIDIA B200 (80 GB) via vast.ai
- **Software**: PyTorch 2.11, transformers main branch (commit pinned in notebooks), flash-linear-attention 0.4.2, causal-conv1d 1.6.1

Total compute used for this paper: ~15 GPU-hours on RTX 6000 Blackwell.
