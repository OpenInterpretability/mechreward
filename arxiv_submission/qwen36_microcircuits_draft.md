# Feature-Level Circuits in Hybrid MoE+GDN Reasoning Models: A Multi-Substrate Negative Study with Methodological Post-mortem

**Caio Vicentino**
*Independent researcher*

## Abstract

We attempt Sparse Autoencoder (SAE)-based mechanistic-interpretability analysis of Qwen3.6-35B-A3B, a hybrid Mixture-of-Experts + Gated Delta Network reasoning model, across **four distinct computational substrates**: the residual stream, the MoE sub-block output, the Gated-Attention sub-block output, and the Gated-Delta-Network sub-block output at layers L11/L17/L23. Two consistent negative findings emerge: **(i)** cross-layer feature correlations range from effectively zero at the residual stream (max |Pearson| = 0.009 over 3×10⁶ pairs) to 0.70–0.77 at sub-block outputs, but single-feature causal ablation systematically contradicts correlation-implied circuits (Pearson(corr, ATE) = −0.22, mean |ATE| = 0.0005 at MoE); **(ii)** sign-consistency filtering (|ATE| + sign consistency ≥ 0.80) recovers a small set of mono-directional causal edges at 100× smaller magnitude than reported for dense transformers (|ATE| 0.003–0.006), and their semantic characterization identifies **task-format circuits** (chat boundaries, option-letter detection, math-content mode switching) rather than reasoning. A proposed rescue — selecting correctness-conditional features from per-prompt fire rates — appeared promising at AUROC 0.72 with n=100 but **failed to replicate** under proper train/test protocol at n=124 (AUROC 0.55 ± 0.05, bootstrap 95% CI including chance). We document this selection-bias mechanism as a methodological warning: SAE feature studies that pre-select features from the same corpus used for evaluation systematically over-estimate predictive power by ~15 pp in this setting. All SAEs, correlation matrices, ATE measurements, ablation controls, and failed-predictor data are released at `caiovicentino1/qwen36-feature-circuits`. Total experimental compute: ~$50.

## 1. Introduction

Mechanistic interpretability of large language models has been built around dense Transformer architectures. Sparse Feature Circuits (Marks et al., arXiv:2403.19647), Cross-Layer Transcoders (Ameisen et al., 2025), and Crosscoders (Lindsey et al., 2024) all assume a single residual stream mediated by standard attention and MLP sub-blocks. State-of-the-art open models increasingly adopt hybrid architectures: routed Mixture-of-Experts combined with Gated Delta Network recurrent sub-blocks. Whether SAE-based methods transfer is an open empirical question.

Qwen3.6-35B-A3B combines 40 decoder layers — 10 using Gated Attention (at L3, L7, L11, L15, L19, L23, L27, L31, L35, L39), 30 using Gated Delta Networks — and replaces every feed-forward block with a 256-expert MoE (9 active per token). No published mechanistic-interpretability study has analyzed this architecture at the SAE-feature level.

We present three connected findings from an honest multi-substrate attack:

**Contribution 1 (negative).** Cross-layer SAE feature correlations in Qwen3.6-35B-A3B range from effectively zero at the residual stream to 0.7 at sub-block outputs. This pattern is invariant across MoE, GA, and GDN sub-blocks. Despite high correlations, single-feature causal ablation yields mean |ATE| ≈ 0.0005, with Pearson(corr, ATE) = −0.22 — correlations do not imply causation in this architecture.

**Contribution 2 (positive but weak).** Sign-consistency filtering of causal ATE recovers small mono-directional edges at magnitudes 100× below those typically reported for dense Transformers. Semantic characterization of their hub features identifies **task-format circuits** (chat-template boundaries, option-letter recognition, math-content mode switching) — not reasoning. This pattern persists across all three sub-block substrates.

**Contribution 3 (methodological).** Correctness-conditional feature selection, initially promising at AUROC 0.72 on 100 prompts, collapses to 0.55 ± 0.05 under proper train/test protocol on 124 prompts. We document the selection-bias mechanism and offer a warning: feature studies that select features from the same corpus used for evaluation systematically over-estimate predictive power by ~15 pp in this setting.

We release 9 SAEs (three substrates × three layers), correlation matrices, ATE measurements, ablation controls, and the replication of the failed correctness-predictor. All experiments run on Google Colab RTX PRO 6000 Blackwell (96 GB) in ~4 hours wall-clock at ~$50.

## 2. Related Work

**SAE-based circuit discovery.** Marks et al. (2024) introduced Sparse Feature Circuits on dense Gemma/Pythia residual SAEs. Ameisen et al. (2025) scaled to Cross-Layer Transcoders on Claude 3.5 Haiku. Rajamanoharan et al. (2024) proposed JumpReLU SAE to address dead features. All evaluations target dense decoder-only Transformers.

**Hybrid architecture interpretability.** We find zero published mechanistic SAE work on Mamba/GDN state, RWKV, or hybrid MoE+GDN as of April 2026. The closest is Ali et al. (2024) on "Hidden Attention" in Mamba; no feature-level follow-up.

**Correctness probes on residual streams.** Marks & Tegmark (2023) and subsequent work demonstrated that simple linear probes on residual stream predict truthfulness and correctness. Our prior `qwen-honest` project achieves AUROC 0.782 via a 192-dim PCA + logistic probe on raw L11 residual on the same base model. Here we test whether interpretable SAE features can approach similar predictive power — they cannot, under proper protocol.

**Causal validation of circuits.** Syed et al. (2023) introduced Edge Attribution Patching; Hanna et al. (2024) characterized faithfulness of attribution patching. We use exact single-feature ablation (subtract `z_i · W_dec[i]` from sub-block output) as the strictest causal test.

**Feature-selection selection bias.** The selection-bias mechanism we document (pre-selecting features on the same corpus used for evaluation) is a well-known problem in genomic studies (Simon et al., 2003; Ambroise & McLachlan, 2002). We document its specific magnitude in the SAE-interpretability context.

## 3. Methods

### 3.1 Model and corpus

All experiments target `Qwen/Qwen3.6-35B-A3B`. We use the MCR Stage B corpus: 624 sampled rollouts on SuperGPQA 10-option multiple-choice questions, filtered to response length ≥ 200 tokens, balanced 50/50 on correctness. For correctness-conditional analysis we re-forward 124 Stage B prompts (50% correct / 50% wrong, stratified).

### 3.2 SAE training

For each target substrate and layer, we train a TopK SAE:

- Architecture: `z = TopK(ReLU((x − b_dec) W_enc + b_enc))`, output `z W_dec + b_dec`
- Size: n_features = 4096 (2× expansion), d_model = 2048, k = 32
- Training: Adam lr = 3e-4, batch 4096, 25 epochs, MSE reconstruction
- Stabilizers: decoder row-norm projection every step; dead-feature revival every 5 epochs

Variance explained ranges: residual SAEs 77–82%, MoE sub-block 79–82%, GA/GDN sub-block 57–74%. The L17 GDN sub-block is hardest to compress (57% at n=4096) — consistent with richer recurrent structure.

### 3.3 Cross-layer correlation (Phase A/B/D)

For each pair (L_A, L_B) ∈ {(11,17), (17,23), (11,23)} and each pair of healthy features (f_i at L_A, f_j at L_B), we compute fire→pre-activation Pearson correlation: between the binary fire of f_i and the continuous pre-activation of f_j. Features are filtered by healthy mask (fire rate ∈ [0.5%, 30%]) and minimum support (≥ 100 firing tokens).

### 3.4 Causal ATE (Phase C)

For each top edge (f_i → f_j), we ablate f_i by subtracting `z_i · W_dec_A[i]` from the L_A sub-block output and measure `pre_j` at L_B. ATE is the mean signed delta over firing tokens; |ATE| is the mean absolute delta; sign consistency is the fraction of firing tokens where delta sign equals the mean sign.

### 3.5 Controls (Phase C)

**Full-MoE ablation**, **full-attention ablation**, **adjacent-layer propagation** (L11→L12 vs L11→L23), and **SVD of intervention deltas** — documented in supplementary data at `phase_c/control_summary.json`.

### 3.6 Correctness-conditional attempt (Phase D) and its failure

For 100 Stage B prompts with correctness labels, we:
1. Forward each prompt, encode attention/GDN sub-block output at L11/L17/L23 through trained SAEs
2. Compute per-prompt feature fire rate
3. Rank features by |Δfire_rate(correct) − fire_rate(wrong)|
4. Train top-10 feature logistic regression per layer, report 5-fold CV AUROC

*This initial protocol is selection-biased*: features are chosen using the same labels against which the classifier is evaluated. Even with CV inside the classifier training, the *feature selection step* has already touched all labels. We therefore replicate on 124 new prompts using a stratified 5-fold train/test split, selecting top-10 features on each training fold only, and evaluating on held-out test folds.

## 4. Results

### 4.1 Cross-layer correlation null at residual, moderate at sub-block outputs (negative)

Training TopK SAEs (n = 4096, k = 32) on three substrates:

| substrate | max \|corr\| L11→L17 | max \|corr\| L17→L23 | max \|corr\| L11→L23 | frac > 0.1 |
|---|---|---|---|---|
| Residual stream | 0.009 | 0.009 | 0.009 | 0.0% |
| MoE sub-block | 0.71 | 0.62 | 0.62 | 2.9–5.3% |
| GA/GDN sub-block | 0.76 | 0.70 | 0.77 | 15–23% |

The residual substrate shows no cross-layer feature dependencies. Sub-block substrates show moderate correlations with hub-like structure. Initial impression: sub-block circuits exist.

### 4.2 Correlations are causally empty (negative)

Single-feature ablation of the top-30 MoE-substrate edges (union of top-10 per pair):

- mean signed ATE = +0.0001
- mean |ATE| = 0.0005
- Pearson(corr, ATE) = **−0.22**

A 0.71 correlation dissolves to noise under causal intervention. We interpret this as input confounding: the same tokens activate features at L_A and L_B via independent paths; removing f_i's contribution leaves the shared-input signal still propagating through attention/MLP transformations.

### 4.3 Sign-consistent microcircuits at 100× smaller magnitude than dense (weakly positive)

Filtering by sign consistency ≥ 0.80 on per-token signed deltas:

| edge | |ATE| | signed | sign consistency | n_tok |
|---|---|---|---|---|
| L17→L23 f1084→f3540 | 0.005 | −0.005 | **1.00** | 10 |
| L17→L23 f1084→f4014 | 0.005 | +0.005 | **1.00** | 10 |
| L17→L23 f1137→f2308 | 0.006 | −0.005 | 0.88 | 40 |
| L17→L23 f1137→f2003 | 0.005 | −0.004 | 0.85 | 40 |
| L17→L23 f1137→f3697 | 0.005 | −0.005 | 0.88 | 40 |
| L17→L23 f3582→f1398 | 0.005 | −0.003 | 0.81 | 31 |
| L11→L17 f4007→f852 | 0.004 | +0.004 | 0.90 | 10 |

Three hub motifs emerge. Semantic characterization over 20 held-out prompts identifies them as **task-format circuits**:

- **f1137 (L17)** fires strongest on the `"G"` option letter (z=0.25, highly consistent), silencing three L23 features that fire on math-unit tokens (`"~"`, `mathrm`, `"J"`, `"K"`) with sign consistency 0.85–0.88. Interpretable as: when processing the letter "G" in option lists, suppress the math-unit representation that dominates answer-content tokens.
- **f1084 (L17)** fires on `" multiple"` (task-format word "multiple-choice"); its antagonistic target pair f3540/f4014 (sign consistency 1.00) appear to encode math-mode vs. content-mode representations respectively — interpretable as a task-format-driven mode switch.
- **f4007 (L11)** fires on `<|im_start|>` (chat template marker); its causal effect on L17 f852 (sign consistency 0.90) appears to encode "message boundary" propagation.

No sign-consistent hub fires on reasoning-specific vocabulary (math operators, logical connectives, semantic-content words). We tested three substrates (MoE sub-block output, Gated-Attention output at L11/L23, Gated-Delta-Network output at L17); the format-circuit pattern is invariant across substrates.

### 4.4 Failed correctness-conditional predictor (post-mortem)

On 100 prompts, ranking top-10 features by |Δfire_rate(correct) − fire_rate(wrong)| and training a 5-fold CV logistic regression gave AUROC 0.715 for L11, 0.605 for L17, 0.630 for L23. The top-10 L11 features appeared to encode technical vocabulary (math variables, domain terminology); the narrative suggested a deployable correctness monitor.

Under proper train/test protocol on a re-sampled set of 124 prompts (stratified 5-fold split, features selected on train fold only):

| layer | selection-biased AUROC (n=100) | proper AUROC (n=124) | delta |
|---|---|---|---|
| L11 | 0.715 | **0.552 ± 0.051** | −0.163 |
| L17 | 0.605 | 0.565 ± 0.019 | −0.040 |
| L23 | 0.630 | **0.597 ± 0.040** | −0.033 |

Bootstrap 95% CI for L11 proper AUROC: [0.396, 0.685] — includes chance. The "early-layer dominance" claim (L11 > L17+L23 combined at n=100) also disappears: L23 becomes the best single layer under proper protocol.

**Root cause**: top-10 features were chosen using the same labels against which the classifier was evaluated. Cross-validation folds were applied inside classifier training, but not to feature selection. Features that happened to correlate with correctness on the specific 100 prompts were selected; on a new 124 prompts, the same features show only weak correlation.

**Magnitude of the bias**: ~15 pp inflation of AUROC in this setting. This is meaningful in a space where published predictive-feature claims often hover in the 0.65–0.80 range.

### 4.5 Full-sub-block ablation has substantial but distributed effect

Zeroing the entire L11 MoE output and measuring downstream MoE output delta:

| target | max per-token |Δ| | relative norm |
|---|---|---|
| L12 (adjacent) | 0.26 | 59% |
| L17 (+6 layers) | 0.50 | 43% |
| L23 (+12 layers) | 0.92 | 53% |

The intervention has a real ~50% relative-norm downstream effect that amplifies with depth. At L23, 89% of SAE features shift by > 0.05 at some token, but top-10 carry only 1.1% of total delta. The causal effect exists but spreads over hundreds of features — single-feature ablation cannot recover it.

### 4.6 Intervention subspace is medium-rank

SVD of stacked intervention deltas (4792 × 2048):

| layer | top-1 var | dim @ 90% var | dim @ 99% var |
|---|---|---|---|
| L17 | 3.2% | 396 | 1089 |
| L23 | 5.2% | 550 | 1307 |

Intervention signal lives in ~20% of residual dimensions — a 5× compression versus Gaussian noise, but not a sparse (<50 dim) basis. TopK SAE with k=32 cannot naturally represent this.

## 5. Discussion

### 5.1 Why correlation-based SAE circuit discovery misleads in hybrid MoE+GDN

Three mechanisms:

1. **Skip-dominance with MoE norm attenuation.** MoE sub-block outputs have Frobenius norm 0.3–0.5 against a much larger residual stream. Single-feature contribution is bounded at 100× smaller scale than in dense architectures.

2. **GDN recurrent distribution.** Gated Delta Networks propagate perturbations through a linear recurrent scan, spreading single-feature effects across token positions and dimensions in a non-local way.

3. **Input confounding.** Token content itself activates features at L_A and L_B through independent paths; cross-layer correlation captures shared-input structure, not causal computation. 6 decoder layers between L_A and L_B preserve input context strongly enough that most sub-block feature co-firing at L_B is explained by input tokens, not by L_A's computed features.

### 5.2 What the surviving microcircuits tell us

The sign-consistent edges (f1137, f1084, f4007, f3582) are real, causally verified, and interpretable — but they encode task-format metadata, not reasoning. This has a methodological consequence: sparse-feature circuits may exist in this architecture only at scales orders of magnitude smaller than in dense, and they may encode aspects of computation (format recognition, mode switching) rather than what interpretability researchers typically care about (mathematical manipulation, logical inference).

### 5.3 Selection bias in SAE feature-correctness studies

The correctness-predictor experiment illustrates an easy-to-make mistake: cross-validation inside classifier training does not correct feature selection that has already touched all labels. In our n=100 setup, this inflated AUROC by ~15 pp; the "positive finding" was an artifact of sample-specific feature selection.

Recommendations:
1. Split corpus into *selection*, *training*, and *evaluation* partitions before looking at labels.
2. Report AUROC under clean protocol as primary metric; biased AUROC is at best illustrative.
3. Treat feature-predictor studies with n < 200 as exploratory.

### 5.4 Where reasoning might live in hybrid MoE+GDN

We tested three sub-block output substrates and the residual stream; none yield sparse reasoning-specific features detectable via standard TopK SAE + correlation + single-feature ablation. Remaining candidates:

- **GDN recurrent hidden state** (dim `n_heads × d_state`, not d_model). The scan carries information through per-token state updates; features may live there rather than at the block output.
- **MoE router logits** (256-dim per token). Expert selection patterns across layers might encode reasoning.
- **Medium-rank residual subspace** (~400-dim, per Section 4.6). Not accessible to sparse-feature methods but accessible to PCA-based probes.
- **Distributed non-sparse**: possibly reasoning simply does not admit sparse SAE representation in this architecture.

All four are open directions beyond the scope of standard SAE methodology.

## 6. Limitations

- **Single architecture**: all findings on Qwen3.6-35B-A3B. A matched dense baseline (Qwen2.5-32B) would isolate hybrid-specific effects.
- **Single SAE architecture**: TopK with k=32, n=4096. JumpReLU or Gated SAE may change results.
- **Sample size for causal ATE**: 5 prompts (~40–100 firing tokens per edge). Single-edge claims have wide CIs; our claims are at the aggregate level.
- **Correctness-conditional n=124**: larger n (500+) might yet reveal a weak-but-real correctness signal below our detection floor.
- **Dead-feature rate**: residual-stream SAEs produced 75% dead features even with revival.
- **Corpus**: SuperGPQA is dense with domain vocabulary; pure-linguistic reasoning may behave differently.

## 7. Conclusion

We present the first SAE-based mechanistic-interpretability investigation of a hybrid MoE+GDN reasoning model. Four substrates tested, consistent patterns across all: cross-layer correlations are dominated by task-format circuits; single-feature causal ablation yields 100× smaller magnitudes than dense-transformer reports; feature-selection-based correctness prediction fails under clean protocol. The negative result is robust; the methodological post-mortem is, we argue, valuable in its own right. The architecture's reasoning substrate remains an open question.

## Reproducibility

- Repo: `caiovicentino1/qwen36-feature-circuits`
- Notebooks: `qwen36_feature_circuits_phase_a.ipynb`, `qwen36_feature_circuits_phase_b_moe.ipynb`, `qwen36_feature_circuits_phase_c_ate.ipynb`, `qwen36_feature_circuits_phase_d_gdn.ipynb`
- Compute: Google Colab RTX PRO 6000 Blackwell (96 GB), ~4 h total wall-clock
- Stage B corpus: `caiovicentino1/Qwen3.6-35B-A3B-mcr-stage-b`
- All data (SAEs, correlation matrices, ATE, controls, failed-predictor replication) available at the repo

## Author's note

This paper began life with a different title — "Domain-Comprehension Features Predict Reasoning Correctness" — built around the n=100 AUROC 0.72 result in Section 4.4. The result was an artifact of feature-selection bias; honest replication on n=124 gave AUROC 0.55 (95% CI including chance). We chose to report the replication outcome and rewrite rather than submit the biased claim. We document the error mechanism in full (Section 5.3) and hope others will catch similar errors earlier than we did.
