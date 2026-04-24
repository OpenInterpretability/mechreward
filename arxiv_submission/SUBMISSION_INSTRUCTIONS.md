# arXiv Submission — Step-by-Step

## Package contents

```
arxiv_submission/
├── main.tex              ← LaTeX source (this is THE paper)
├── references.bib        ← BibTeX references
└── SUBMISSION_INSTRUCTIONS.md  ← this file
```

## Local compile (sanity check before upload)

Test locally first. If you have LaTeX installed:

```bash
cd "/Volumes/SSD Major/fish/mechreward/arxiv_submission"
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # twice to resolve cross-refs
open main.pdf
```

If you don't have LaTeX locally, use Overleaf:

1. Go to https://www.overleaf.com/project
2. "New Project" → "Upload Project"
3. Upload `main.tex` + `references.bib`
4. Overleaf auto-compiles → verify PDF looks right
5. If OK, download .tex + .bib for arXiv upload

## arXiv upload

1. Go to https://arxiv.org/submit
2. Sign in (create account if needed)
3. **Start new submission**
4. **License**: `CC BY 4.0` (Creative Commons Attribution)
5. **Primary category**: `cs.LG` (Machine Learning)
6. **Secondary categories**: `cs.CL` (Computation and Language), `cs.AI`
7. **Upload files**:
   - Click "Files" → "Upload"
   - Upload `main.tex` and `references.bib`
   - arXiv compiles automatically
8. **Wait for compile** — check for errors (should be clean)
9. **Verify preview** — check all 15 pages render
10. **Metadata** — arXiv will parse title, abstract, authors from `.tex`
11. **Submit** — paper goes to moderation queue, typically live within 24h

## Recommended arXiv categories

- **Primary**: `cs.LG` (Machine Learning) — standard for interp papers
- **Cross-listed**:
  - `cs.CL` (Computation and Language) — for the reasoning/MCQ context
  - `cs.AI` (Artificial Intelligence) — broader AI relevance

## Abstract to paste (if arXiv asks separately)

```
We conduct nine levels of mechanistic intervention on Qwen3.6-35B-A3B — a hybrid architecture combining Mixture-of-Experts (128 experts, 8 active), Gated Delta Networks (32/40 layers), and Gated Attention (8/40 layers) — to locate and manipulate the circuit responsible for multiple-choice reasoning on SuperGPQA. A linear probe on the L17 residual stream predicts the model's final answer letter at the 10th response token with 5.2x chance accuracy (AUROC 0.78), reproducing the forward-planning detection signal reported for dense transformers. However, eight single-point intervention methods produce null effects, anti-causal effects, or monotonically destructive effects. Ablating the probe direction INCREASES source-letter accuracy by +15pp, a canonical amnesic-probing result (Elazar et al., 2021). We introduce Loop-Intolerance Profiling (LIP), a destructive-localization method that identifies reasoning-critical zones at L15-L20. Amnesic inference validation yields a +2.7pp Pareto-positive accuracy boost (10 gained / 0 lost across 617 items, pooled p=0.002) with an exact alpha=1.0 peak. The effect is sharply hybrid-specific: Qwen2.5-32B (dense) shows -4.5pp, and Mixtral-8x7B (MoE-only) shows -2.5pp — a 7.2pp architecture gap with MoE routing alone falsified as a sufficient enabler. We release the 711-rollout Stage B corpus with per-token L11/L17/L23 activations.
```

## Title (for arXiv metadata)

```
Loop-Intolerance Profiling: Localizing Distributed Reasoning in a Hybrid MoE Architecture via Nine Convergent Intervention Experiments
```

## Authors

```
Caio Vicentino
```

## Comments field (arXiv metadata)

Suggested:
```
15 pages, 9 intervention experiments. Dataset: https://huggingface.co/datasets/caiovicentino1/Qwen3.6-35B-A3B-mcr-stage-b
```

## If arXiv complains about BibTeX

If arXiv doesn't auto-run BibTeX, you can either:

**Option A — Use biblatex** (replace bibliography block):
```latex
\usepackage[backend=biber,style=numeric]{biblatex}
\addbibresource{references.bib}
% ... in body:
\printbibliography
```

**Option B — Inline the bibliography** (replace `\bibliography{references}` with):

```latex
\begin{thebibliography}{99}

\bibitem{alain2018understanding}
G. Alain and Y. Bengio. Understanding intermediate layers using linear classifier probes. ICLR Workshop, 2018. arXiv:1610.01644.

\bibitem{ali2024hidden}
A. Ali, I. Zimerman, L. Wolf. The Hidden Attention of Mamba Models. arXiv:2403.01590, 2024.

\bibitem{ameisen2025circuits}
E. Ameisen, J. Lindsey, A. Pearce, W. Gurnee, et al. Circuit Tracing: Revealing Computational Graphs in Language Models. Transformer Circuits Thread, 2025.

\bibitem{dunefsky2024transcoders}
J. Dunefsky, P. Chlenski, N. Nanda. Transcoders Find Interpretable LLM Feature Circuits. NeurIPS 2024. arXiv:2406.11944.

\bibitem{elazar2021amnesic}
Y. Elazar, S. Ravfogel, A. Jacovi, Y. Goldberg. Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. TACL 9, pp. 160-175, 2021.

\bibitem{gao2024scaling}
L. Gao et al. Scaling and evaluating sparse autoencoders. arXiv:2406.04093, 2024.

\bibitem{hanna2024eapig}
M. Hanna, S. Liu, A. Variengien. Have Faith in Faithfulness. arXiv:2403.17806, 2024.

\bibitem{kramar2024atp}
J. Kramár, T. Lieberum, R. Shah, N. Nanda. AtP*: An efficient and scalable method for localizing LLM behaviour to components. arXiv:2403.00745, 2024.

\bibitem{lindsey2025biology}
J. Lindsey, W. Gurnee, E. Ameisen, et al. On the Biology of a Large Language Model. Transformer Circuits Thread, 2025.

\bibitem{marks2024sparse}
S. Marks, C. Rager, E. Michaud, Y. Belinkov, D. Bau, A. Mueller. Sparse Feature Circuits. ICLR 2025. arXiv:2403.19647.

\bibitem{mueller2024reliability}
A. Mueller, A. Geiger. Measuring the Reliability of Causal Probing Methods. arXiv:2408.15510, 2024.

\bibitem{prairie2026parcae}
Q. Prairie et al. Parcae: Stable training of looped transformers. arXiv:2603.21014, 2026.

\bibitem{saunshi2024looped}
N. Saunshi, S. Kakade, et al. Looped Transformers simulate Chain-of-Thought in latent space. arXiv:2402.09710, 2024.

\bibitem{syed2024attribution}
A. Syed, C. Rager, A. Conmy. Attribution Patching Outperforms Automated Circuit Discovery. BlackboxNLP 2024. arXiv:2310.10348.

\end{thebibliography}
```

## Last checks before submitting

- [ ] LaTeX compiles without errors in Overleaf
- [ ] PDF looks clean (15 pages, tables render)
- [ ] HF dataset link works: https://huggingface.co/datasets/caiovicentino1/Qwen3.6-35B-A3B-mcr-stage-b
- [ ] Abstract < 1920 chars (arXiv limit)
- [ ] Title without special chars
- [ ] Author name correct
- [ ] License: CC BY 4.0

## After submission

- Note the arXiv ID (e.g., 2604.XXXXX)
- Update HF README with arXiv link
- Post on X/LinkedIn: "Paper live on arXiv: [link] — 9 convergent mechanistic interventions on Qwen3.6-35B-A3B"
- DM to Anthropic (Emmanuel Ameisen), Goodfire (Liv Gorton), Transluce team

## Takes about 10-15 minutes total once LaTeX compiles cleanly in Overleaf.

Good luck!
