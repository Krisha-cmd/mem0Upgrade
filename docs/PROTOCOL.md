# Pre-registration protocol

This file must be completed and dated **before** any E2 test-set generation (Phase 5).

## Hypotheses (from the build spec, section 7)
- **H1 (E1, retrieval):** at matched preference tokens, similarity gates have clearly lower recall of the gold preference than scope gates, and the gap grows with N.
- **H2 (E2, non-inferiority):** following(S_best) − following(B2_all) > −2.0 points. Tested one-sided, TOST-style, on the paired difference with a bootstrap CI (10,000 resamples, seed 0).
- **H3 (E2, superiority):** following(S_best) > following(B3_sim_k5). Paired McNemar test.
- **H4 (E2, tokens):** preference tokens(S_best) ≤ 0.3 × preference tokens(B2_all). Wilcoxon signed-rank test.
- **Correction:** Holm across H2–H4, within each backbone.

## Amendments (dated, made before any cluster run or test-set result)
- **2026-11-21, A1 (tuning objective, D-032):** τ and k are tuned on dev folds as the strictest setting with pooled dev recall(G) ≥ 0.95. This replaces micro-F1.
- **2026-11-21, A2 (cross-topic split, D-033):** every E1 and E2 analysis is reported on two splits, `full` (PrefEval same-topic queries) and `xtopic` (verified cross-topic requests). Holm correction runs across H2–H4 within each backbone *and split*.
  - *Proposed (Krisha to confirm before E2):* `xtopic` is the primary split for H1 and H3, which are the claims about similarity missing applicable preferences. Both splits are co-primary for H2 and H4.
- **2026-11-21, A3 (xtopic acceptance):** the `xtopic` split is used for reported results only if the blind human audit gives kept-request precision ≥ 0.85.

## S_best
- **Chosen on dev folds (E1 tuning, N=25):** _TBD — Phase 5_
- **Selection rule:** _TBD — to be written here before looking at E2 outputs_
- **Date:** _TBD_
