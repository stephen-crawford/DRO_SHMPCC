# ⚠️ Risk-score update (follow-up work) — reconceived as CVaR

**Date:** 2026-07-08
**Status:** MARKER for future CDC-paper revisions — no code in *this* repo changed yet.

## What changed (in the follow-up papers, not here)

This codebase implements the **Wasserstein mode-reweighting** primitive of the CDC
paper (Crawford & Ayanian, *Wasserstein Mode Reweighting for Scenario-Based MPC
under Switching Obstacle Dynamics*). In that paper the per-mode **risk score** is an
inverse mean-proximity / mean-Mahalanobis-clearance cost.

In the follow-up work (the continuous-mode $W_1$ paper "Paper A" and the
Bures–Wasserstein SE(3) paper "Paper E", both in
`~/Documents/templates/mpc-template-python/paper_scaffolds/`), the per-mode risk has
been **reconceived as a coherent, tail-aware Conditional Value-at-Risk (CVaR)** of
the collision margin, replacing the ad-hoc inverse-mean-proximity.

- **Formal model:**
  `~/Documents/templates/mpc-template-python/mpc_template/modules/samplers/CVAR_RISK_MODEL.md`
- **Implementation:**
  `~/Documents/templates/mpc-template-python/mpc_template/modules/samplers/risk.py`
  (`cvar_mode_risk`, `cvar_gaussian_coeff`).
- **Short form:** `v_i = 1 / ( max(d_i − k_α, 0) + ε₀ )`, where `d_i` is the nominal
  min-Mahalanobis clearance and `k_α = φ(Φ⁻¹(α))/(1−α)` is the Gaussian CVaR
  coefficient (α=0 recovers the old mean-margin risk; α=0.9 ⇒ k≈1.755).

## Why (empirically verified)
Swapping mean → VaR → CVaR does **not** change the qualitative reweighting outcome
(the bang-bang-vs-entropic diagnosis is a property of the allocation LP, not of the
risk score). CVaR is adopted for **coherence** (subadditive/convex, unlike VaR),
**tail-awareness**, and to connect to the risk-averse / chance-constrained MPC
literature (Rockafellar–Uryasev).

## ACTION for a future CDC-paper revision
If this repo's code or the CDC paper is revised, **update the "risk-score" /
per-mode-cost section to the CVaR model** above (cite `CVAR_RISK_MODEL.md`), and
correspondingly update the risk computation in this codebase. Until then this file
is a standing reminder that the canonical risk score in the follow-up line is CVaR,
not the inverse-mean-proximity used in the original submission.
