import DROSafety.DRO.FiniteTransportDuality
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# True-distribution risk bounds from finite Wasserstein DRO

This file proves the central deterministic DRO safety implication.

Let

* `pNominal` be the learned nominal mode distribution,
* `pTrue` be the unknown true mode distribution,
* `Qρ(pNominal)` be the finite Wasserstein ambiguity set,
* `risk j` be the conditional risk associated with mode `j`,
* `qStar` be a worst-case distribution in the ambiguity set.

If

    pTrue ∈ Qρ(pNominal),

then worst-case optimality of `qStar` implies

    expectedRisk pTrue risk
      ≤ expectedRisk qStar risk.

Consequently, any Safe-Horizon bound proved for the DRO distribution
`qStar` transfers directly to the true distribution whenever the true
distribution is covered by the ambiguity set.

This is the key deterministic bridge between the DRO layer and the
Safe-Horizon layer.
-/

namespace DROSafety.DRO

/--
The ambiguity set covers the true distribution.
-/
def TrueDistributionCovered
    {d : ℕ}
    (D : FiniteGroundCost d)
    (pNominal pTrue : ModeDistribution d)
    (ρ : ℝ) :
    Prop :=
  pTrue ∈
    FiniteWassersteinAmbiguity
      D
      pNominal
      ρ

/--
Coverage implies that the true distribution is itself a valid probability
vector.
-/
theorem trueDistribution_isProbabilityVector_of_covered
    {d : ℕ}
    (D : FiniteGroundCost d)
    {pNominal pTrue : ModeDistribution d}
    {ρ : ℝ}
    (hCovered :
      TrueDistributionCovered
        D
        pNominal
        pTrue
        ρ) :
    IsProbabilityVector pTrue := by

  exact hCovered.1

/--
Central DRO domination theorem.

If the unknown true distribution belongs to the ambiguity set and `qStar`
is genuinely worst-case over that set, then the risk under the true
distribution cannot exceed the risk under `qStar`.
-/
theorem trueRisk_le_worstCaseRisk
    {d : ℕ}
    (D : FiniteGroundCost d)
    {pNominal pTrue risk qStar : ModeDistribution d}
    {ρ : ℝ}
    (hCovered :
      TrueDistributionCovered
        D
        pNominal
        pTrue
        ρ)
    (hWorst :
      IsWorstCase
        (FiniteWassersteinAmbiguity
          D
          pNominal
          ρ)
        risk
        qStar) :
    expectedRisk pTrue risk ≤
      expectedRisk qStar risk := by

  exact
    hWorst.2
      pTrue
      hCovered

/--
If Safe Horizon establishes an upper bound `ε` on the risk under the
worst-case DRO distribution, then the same bound holds under every true
distribution covered by the ambiguity set.
-/
theorem trueRisk_le_of_worstCaseRisk_le
    {d : ℕ}
    (D : FiniteGroundCost d)
    {pNominal pTrue risk qStar : ModeDistribution d}
    {ρ ε : ℝ}
    (hCovered :
      TrueDistributionCovered
        D
        pNominal
        pTrue
        ρ)
    (hWorst :
      IsWorstCase
        (FiniteWassersteinAmbiguity
          D
          pNominal
          ρ)
        risk
        qStar)
    (hSafe :
      expectedRisk qStar risk ≤ ε) :
    expectedRisk pTrue risk ≤ ε := by

  exact
    le_trans
      (trueRisk_le_worstCaseRisk
        D
        hCovered
        hWorst)
      hSafe

/--
The primal-dual certificate from `FiniteTransportDuality` is sufficient to
establish the true-distribution risk domination theorem.

This removes `IsWorstCase` as an external assumption.
-/
theorem trueRisk_le_primalDualCertifiedRisk
    {d : ℕ}
    (D : FiniteGroundCost d)
    {pNominal pTrue risk qStar : ModeDistribution d}
    {ρ lam : ℝ}
    {α : Fin d → ℝ}
    (hCovered :
      TrueDistributionCovered
        D
        pNominal
        pTrue
        ρ)
    (hStarFeasible :
      qStar ∈
        FiniteWassersteinAmbiguity
          D
          pNominal
          ρ)
    (hDual :
      TransportDualFeasible
        D
        risk
        α
        lam)
    (hEquality :
      expectedRisk qStar risk =
        transportDualObjective
          pNominal
          ρ
          α
          lam) :
    expectedRisk pTrue risk ≤
      expectedRisk qStar risk := by

  have hWorst :
      IsWorstCase
        (FiniteWassersteinAmbiguity
          D
          pNominal
          ρ)
        risk
        qStar :=
    isWorstCase_of_primalDualEquality
      D
      hStarFeasible
      hDual
      hEquality

  exact
    trueRisk_le_worstCaseRisk
      D
      hCovered
      hWorst

/--
Main deterministic DRO-to-Safe-Horizon bridge.

If

1. the true distribution is covered by the Wasserstein ambiguity set,
2. `qStar` has a valid primal-dual worst-case certificate, and
3. Safe Horizon establishes risk at most `ε` under `qStar`,

then the risk under the true distribution is also at most `ε`.
-/
theorem trueRisk_le_safeThreshold_of_primalDualCertificate
    {d : ℕ}
    (D : FiniteGroundCost d)
    {pNominal pTrue risk qStar : ModeDistribution d}
    {ρ lam ε : ℝ}
    {α : Fin d → ℝ}
    (hCovered :
      TrueDistributionCovered
        D
        pNominal
        pTrue
        ρ)
    (hStarFeasible :
      qStar ∈
        FiniteWassersteinAmbiguity
          D
          pNominal
          ρ)
    (hDual :
      TransportDualFeasible
        D
        risk
        α
        lam)
    (hEquality :
      expectedRisk qStar risk =
        transportDualObjective
          pNominal
          ρ
          α
          lam)
    (hSafe :
      expectedRisk qStar risk ≤ ε) :
    expectedRisk pTrue risk ≤ ε := by

  have hTrueLeStar :
      expectedRisk pTrue risk ≤
        expectedRisk qStar risk :=
    trueRisk_le_primalDualCertifiedRisk
      D
      hCovered
      hStarFeasible
      hDual
      hEquality

  exact
    le_trans
      hTrueLeStar
      hSafe

/--
Equivalent violation form.

If the true risk exceeds `ε`, then at least one of the assumptions needed
for the deterministic DRO safety bridge cannot hold.

Under the stated coverage and optimality assumptions, exceeding `ε`
contradicts a Safe-Horizon bound on the worst-case distribution.
-/
theorem trueRisk_not_gt_safeThreshold
    {d : ℕ}
    (D : FiniteGroundCost d)
    {pNominal pTrue risk qStar : ModeDistribution d}
    {ρ ε : ℝ}
    (hCovered :
      TrueDistributionCovered
        D
        pNominal
        pTrue
        ρ)
    (hWorst :
      IsWorstCase
        (FiniteWassersteinAmbiguity
          D
          pNominal
          ρ)
        risk
        qStar)
    (hSafe :
      expectedRisk qStar risk ≤ ε) :
    ¬ ε < expectedRisk pTrue risk := by

  have hTrueSafe :
      expectedRisk pTrue risk ≤ ε :=
    trueRisk_le_of_worstCaseRisk_le
      D
      hCovered
      hWorst
      hSafe

  exact
    not_lt_of_ge
      hTrueSafe

end DROSafety.DRO
