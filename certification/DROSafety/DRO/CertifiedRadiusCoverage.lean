import DROSafety.DRO.FiniteEnumeratorRadiusFamily
import DROSafety.DRO.ClopperPearsonBonferroni
import DROSafety.DRO.ConfidencePolytopeWasserstein
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Certified finite-enumerator Wasserstein coverage

This file records the point reached by the calibration proof.

The box-simplex geometry and the finite Wasserstein-radius construction
are now completely discharged.  The only remaining statistical premise
for confidence-set coverage is the per-coordinate Clopper-Pearson
coverage statement.

For each sample realization `ω`:

* `lower ω`, `upper ω` are the coordinate confidence intervals;
* `center ω` is the Wasserstein-ball center;
* `data ω` contains the complete finite enumerator output and transport
  witnesses;
* `finiteEnumeratorRadiusFamily data ω` is the certified radius.

The theorem below packages the radius-cover conclusion needed by the
existing Clopper-Pearson/Wasserstein coverage theorem.
-/

namespace DROSafety.DRO

/--
Finite enumerator data discharges the Wasserstein radius-cover premise
pointwise.
-/
theorem certifiedRadiusCover
    {Ω : Type*}
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : Ω → ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (data :
      ∀ ω : Ω,
        FiniteEnumeratorTransportData
          D
          (center ω)
          (lower ω)
          (upper ω))
    (hd :
      0 < d)
    (hLower :
      ∀ ω : Ω,
        ∀ i : Fin d,
          0 ≤ lower ω i) :
    ∀ ω : Ω,
      RadiusCoversConfidencePolytope
        D
        (center ω)
        (lower ω)
        (upper ω)
        (finiteEnumeratorRadiusFamily data ω) := by

  exact
    radiusCover_family_of_finiteEnumeratorData
      D
      center
      lower
      upper
      data
      hd
      hLower

/--
Consequently, if the true distribution lies in the coordinate confidence
polytope for a realization, it lies in the certified Wasserstein ball.
-/
theorem trueDistributionCovered_of_mem_confidencePolytope
    {Ω : Type*}
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : Ω → ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (data :
      ∀ ω : Ω,
        FiniteEnumeratorTransportData
          D
          (center ω)
          (lower ω)
          (upper ω))
    (hd :
      0 < d)
    (hLower :
      ∀ ω : Ω,
        ∀ i : Fin d,
          0 ≤ lower ω i)
    (pTrue : ModeDistribution d)
    (ω : Ω)
    (hp :
      pTrue ∈
        ConfidencePolytope
          (lower ω)
          (upper ω)) :
    TrueDistributionCovered
      D
      (center ω)
      pTrue
      (finiteEnumeratorRadiusFamily data ω) := by

  unfold TrueDistributionCovered

  exact
    confidencePoint_mem_ambiguity_of_finiteEnumeratorFamily
      D
      center
      lower
      upper
      data
      hd
      hLower
      ω
      hp

end DROSafety.DRO
