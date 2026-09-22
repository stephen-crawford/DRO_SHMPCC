import DROSafety.DRO.FiniteEnumeratorRadiusFamily
import DROSafety.DRO.ConfidencePolytopeWasserstein
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Combined safety theorem with finite enumerator radius

This file substitutes the certified finite box-simplex enumerator radius
directly into the existing Clopper-Pearson + Wasserstein + Safe-Horizon
safety theorem.

The radius is no longer an arbitrary function accompanied by an abstract

    hRadiusCover

assumption.

Instead, for every sample realization `ω`, we are given finite enumerator
transport data

    data ω :
      FiniteEnumeratorTransportData
        D
        (pNominal ω)
        (lower ω)
        (upper ω),

and define

    rho(ω) = finiteEnumeratorRadiusFamily data ω.

The previously proved box-simplex geometry and finite transport certificate
automatically imply that this radius covers the complete confidence
polytope.

Thus the remaining statistical calibration assumption is only the
per-coordinate Clopper-Pearson coverage property.

The combined theorem then yields

    μ(TrueRiskViolation pTrue risk ε)
      ≤ βDRO + βSH.
-/

namespace DROSafety.DRO

open MeasureTheory

/--
Finite enumerator data removes the abstract Wasserstein radius-cover
assumption from the Clopper-Pearson coverage theorem.

The only remaining statistical premise is the per-coordinate confidence
interval coverage bound.
-/
theorem clopperPearson_finiteEnumeratorCoverageFailure_bound
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (data :
      ∀ ω : Ω,
        FiniteEnumeratorTransportData
          D
          (pNominal ω)
          (lower ω)
          (upper ω))
    (α : Fin d → ENNReal)
    (βDRO : ENNReal)
    (hd :
      0 < d)
    (hLower :
      ∀ ω : Ω,
        ∀ i : Fin d,
          0 ≤ lower ω i)
    (hpTrue :
      IsProbabilityVector pTrue)
    (hCoordinateCP :
      ∀ i : Fin d,
        μ
            (CoordinateIntervalFailure
              pTrue
              lower
              upper
              i)
          ≤
        α i)
    (hAllocation :
      ∑' i : Fin d,
        α i
        ≤
      βDRO) :
    μ
        (CoverageFailure
          D
          pNominal
          pTrue
          (finiteEnumeratorRadiusFamily data))
      ≤
    βDRO := by

  have hRadiusCover :
      ∀ ω : Ω,
        RadiusCoversConfidencePolytope
          D
          (pNominal ω)
          (lower ω)
          (upper ω)
          (finiteEnumeratorRadiusFamily data ω) :=
    radiusCover_family_of_finiteEnumeratorData
      D
      pNominal
      lower
      upper
      data
      hd
      hLower

  exact
    clopperPearson_wassersteinCoverageFailure_bound
      μ
      D
      pNominal
      pTrue
      lower
      upper
      (finiteEnumeratorRadiusFamily data)
      α
      βDRO
      hpTrue
      hCoordinateCP
      hAllocation
      hRadiusCover

/--
Main integrated finite-enumerator safety theorem.

The Wasserstein ambiguity radius is computed pointwise from the finite
enumerator data.

Therefore `hRadiusCover` is no longer an external assumption.

If:

* the true mode distribution is a probability vector;
* each coordinate confidence interval has its Clopper-Pearson coverage
  guarantee;
* the coordinate failure budgets sum to at most `βDRO`;
* `qStar` is worst-case over the certified finite Wasserstein ambiguity set;
* the Safe-Horizon worst-case-risk violation event has measure at most
  `βSH`;

then the true-risk violation event has measure at most

    βDRO + βSH.
-/
theorem combined_trueRiskViolation_bound_of_finiteEnumeratorClopperPearson
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (qStar risk : Ω → ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (data :
      ∀ ω : Ω,
        FiniteEnumeratorTransportData
          D
          (pNominal ω)
          (lower ω)
          (upper ω))
    (ε : Ω → ℝ)
    (α : Fin d → ENNReal)
    (βDRO βSH : ENNReal)
    (hd :
      0 < d)
    (hLower :
      ∀ ω : Ω,
        ∀ i : Fin d,
          0 ≤ lower ω i)
    (hpTrue :
      IsProbabilityVector pTrue)
    (hCoordinateCP :
      ∀ i : Fin d,
        μ
            (CoordinateIntervalFailure
              pTrue
              lower
              upper
              i)
          ≤
        α i)
    (hAllocation :
      ∑' i : Fin d,
        α i
        ≤
      βDRO)
    (hWorst :
      ∀ ω : Ω,
        IsWorstCase
          (FiniteWassersteinAmbiguity
            D
            (pNominal ω)
            (finiteEnumeratorRadiusFamily
              data
              ω))
          (risk ω)
          (qStar ω))
    (hSafeHorizon :
      μ
          (WorstCaseRiskViolation
            qStar
            risk
            ε)
        ≤
      βSH) :
    μ
        (TrueRiskViolation
          pTrue
          risk
          ε)
      ≤
    βDRO + βSH := by

  have hRadiusCover :
      ∀ ω : Ω,
        RadiusCoversConfidencePolytope
          D
          (pNominal ω)
          (lower ω)
          (upper ω)
          (finiteEnumeratorRadiusFamily data ω) :=
    radiusCover_family_of_finiteEnumeratorData
      D
      pNominal
      lower
      upper
      data
      hd
      hLower

  exact
    combined_trueRiskViolation_bound_of_clopperPearson
      μ
      D
      pNominal
      pTrue
      qStar
      risk
      lower
      upper
      (finiteEnumeratorRadiusFamily data)
      ε
      α
      βDRO
      βSH
      hpTrue
      hCoordinateCP
      hAllocation
      hRadiusCover
      hWorst
      hSafeHorizon

end DROSafety.DRO
