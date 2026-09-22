import DROSafety.DRO.TrueRiskBound
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Combined DRO + Safe-Horizon safety bound

This file states the end-to-end probabilistic structure of the
DRO Safe-Horizon argument.

There are two ways the desired true-risk guarantee can fail:

1. DRO coverage failure:
   the unknown true mode distribution is outside the Wasserstein
   ambiguity set.

2. Safe-Horizon failure:
   the risk under the worst-case DRO distribution exceeds the
   Safe-Horizon threshold.

If `qStar` is genuinely worst-case over the ambiguity set, then whenever
neither failure occurs,

    true risk ≤ worst-case DRO risk ≤ ε.

Therefore the true-risk violation event is contained in the union of the
two failure events.

Taking measures gives

    P(true risk > ε)
      ≤ β_DRO + β_SH.

This is the target end-to-end safety bound.
-/

namespace DROSafety.DRO

open MeasureTheory

/--
The event that the unknown true mode distribution is not contained in the
finite Wasserstein ambiguity set generated from the nominal distribution.
-/
def CoverageFailure
    {Ω : Type*}
    {d : ℕ}
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (ρ : Ω → ℝ) :
    Set Ω :=
  {
    ω |
      ¬ TrueDistributionCovered
          D
          (pNominal ω)
          pTrue
          (ρ ω)
  }

/--
The event that the risk under the certified worst-case DRO distribution
exceeds the requested Safe-Horizon threshold.
-/
def WorstCaseRiskViolation
    {Ω : Type*}
    {d : ℕ}
    (qStar : Ω → ModeDistribution d)
    (risk : Ω → ModeDistribution d)
    (ε : Ω → ℝ) :
    Set Ω :=
  {
    ω |
      ε ω <
        expectedRisk
          (qStar ω)
          (risk ω)
  }

/--
The event we ultimately care about: the true distribution has risk above
the Safe-Horizon threshold.
-/
def TrueRiskViolation
    {Ω : Type*}
    {d : ℕ}
    (pTrue : ModeDistribution d)
    (risk : Ω → ModeDistribution d)
    (ε : Ω → ℝ) :
    Set Ω :=
  {
    ω |
      ε ω <
        expectedRisk
          pTrue
          (risk ω)
  }

/--
Any true-risk violation must arise from either ambiguity-set coverage
failure or violation of the threshold under the worst-case distribution.
-/
theorem trueRiskViolation_subset_failureUnion
    {Ω : Type*}
    {d : ℕ}
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (qStar : Ω → ModeDistribution d)
    (risk : Ω → ModeDistribution d)
    (ρ ε : Ω → ℝ)
    (hWorst :
      ∀ ω,
        IsWorstCase
          (FiniteWassersteinAmbiguity
            D
            (pNominal ω)
            (ρ ω))
          (risk ω)
          (qStar ω)) :
    TrueRiskViolation pTrue risk ε
      ⊆
    Set.union
      (CoverageFailure D pNominal pTrue ρ)
      (WorstCaseRiskViolation qStar risk ε) := by

  intro ω hTrue

  by_cases hCovered :
      TrueDistributionCovered
        D
        (pNominal ω)
        pTrue
        (ρ ω)

  · have hDom :
        expectedRisk pTrue (risk ω)
          ≤
        expectedRisk (qStar ω) (risk ω) :=
      trueRisk_le_worstCaseRisk
        D
        hCovered
        (hWorst ω)

    have hStarViolation :
        ε ω <
          expectedRisk
            (qStar ω)
            (risk ω) :=
      lt_of_lt_of_le
        hTrue
        hDom

    exact Or.inr hStarViolation

  · exact Or.inl hCovered

/--
Combined DRO + Safe-Horizon probability bound.

If ambiguity-set coverage fails with probability at most `βDRO` and
the worst-case Safe-Horizon bound fails with probability at most `βSH`,
then true risk exceeds the threshold with probability at most

    βDRO + βSH.
-/
theorem combined_trueRiskViolation_bound
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (qStar : Ω → ModeDistribution d)
    (risk : Ω → ModeDistribution d)
    (ρ ε : Ω → ℝ)
    (βDRO βSH : ENNReal)
    (hWorst :
      ∀ ω,
        IsWorstCase
          (FiniteWassersteinAmbiguity
            D
            (pNominal ω)
            (ρ ω))
          (risk ω)
          (qStar ω))
    (hCoverage :
      μ (CoverageFailure D pNominal pTrue ρ)
        ≤ βDRO)
    (hSafeHorizon :
      μ (WorstCaseRiskViolation qStar risk ε)
        ≤ βSH) :
    μ (TrueRiskViolation pTrue risk ε)
      ≤ βDRO + βSH := by

  have hSubset :
      TrueRiskViolation pTrue risk ε
        ⊆
      Set.union
        (CoverageFailure D pNominal pTrue ρ)
        (WorstCaseRiskViolation qStar risk ε) :=
    trueRiskViolation_subset_failureUnion
      D
      pNominal
      pTrue
      qStar
      risk
      ρ
      ε
      hWorst

  have hMono :
      μ (TrueRiskViolation pTrue risk ε)
        ≤
      μ
        (Set.union
          (CoverageFailure D pNominal pTrue ρ)
          (WorstCaseRiskViolation qStar risk ε)) :=
    measure_mono hSubset

  have hUnion :
      μ
          (Set.union
            (CoverageFailure D pNominal pTrue ρ)
            (WorstCaseRiskViolation qStar risk ε))
        ≤
      μ (CoverageFailure D pNominal pTrue ρ)
        +
      μ (WorstCaseRiskViolation qStar risk ε) :=
    measure_union_le
      (CoverageFailure D pNominal pTrue ρ)
      (WorstCaseRiskViolation qStar risk ε)

  have hComponent :
      μ (CoverageFailure D pNominal pTrue ρ)
        +
      μ (WorstCaseRiskViolation qStar risk ε)
        ≤
      βDRO + βSH :=
    add_le_add
      hCoverage
      hSafeHorizon

  exact
    le_trans
      hMono
      (le_trans hUnion hComponent)

/--
Real-valued confidence-parameter version of the combined bound.
-/
theorem combined_trueRiskViolation_bound_ofReal
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (qStar : Ω → ModeDistribution d)
    (risk : Ω → ModeDistribution d)
    (ρ ε : Ω → ℝ)
    (βDRO βSH : ℝ)
    (hWorst :
      ∀ ω,
        IsWorstCase
          (FiniteWassersteinAmbiguity
            D
            (pNominal ω)
            (ρ ω))
          (risk ω)
          (qStar ω))
    (hCoverage :
      μ (CoverageFailure D pNominal pTrue ρ)
        ≤ ENNReal.ofReal βDRO)
    (hSafeHorizon :
      μ (WorstCaseRiskViolation qStar risk ε)
        ≤ ENNReal.ofReal βSH) :
    μ (TrueRiskViolation pTrue risk ε)
      ≤
    ENNReal.ofReal βDRO +
      ENNReal.ofReal βSH := by

  exact
    combined_trueRiskViolation_bound
      μ
      D
      pNominal
      pTrue
      qStar
      risk
      ρ
      ε
      (ENNReal.ofReal βDRO)
      (ENNReal.ofReal βSH)
      hWorst
      hCoverage
      hSafeHorizon

end DROSafety.DRO
