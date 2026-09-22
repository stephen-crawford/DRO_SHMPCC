import DROSafety.DRO.ClopperPearsonBonferroni
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# From Clopper-Pearson confidence polytopes to Wasserstein coverage

The implementation constructs coordinate-wise Clopper-Pearson intervals

    lower_i <= p_i <= upper_i

and intersects the resulting box with the probability simplex.

Define the resulting confidence polytope

    C =
      { q in Delta_d :
          lower_i <= q_i <= upper_i for every i }.

Suppose the calibrated Wasserstein radius `rho` satisfies

    C subseteq Q_rho(center),

where `Q_rho(center)` is the finite Wasserstein ambiguity set.

Then whenever the true distribution lies inside the confidence polytope,
it also lies inside the Wasserstein ambiguity set.

Consequently,

    Wasserstein coverage failure
      subseteq
    Clopper-Pearson confidence-box failure.

Combining this deterministic inclusion with the Bonferroni theorem gives

    P[pTrue notin Q_rho(center)] <= beta_DRO.

This is exactly the `hCoverage` premise required by
`CombinedSafetyBound.lean`.
-/

namespace DROSafety.DRO

open MeasureTheory

/--
The box-simplex confidence polytope used by the finite-sample calibration.

A distribution belongs to the polytope when

1. it is a valid probability vector, and
2. every coordinate lies inside its confidence interval.
-/
def ConfidencePolytope
    {d : ℕ}
    (lower upper : Fin d → ℝ) :
    Set (ModeDistribution d) :=
  {
    q |
      IsProbabilityVector q ∧
      ∀ i,
        lower i ≤ q i ∧
        q i ≤ upper i
  }

/--
The deterministic property required of the calibrated radius.

`RadiusCoversConfidencePolytope D center lower upper rho` means that the
entire Clopper-Pearson confidence polytope is contained inside the
Wasserstein ambiguity ball of radius `rho` centered at `center`.

The implementation intends to establish this property by choosing

    rho = max_{q in C} W_D(center,q).
-/
def RadiusCoversConfidencePolytope
    {d : ℕ}
    (D : FiniteGroundCost d)
    (center : ModeDistribution d)
    (lower upper : Fin d → ℝ)
    (ρ : ℝ) :
    Prop :=
  ∀ q,
    q ∈ ConfidencePolytope lower upper →
    q ∈ FiniteWassersteinAmbiguity
      D
      center
      ρ

/--
If the true distribution is a probability vector and the simultaneous
confidence box does not fail, then the true distribution lies inside the
box-simplex confidence polytope.
-/
theorem trueDistribution_mem_confidencePolytope_of_not_failure
    {Ω : Type*}
    {d : ℕ}
    (pTrue : ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (ω : Ω)
    (hpTrue :
      IsProbabilityVector pTrue)
    (hNoFailure :
      ω ∉ ConfidenceBoxFailure
        pTrue
        lower
        upper) :
    pTrue ∈
      ConfidencePolytope
        (lower ω)
        (upper ω) := by

  classical

  constructor

  · exact hpTrue

  · change
      ¬ (¬ ∀ i,
        lower ω i ≤ pTrue i ∧
        pTrue i ≤ upper ω i)
      at hNoFailure

    exact
      Classical.not_not.mp
        hNoFailure

/--
Core deterministic inclusion.

If the calibrated radius contains the complete confidence polytope for
every realization, then any failure of Wasserstein coverage implies that
the Clopper-Pearson simultaneous confidence box must also have failed.
-/
theorem coverageFailure_subset_confidenceBoxFailure
    {Ω : Type*}
    {d : ℕ}
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (ρ : Ω → ℝ)
    (hpTrue :
      IsProbabilityVector pTrue)
    (hRadiusCover :
      ∀ ω,
        RadiusCoversConfidencePolytope
          D
          (pNominal ω)
          (lower ω)
          (upper ω)
          (ρ ω)) :
    CoverageFailure
        D
        pNominal
        pTrue
        ρ
      ⊆
    ConfidenceBoxFailure
      pTrue
      lower
      upper := by

  intro ω hCoverageFailure

  by_contra hNoBoxFailure

  have hTrueInPolytope :
      pTrue ∈
        ConfidencePolytope
          (lower ω)
          (upper ω) :=
    trueDistribution_mem_confidencePolytope_of_not_failure
      pTrue
      lower
      upper
      ω
      hpTrue
      hNoBoxFailure

  have hTrueCovered :
      TrueDistributionCovered
        D
        (pNominal ω)
        pTrue
        (ρ ω) := by

    exact
      hRadiusCover
        ω
        pTrue
        hTrueInPolytope

  exact
    hCoverageFailure
      hTrueCovered

/--
A probability bound on confidence-box failure transfers immediately to
a probability bound on Wasserstein ambiguity-set coverage failure.
-/
theorem wassersteinCoverageFailure_measure_le_of_confidenceBox
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (ρ : Ω → ℝ)
    (β : ENNReal)
    (hpTrue :
      IsProbabilityVector pTrue)
    (hRadiusCover :
      ∀ ω,
        RadiusCoversConfidencePolytope
          D
          (pNominal ω)
          (lower ω)
          (upper ω)
          (ρ ω))
    (hBoxFailure :
      μ
          (ConfidenceBoxFailure
            pTrue
            lower
            upper)
        ≤
      β) :
    μ
        (CoverageFailure
          D
          pNominal
          pTrue
          ρ)
      ≤
    β := by

  have hSubset :
      CoverageFailure
          D
          pNominal
          pTrue
          ρ
        ⊆
      ConfidenceBoxFailure
        pTrue
        lower
        upper :=
    coverageFailure_subset_confidenceBoxFailure
      D
      pNominal
      pTrue
      lower
      upper
      ρ
      hpTrue
      hRadiusCover

  have hMono :
      μ
          (CoverageFailure
            D
            pNominal
            pTrue
            ρ)
        ≤
      μ
          (ConfidenceBoxFailure
            pTrue
            lower
            upper) :=
    measure_mono
      hSubset

  exact
    le_trans
      hMono
      hBoxFailure

/--
Clopper-Pearson + Bonferroni + radius containment gives the desired
Wasserstein coverage bound.

This theorem discharges the `hCoverage` premise of the combined
DRO Safe-Horizon theorem.
-/
theorem clopperPearson_wassersteinCoverageFailure_bound
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (ρ : Ω → ℝ)
    (α : Fin d → ENNReal)
    (βDRO : ENNReal)
    (hpTrue :
      IsProbabilityVector pTrue)
    (hCoordinateCP :
      ∀ i,
        μ
            (CoordinateIntervalFailure
              pTrue
              lower
              upper
              i)
          ≤
        α i)
    (hAllocation :
      (∑' i, α i) ≤ βDRO)
    (hRadiusCover :
      ∀ ω,
        RadiusCoversConfidencePolytope
          D
          (pNominal ω)
          (lower ω)
          (upper ω)
          (ρ ω)) :
    μ
        (CoverageFailure
          D
          pNominal
          pTrue
          ρ)
      ≤
    βDRO := by

  have hBoxFailure :
      μ
          (ConfidenceBoxFailure
            pTrue
            lower
            upper)
        ≤
      βDRO :=
    clopperPearsonBonferroni_failure_bound
      μ
      pTrue
      lower
      upper
      α
      βDRO
      hCoordinateCP
      hAllocation

  exact
    wassersteinCoverageFailure_measure_le_of_confidenceBox
      μ
      D
      pNominal
      pTrue
      lower
      upper
      ρ
      βDRO
      hpTrue
      hRadiusCover
      hBoxFailure

/--
End-to-end bound using the Clopper-Pearson Wasserstein calibration.

Assume:

1. each Clopper-Pearson coordinate interval has its assigned coverage;
2. the Bonferroni allocations total at most `βDRO`;
3. the calibrated Wasserstein radius contains the entire confidence
   polytope;
4. `qStar` is genuinely worst-case in the ambiguity set; and
5. Safe Horizon violates its threshold with probability at most `βSH`.

Then the true-distribution risk exceeds the Safe-Horizon threshold with
probability at most

    βDRO + βSH.
-/
theorem combined_trueRiskViolation_bound_of_clopperPearson
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (qStar : Ω → ModeDistribution d)
    (risk : Ω → ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (ρ ε : Ω → ℝ)
    (α : Fin d → ENNReal)
    (βDRO βSH : ENNReal)
    (hpTrue :
      IsProbabilityVector pTrue)
    (hCoordinateCP :
      ∀ i,
        μ
            (CoordinateIntervalFailure
              pTrue
              lower
              upper
              i)
          ≤
        α i)
    (hAllocation :
      (∑' i, α i) ≤ βDRO)
    (hRadiusCover :
      ∀ ω,
        RadiusCoversConfidencePolytope
          D
          (pNominal ω)
          (lower ω)
          (upper ω)
          (ρ ω))
    (hWorst :
      ∀ ω,
        IsWorstCase
          (FiniteWassersteinAmbiguity
            D
            (pNominal ω)
            (ρ ω))
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

  have hCoverage :
      μ
          (CoverageFailure
            D
            pNominal
            pTrue
            ρ)
        ≤
      βDRO :=
    clopperPearson_wassersteinCoverageFailure_bound
      μ
      D
      pNominal
      pTrue
      lower
      upper
      ρ
      α
      βDRO
      hpTrue
      hCoordinateCP
      hAllocation
      hRadiusCover

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
      βDRO
      βSH
      hWorst
      hCoverage
      hSafeHorizon

end DROSafety.DRO
