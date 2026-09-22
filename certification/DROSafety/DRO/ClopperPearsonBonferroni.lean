import DROSafety.DRO.CombinedSafetyBound
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Bonferroni simultaneous coverage for categorical mode probabilities

The Wasserstein radius calibration used by the implementation first builds
one two-sided Clopper-Pearson interval for each categorical mode.

For mode `i`, suppose the interval satisfies

    P[pTrue_i notin [L_i, U_i]] ≤ α_i.

The mode-count coordinates of a multinomial random variable are dependent,
but independence is not required. By the union bound,

    P[exists i, pTrue_i notin [L_i, U_i]]
      ≤ sum_i α_i.

Therefore, if the confidence budget satisfies

    sum_i α_i ≤ β,

then

    P[forall i, L_i ≤ pTrue_i ≤ U_i]
      ≥ 1 - β.

This file proves the failure-probability form needed by the later
Wasserstein-coverage theorem.

The actual Clopper-Pearson per-coordinate coverage theorem is kept separate.
-/

namespace DROSafety.DRO

open MeasureTheory
open scoped BigOperators

/--
Failure of the confidence interval for one categorical coordinate.
-/
def CoordinateIntervalFailure
    {Ω : Type*}
    {d : ℕ}
    (pTrue : ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (i : Fin d) :
    Set Ω :=
  {
    ω |
      ¬ (
        lower ω i ≤ pTrue i ∧
        pTrue i ≤ upper ω i
      )
  }

/--
Failure of simultaneous coordinate-wise coverage.

This is precisely the event that the true categorical distribution is
outside the box

    product_i [lower_i, upper_i].
-/
def ConfidenceBoxFailure
    {Ω : Type*}
    {d : ℕ}
    (pTrue : ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ) :
    Set Ω :=
  {
    ω |
      ¬ ∀ i,
        lower ω i ≤ pTrue i ∧
        pTrue i ≤ upper ω i
  }

/--
Simultaneous confidence-box failure is the union of the coordinate-wise
failure events.
-/
theorem confidenceBoxFailure_eq_iUnion
    {Ω : Type*}
    {d : ℕ}
    (pTrue : ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ) :
    ConfidenceBoxFailure pTrue lower upper
      =
    ⋃ i,
      CoordinateIntervalFailure
        pTrue
        lower
        upper
        i := by

  classical

  ext ω

  simp [
    ConfidenceBoxFailure,
    CoordinateIntervalFailure
  ]

/--
Generic Bonferroni / union bound.

No independence assumption is required.
-/
theorem bonferroni_iUnion_measure_le
    {Ω : Type*}
    [MeasurableSpace Ω]
    {ι : Type*}
    [Countable ι]
    (μ : Measure Ω)
    (failure : ι → Set Ω)
    (α : ι → ENNReal)
    (hEach :
      ∀ i,
        μ (failure i) ≤ α i) :
    μ (⋃ i, failure i)
      ≤
    ∑' i, α i := by

  calc
    μ (⋃ i, failure i)
        ≤
      ∑' i, μ (failure i) := by

      exact measure_iUnion_le failure

    _ ≤
      ∑' i, α i := by

      exact
        ENNReal.tsum_le_tsum
          (fun i => hEach i)

/--
Bonferroni bound for the simultaneous Clopper-Pearson confidence box.

If coordinate `i` fails with probability at most `α i`, then simultaneous
coverage fails with probability at most the sum of the coordinate budgets.
-/
theorem confidenceBoxFailure_measure_le
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (pTrue : ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (α : Fin d → ENNReal)
    (hEach :
      ∀ i,
        μ
            (CoordinateIntervalFailure
              pTrue
              lower
              upper
              i)
          ≤
        α i) :
    μ
        (ConfidenceBoxFailure
          pTrue
          lower
          upper)
      ≤
    ∑' i, α i := by

  rw [
    confidenceBoxFailure_eq_iUnion
      pTrue
      lower
      upper
  ]

  exact
    bonferroni_iUnion_measure_le
      μ
      (fun i =>
        CoordinateIntervalFailure
          pTrue
          lower
          upper
          i)
      α
      hEach

/--
If the total coordinate-wise confidence budget is at most `β`, then the
simultaneous confidence box fails with probability at most `β`.

This is the exact Bonferroni statement required by the implementation.
-/
theorem confidenceBoxFailure_measure_le_beta
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (pTrue : ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (α : Fin d → ENNReal)
    (β : ENNReal)
    (hEach :
      ∀ i,
        μ
            (CoordinateIntervalFailure
              pTrue
              lower
              upper
              i)
          ≤
        α i)
    (hBudget :
      (∑' i, α i) ≤ β) :
    μ
        (ConfidenceBoxFailure
          pTrue
          lower
          upper)
      ≤
    β := by

  have hBonferroni :
      μ
          (ConfidenceBoxFailure
            pTrue
            lower
            upper)
        ≤
      ∑' i, α i :=
    confidenceBoxFailure_measure_le
      μ
      pTrue
      lower
      upper
      α
      hEach

  exact
    le_trans
      hBonferroni
      hBudget

/--
Specialization useful for Clopper-Pearson calibration.

If every coordinate failure probability is bounded by its assigned
Bonferroni budget and those assigned budgets total at most `βDRO`, then
the true categorical distribution lies inside all coordinate intervals
except on an event of measure at most `βDRO`.
-/
theorem clopperPearsonBonferroni_failure_bound
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (pTrue : ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (α : Fin d → ENNReal)
    (βDRO : ENNReal)
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
      (∑' i, α i) ≤ βDRO) :
    μ
        (ConfidenceBoxFailure
          pTrue
          lower
          upper)
      ≤
    βDRO := by

  exact
    confidenceBoxFailure_measure_le_beta
      μ
      pTrue
      lower
      upper
      α
      βDRO
      hCoordinateCP
      hAllocation

end DROSafety.DRO
