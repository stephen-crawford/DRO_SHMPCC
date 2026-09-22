import DROSafety.DRO.ClopperPearsonTailUnion
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Clopper-Pearson endpoint inversion

The previous file reduced two-sided Clopper-Pearson coverage to two
one-sided binomial bounds.

This file proves those one-sided bounds from the structural properties that
the exact Clopper-Pearson endpoints are supposed to satisfy.

For the lower endpoint `L(k)`:

* `L` is monotone in the observed count `k`;
* whenever `p < L(k)`, the upper binomial tail from `k` has probability
  at most `αLower`.

Then

    {k | p < L(k)}

is either empty or exactly an upper tail `{k | k0 ≤ k}`, so its total
probability is at most `αLower`.

For the upper endpoint `U(k)`:

* `U` is monotone in `k`;
* some count has `p ≤ U(k)` (for Clopper-Pearson, `U(m) = 1`);
* whenever `U(k) < p`, the lower binomial tail through `k` has probability
  at most `αUpper`.

Then

    {k | U(k) < p}

is either empty or exactly a lower tail `{k | k ≤ k0}`, so its probability
is at most `αUpper`.

The next file only needs to prove that the actual Clopper-Pearson endpoint
formulas satisfy these inversion properties.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
The lower-endpoint inversion property.

If the true parameter lies below the lower endpoint reported after
observing `k`, then observing at least `k` successes has probability at
most `α`.
-/
def LowerEndpointBinomialInversion
    (m : ℕ)
    (p : unitInterval)
    (lowerCP : ℕ → ℝ)
    (α : ENNReal) :
    Prop :=
  ∀ k : ℕ,
    (p : ℝ) < lowerCP k →
    ProbabilityTheory.binomial m p
        (Set.Ici k)
      ≤
    α

/--
The upper-endpoint inversion property.

If the true parameter lies above the upper endpoint reported after
observing `k`, then observing at most `k` successes has probability at
most `α`.
-/
def UpperEndpointBinomialInversion
    (m : ℕ)
    (p : unitInterval)
    (upperCP : ℕ → ℝ)
    (α : ENNReal) :
    Prop :=
  ∀ k : ℕ,
    upperCP k < (p : ℝ) →
    ProbabilityTheory.binomial m p
        (Set.Iic k)
      ≤
    α

/--
For a monotone lower endpoint, if lower-endpoint failure occurs at all,
the failure set is exactly an upper tail beginning at its first failing
count.
-/
theorem lowerEndpointFailure_eq_Ici_find
    (p : ℝ)
    (lowerCP : ℕ → ℝ)
    (hMono :
      Monotone lowerCP)
    (hExists :
      ∃ k : ℕ,
        p < lowerCP k) :
    LowerEndpointFailure
        p
        lowerCP
      =
    Set.Ici
      (Nat.find hExists) := by

  classical

  ext k

  simp only [
    LowerEndpointFailure,
    Set.mem_ofPred_eq,
    Set.mem_Ici
  ]

  constructor

  · intro hk

    exact
      Nat.find_min'
        hExists
        hk

  · intro hk

    have hFirst :
        p <
          lowerCP
            (Nat.find hExists) :=
      Nat.find_spec
        hExists

    have hMonoStep :
        lowerCP
            (Nat.find hExists)
          ≤
        lowerCP k :=
      hMono hk

    exact
      lt_of_lt_of_le
        hFirst
        hMonoStep

/--
A monotone lower endpoint satisfying the binomial inversion property has
the desired lower-tail coverage guarantee.
-/
theorem lowerEndpointFailure_measure_le_of_inversion
    (m : ℕ)
    (p : unitInterval)
    (lowerCP : ℕ → ℝ)
    (α : ENNReal)
    (hMono :
      Monotone lowerCP)
    (hInvert :
      LowerEndpointBinomialInversion
        m
        p
        lowerCP
        α) :
    ProbabilityTheory.binomial m p
        (LowerEndpointFailure
          (p : ℝ)
          lowerCP)
      ≤
    α := by

  classical

  by_cases hExists :
      ∃ k : ℕ,
        (p : ℝ) < lowerCP k

  · have hSet :
        LowerEndpointFailure
            (p : ℝ)
            lowerCP
          =
        Set.Ici
          (Nat.find hExists) :=
      lowerEndpointFailure_eq_Ici_find
        (p : ℝ)
        lowerCP
        hMono
        hExists

    rw [hSet]

    exact
      hInvert
        (Nat.find hExists)
        (Nat.find_spec hExists)

  · have hEmpty :
        LowerEndpointFailure
            (p : ℝ)
            lowerCP
          =
        ∅ := by

      ext k

      change
        ((p : ℝ) < lowerCP k)
          ↔
        False

      constructor

      · intro hk

        exact
          hExists
            ⟨k, hk⟩

      · intro hFalse

        exact
          False.elim hFalse

    rw [hEmpty]

    simp

/--
For a monotone upper endpoint, let `c` be the first count for which the
true parameter lies below the reported upper endpoint.

Then the upper-endpoint failure set is exactly `{k | k < c}`.
-/
theorem upperEndpointFailure_eq_Iio_find
    (p : ℝ)
    (upperCP : ℕ → ℝ)
    (hMono :
      Monotone upperCP)
    (hGoodExists :
      ∃ k : ℕ,
        p ≤ upperCP k) :
    UpperEndpointFailure
        p
        upperCP
      =
    Set.Iio
      (Nat.find hGoodExists) := by

  classical

  ext k

  simp only [
    UpperEndpointFailure,
    Set.mem_ofPred_eq,
    Set.mem_Iio
  ]

  constructor

  · intro hk

    by_contra hNotLt

    have hFindLe :
        Nat.find hGoodExists ≤ k :=
      Nat.le_of_not_gt
        hNotLt

    have hGood :
        p ≤
          upperCP
            (Nat.find hGoodExists) :=
      Nat.find_spec
        hGoodExists

    have hMonoStep :
        upperCP
            (Nat.find hGoodExists)
          ≤
        upperCP k :=
      hMono
        hFindLe

    have hpLe :
        p ≤ upperCP k :=
      le_trans
        hGood
        hMonoStep

    exact
      (not_lt_of_ge hpLe)
        hk

  · intro hk

    have hNotGood :
        ¬ p ≤ upperCP k :=
      Nat.find_min
        hGoodExists
        hk

    exact
      lt_of_not_ge
        hNotGood

/--
If upper-endpoint failure occurs somewhere, the first non-failing count is
strictly positive.
-/
theorem upperEndpoint_firstGood_pos
    (p : ℝ)
    (upperCP : ℕ → ℝ)
    (hMono :
      Monotone upperCP)
    (hGoodExists :
      ∃ k : ℕ,
        p ≤ upperCP k)
    (hFailExists :
      ∃ k : ℕ,
        upperCP k < p) :
    0 <
      Nat.find hGoodExists := by

  classical

  rcases hFailExists with
    ⟨k, hk⟩

  have hGood :
      p ≤
        upperCP
          (Nat.find hGoodExists) :=
    Nat.find_spec
      hGoodExists

  by_contra hNotPos

  have hZero :
      Nat.find hGoodExists = 0 := by
    omega

  have hZeroLeK :
      Nat.find hGoodExists ≤ k := by

    rw [hZero]

    exact
      Nat.zero_le k

  have hMonoStep :
      upperCP
          (Nat.find hGoodExists)
        ≤
      upperCP k :=
    hMono
      hZeroLeK

  have hpLe :
      p ≤ upperCP k :=
    le_trans
      hGood
      hMonoStep

  exact
    (not_lt_of_ge hpLe)
      hk

/--
When the first non-failing upper-endpoint count is positive, the preceding
count is still a failing count.
-/
theorem upperEndpoint_pred_firstGood_fails
    (p : ℝ)
    (upperCP : ℕ → ℝ)
    (hGoodExists :
      ∃ k : ℕ,
        p ≤ upperCP k)
    (hPos :
      0 < Nat.find hGoodExists) :
    upperCP
        (Nat.find hGoodExists - 1)
      <
    p := by

  classical

  have hPredLt :
      Nat.find hGoodExists - 1
        <
      Nat.find hGoodExists := by
    omega

  have hNotGood :
      ¬ p ≤
        upperCP
          (Nat.find hGoodExists - 1) :=
    Nat.find_min
      hGoodExists
      hPredLt

  exact
    lt_of_not_ge
      hNotGood

/--
For a positive natural cutoff,

    {k | k < c} = {k | k ≤ c - 1}.
-/
theorem Iio_nat_eq_Iic_pred
    (c : ℕ)
    (hc :
      0 < c) :
    Set.Iio c
      =
    Set.Iic (c - 1) := by

  ext k

  simp only [
    Set.mem_Iio,
    Set.mem_Iic
  ]

  omega

/--
A monotone upper endpoint satisfying the binomial inversion property has
the desired upper-tail coverage guarantee.
-/
theorem upperEndpointFailure_measure_le_of_inversion
    (m : ℕ)
    (p : unitInterval)
    (upperCP : ℕ → ℝ)
    (α : ENNReal)
    (hMono :
      Monotone upperCP)
    (hGoodExists :
      ∃ k : ℕ,
        (p : ℝ) ≤ upperCP k)
    (hInvert :
      UpperEndpointBinomialInversion
        m
        p
        upperCP
        α) :
    ProbabilityTheory.binomial m p
        (UpperEndpointFailure
          (p : ℝ)
          upperCP)
      ≤
    α := by

  classical

  by_cases hFailExists :
      ∃ k : ℕ,
        upperCP k < (p : ℝ)

  · have hFirstPos :
        0 <
          Nat.find hGoodExists :=
      upperEndpoint_firstGood_pos
        (p : ℝ)
        upperCP
        hMono
        hGoodExists
        hFailExists

    have hPredFail :
        upperCP
            (Nat.find hGoodExists - 1)
          <
        (p : ℝ) :=
      upperEndpoint_pred_firstGood_fails
        (p : ℝ)
        upperCP
        hGoodExists
        hFirstPos

    have hSet :
        UpperEndpointFailure
            (p : ℝ)
            upperCP
          =
        Set.Iic
          (Nat.find hGoodExists - 1) := by

      calc
        UpperEndpointFailure
            (p : ℝ)
            upperCP
            =
          Set.Iio
            (Nat.find hGoodExists) :=
          upperEndpointFailure_eq_Iio_find
            (p : ℝ)
            upperCP
            hMono
            hGoodExists

        _ =
          Set.Iic
            (Nat.find hGoodExists - 1) :=
          Iio_nat_eq_Iic_pred
            (Nat.find hGoodExists)
            hFirstPos

    rw [hSet]

    exact
      hInvert
        (Nat.find hGoodExists - 1)
        hPredFail

  · have hEmpty :
        UpperEndpointFailure
            (p : ℝ)
            upperCP
          =
        ∅ := by

      ext k

      change
        (upperCP k < (p : ℝ))
          ↔
        False

      constructor

      · intro hk

        exact
          hFailExists
            ⟨k, hk⟩

      · intro hFalse

        exact
          False.elim hFalse

    rw [hEmpty]

    simp

/--
The complete two-sided discrete Clopper-Pearson reduction.

Once the actual endpoint functions are shown to be monotone and to satisfy
the two binomial inversion properties, two-sided interval failure is bounded
by the allocated total failure probability.
-/
theorem binomial_countIntervalFailure_le_of_endpointInversion
    (m : ℕ)
    (p : unitInterval)
    (lowerCP upperCP : ℕ → ℝ)
    (αLower αUpper α : ENNReal)
    (hLowerMono :
      Monotone lowerCP)
    (hLowerInvert :
      LowerEndpointBinomialInversion
        m
        p
        lowerCP
        αLower)
    (hUpperMono :
      Monotone upperCP)
    (hUpperGoodExists :
      ∃ k : ℕ,
        (p : ℝ) ≤ upperCP k)
    (hUpperInvert :
      UpperEndpointBinomialInversion
        m
        p
        upperCP
        αUpper)
    (hBudget :
      αLower + αUpper ≤ α) :
    ProbabilityTheory.binomial m p
        (CountIntervalFailure
          (p : ℝ)
          lowerCP
          upperCP)
      ≤
    α := by

  apply
    binomial_countIntervalFailure_le_of_tailBounds
      m
      p
      lowerCP
      upperCP
      αLower
      αUpper
      α

  · exact
      lowerEndpointFailure_measure_le_of_inversion
        m
        p
        lowerCP
        αLower
        hLowerMono
        hLowerInvert

  · exact
      upperEndpointFailure_measure_le_of_inversion
        m
        p
        upperCP
        αUpper
        hUpperMono
        hUpperGoodExists
        hUpperInvert

  · exact hBudget

end DROSafety.DRO
