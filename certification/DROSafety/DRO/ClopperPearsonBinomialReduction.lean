import DROSafety.DRO.CertifiedRadiusCoverage
import Mathlib.Probability.Distributions.Binomial
import Mathlib.Probability.HasLaw
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Reduction of Clopper-Pearson coverage to a binomial-law statement

For one mode coordinate, let

    X : Ω → ℕ

be the observed count among `m` trials.

If

    X ~ Binomial(m, p),

and the interval endpoints are functions of the observed count,

    lowerCP : ℕ → ℝ
    upperCP : ℕ → ℝ,

then the probability of coordinate-interval failure under the sampling
measure is exactly the binomial probability mass of the set of counts for
which the interval fails to contain the true parameter.

This file therefore reduces the remaining Clopper-Pearson proof obligation
to the purely discrete statement

    Bin(m,p) { k | ¬(lowerCP k ≤ p ∧ p ≤ upperCP k) } ≤ α.

The next file will prove that inequality from the two one-sided
Clopper-Pearson tail conditions.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
Set of observed counts for which the confidence interval fails to contain
the scalar Bernoulli/binomial parameter `p`.
-/
def CountIntervalFailure
    (p : ℝ)
    (lowerCP upperCP : ℕ → ℝ) :
    Set ℕ :=
  {
    k |
      ¬ (
        lowerCP k ≤ p ∧
        p ≤ upperCP k
      )
  }

/--
Sampling-space event corresponding to `CountIntervalFailure`.
-/
def SampleIntervalFailure
    {Ω : Type*}
    (count : Ω → ℕ)
    (p : ℝ)
    (lowerCP upperCP : ℕ → ℝ) :
    Set Ω :=
  {
    ω |
      ¬ (
        lowerCP (count ω) ≤ p ∧
        p ≤ upperCP (count ω)
      )
  }

/--
Every set of natural-number counts is measurable.
-/
theorem countIntervalFailure_measurable
    (p : ℝ)
    (lowerCP upperCP : ℕ → ℝ) :
    MeasurableSet
      (CountIntervalFailure
        p
        lowerCP
        upperCP) := by

  exact
    (Set.to_countable
      (CountIntervalFailure
        p
        lowerCP
        upperCP)).measurableSet

/--
If `count` has the binomial law, then the sampling probability of interval
failure equals the binomial mass of the bad-count set.
-/
theorem sampleIntervalFailure_measure_eq_binomial
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    (m : ℕ)
    (p : unitInterval)
    (count : Ω → ℕ)
    (lowerCP upperCP : ℕ → ℝ)
    (hLaw :
      HasLaw
        count
        (ProbabilityTheory.binomial m p)
        μ) :
    μ
        (SampleIntervalFailure
          count
          (p : ℝ)
          lowerCP
          upperCP)
      =
    ProbabilityTheory.binomial m p
      (CountIntervalFailure
        (p : ℝ)
        lowerCP
        upperCP) := by

  have hMeas :
      MeasurableSet
        (CountIntervalFailure
          (p : ℝ)
          lowerCP
          upperCP) :=
    countIntervalFailure_measurable
      (p : ℝ)
      lowerCP
      upperCP

  simpa [
    SampleIntervalFailure,
    CountIntervalFailure
  ] using
    hLaw.measure_eq hMeas

/--
A binomial bound on the bad-count set transfers immediately to the
sampling-space interval-failure probability.
-/
theorem sampleIntervalFailure_measure_le_of_binomialCoverage
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    (m : ℕ)
    (p : unitInterval)
    (count : Ω → ℕ)
    (lowerCP upperCP : ℕ → ℝ)
    (α : ENNReal)
    (hLaw :
      HasLaw
        count
        (ProbabilityTheory.binomial m p)
        μ)
    (hCoverage :
      ProbabilityTheory.binomial m p
          (CountIntervalFailure
            (p : ℝ)
            lowerCP
            upperCP)
        ≤
      α) :
    μ
        (SampleIntervalFailure
          count
          (p : ℝ)
          lowerCP
          upperCP)
      ≤
    α := by

  rw [
    sampleIntervalFailure_measure_eq_binomial
      μ
      m
      p
      count
      lowerCP
      upperCP
      hLaw
  ]

  exact hCoverage

/--
If the random interval endpoints for coordinate `i` are obtained by applying
`lowerCP` and `upperCP` to the corresponding count, then the existing
`CoordinateIntervalFailure` event is exactly `SampleIntervalFailure`.
-/
theorem coordinateIntervalFailure_eq_sampleIntervalFailure
    {Ω : Type*}
    {d : ℕ}
    (pTrue : ModeDistribution d)
    (i : Fin d)
    (p : unitInterval)
    (count : Ω → ℕ)
    (lowerCP upperCP : ℕ → ℝ)
    (lower upper : Ω → Fin d → ℝ)
    (hp :
      pTrue i = (p : ℝ))
    (hLower :
      ∀ ω : Ω,
        lower ω i =
          lowerCP (count ω))
    (hUpper :
      ∀ ω : Ω,
        upper ω i =
          upperCP (count ω)) :
    CoordinateIntervalFailure
        pTrue
        lower
        upper
        i
      =
    SampleIntervalFailure
      count
      (p : ℝ)
      lowerCP
      upperCP := by

  ext ω

  simp only [
    CoordinateIntervalFailure,
    SampleIntervalFailure,
    Set.mem_ofPred_eq
  ]

  rw [
    hLower ω,
    hUpper ω,
    hp
  ]

/--
Main reduction theorem for one mode coordinate.

To prove the per-coordinate premise currently used by
`ClopperPearsonBonferroni.lean`, it is sufficient to prove:

1. the coordinate count has the binomial law;
2. the deterministic Clopper-Pearson interval has binomial failure
   probability at most `α`.
-/
theorem coordinateIntervalFailure_measure_le_of_binomialCoverage
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (m : ℕ)
    (pTrue : ModeDistribution d)
    (i : Fin d)
    (p : unitInterval)
    (count : Ω → ℕ)
    (lowerCP upperCP : ℕ → ℝ)
    (lower upper : Ω → Fin d → ℝ)
    (α : ENNReal)
    (hp :
      pTrue i = (p : ℝ))
    (hLower :
      ∀ ω : Ω,
        lower ω i =
          lowerCP (count ω))
    (hUpper :
      ∀ ω : Ω,
        upper ω i =
          upperCP (count ω))
    (hLaw :
      HasLaw
        count
        (ProbabilityTheory.binomial m p)
        μ)
    (hCoverage :
      ProbabilityTheory.binomial m p
          (CountIntervalFailure
            (p : ℝ)
            lowerCP
            upperCP)
        ≤
      α) :
    μ
        (CoordinateIntervalFailure
          pTrue
          lower
          upper
          i)
      ≤
    α := by

  rw [
    coordinateIntervalFailure_eq_sampleIntervalFailure
      pTrue
      i
      p
      count
      lowerCP
      upperCP
      lower
      upper
      hp
      hLower
      hUpper
  ]

  exact
    sampleIntervalFailure_measure_le_of_binomialCoverage
      μ
      m
      p
      count
      lowerCP
      upperCP
      α
      hLaw
      hCoverage

end DROSafety.DRO
