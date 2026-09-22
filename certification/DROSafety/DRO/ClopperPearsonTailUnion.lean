import DROSafety.DRO.ClopperPearsonBinomialReduction
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Clopper-Pearson failure as a union of two tails

For deterministic interval functions

    lowerCP : ℕ → ℝ
    upperCP : ℕ → ℝ,

the interval fails to contain the true scalar parameter `p` exactly when

    p < lowerCP k

or

    upperCP k < p.

Thus the total interval-failure probability is bounded by the sum of the
two one-sided tail probabilities.

This isolates the final analytic Clopper-Pearson obligation:

    Bin(m,p) {k | p < lowerCP k} ≤ αLower

and

    Bin(m,p) {k | upperCP k < p} ≤ αUpper.

Once those are proved for the actual Clopper-Pearson endpoints, the
existing Bonferroni proof gives simultaneous multinomial-coordinate
coverage.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
Failure through the lower confidence endpoint:

    p < lowerCP(k).
-/
def LowerEndpointFailure
    (p : ℝ)
    (lowerCP : ℕ → ℝ) :
    Set ℕ :=
  {
    k |
      p < lowerCP k
  }

/--
Failure through the upper confidence endpoint:

    upperCP(k) < p.
-/
def UpperEndpointFailure
    (p : ℝ)
    (upperCP : ℕ → ℝ) :
    Set ℕ :=
  {
    k |
      upperCP k < p
  }

/--
Interval failure is exactly the union of the lower-endpoint and
upper-endpoint failure events.
-/
theorem countIntervalFailure_eq_tail_union
    (p : ℝ)
    (lowerCP upperCP : ℕ → ℝ) :
    CountIntervalFailure
        p
        lowerCP
        upperCP
      =
    LowerEndpointFailure
        p
        lowerCP
      ∪
    UpperEndpointFailure
        p
        upperCP := by

  ext k

  change
    (¬ (
      lowerCP k ≤ p ∧
      p ≤ upperCP k
    ))
      ↔
    (
      p < lowerCP k ∨
      upperCP k < p
    )

  constructor

  · intro hFail

    by_cases hLower :
        lowerCP k ≤ p

    · right

      have hNotUpper :
          ¬ p ≤ upperCP k := by

        intro hUpper

        exact
          hFail
            ⟨hLower, hUpper⟩

      exact
        lt_of_not_ge
          hNotUpper

    · left

      exact
        lt_of_not_ge
          hLower

  · intro hTail

    intro hInside

    rcases hTail with
      hLowerFail | hUpperFail

    · exact
        (not_lt_of_ge
          hInside.1)
          hLowerFail

    · exact
        (not_lt_of_ge
          hInside.2)
          hUpperFail

/--
The binomial probability of interval failure is bounded by the sum of the
two one-sided failure probabilities.
-/
theorem binomial_countIntervalFailure_le_tail_sum
    (m : ℕ)
    (p : unitInterval)
    (lowerCP upperCP : ℕ → ℝ) :
    ProbabilityTheory.binomial m p
        (CountIntervalFailure
          (p : ℝ)
          lowerCP
          upperCP)
      ≤
    ProbabilityTheory.binomial m p
        (LowerEndpointFailure
          (p : ℝ)
          lowerCP)
      +
    ProbabilityTheory.binomial m p
        (UpperEndpointFailure
          (p : ℝ)
          upperCP) := by

  rw [
    countIntervalFailure_eq_tail_union
  ]

  exact
    MeasureTheory.measure_union_le
      (LowerEndpointFailure
        (p : ℝ)
        lowerCP)
      (UpperEndpointFailure
        (p : ℝ)
        upperCP)

/--
Two one-sided binomial bounds, together with a failure-budget allocation,
imply the desired two-sided interval-coverage bound.
-/
theorem binomial_countIntervalFailure_le_of_tailBounds
    (m : ℕ)
    (p : unitInterval)
    (lowerCP upperCP : ℕ → ℝ)
    (αLower αUpper α : ENNReal)
    (hLower :
      ProbabilityTheory.binomial m p
          (LowerEndpointFailure
            (p : ℝ)
            lowerCP)
        ≤
      αLower)
    (hUpper :
      ProbabilityTheory.binomial m p
          (UpperEndpointFailure
            (p : ℝ)
            upperCP)
        ≤
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

  calc
    ProbabilityTheory.binomial m p
        (CountIntervalFailure
          (p : ℝ)
          lowerCP
          upperCP)
      ≤
    ProbabilityTheory.binomial m p
        (LowerEndpointFailure
          (p : ℝ)
          lowerCP)
      +
    ProbabilityTheory.binomial m p
        (UpperEndpointFailure
          (p : ℝ)
          upperCP) := by

      exact
        binomial_countIntervalFailure_le_tail_sum
          m
          p
          lowerCP
          upperCP

    _ ≤ αLower + αUpper := by

      exact
        add_le_add
          hLower
          hUpper

    _ ≤ α := hBudget

/--
Sampling-space version of the two-tail theorem.
-/
theorem sampleIntervalFailure_measure_le_of_binomialTailBounds
    {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    (m : ℕ)
    (p : unitInterval)
    (count : Ω → ℕ)
    (lowerCP upperCP : ℕ → ℝ)
    (αLower αUpper α : ENNReal)
    (hLaw :
      HasLaw
        count
        (ProbabilityTheory.binomial m p)
        μ)
    (hLower :
      ProbabilityTheory.binomial m p
          (LowerEndpointFailure
            (p : ℝ)
            lowerCP)
        ≤
      αLower)
    (hUpper :
      ProbabilityTheory.binomial m p
          (UpperEndpointFailure
            (p : ℝ)
            upperCP)
        ≤
      αUpper)
    (hBudget :
      αLower + αUpper ≤ α) :
    μ
        (SampleIntervalFailure
          count
          (p : ℝ)
          lowerCP
          upperCP)
      ≤
    α := by

  apply
    sampleIntervalFailure_measure_le_of_binomialCoverage
      μ
      m
      p
      count
      lowerCP
      upperCP
      α
      hLaw

  exact
    binomial_countIntervalFailure_le_of_tailBounds
      m
      p
      lowerCP
      upperCP
      αLower
      αUpper
      α
      hLower
      hUpper
      hBudget

/--
Coordinate-level version matching the premise used by
`ClopperPearsonBonferroni.lean`.

Once the two Clopper-Pearson tail inequalities are available, this theorem
directly produces the required bound on `CoordinateIntervalFailure`.
-/
theorem coordinateIntervalFailure_measure_le_of_binomialTailBounds
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
    (αLower αUpper α : ENNReal)
    (hp :
      pTrue i = (p : ℝ))
    (hLowerEndpoint :
      ∀ ω : Ω,
        lower ω i =
          lowerCP (count ω))
    (hUpperEndpoint :
      ∀ ω : Ω,
        upper ω i =
          upperCP (count ω))
    (hLaw :
      HasLaw
        count
        (ProbabilityTheory.binomial m p)
        μ)
    (hLowerTail :
      ProbabilityTheory.binomial m p
          (LowerEndpointFailure
            (p : ℝ)
            lowerCP)
        ≤
      αLower)
    (hUpperTail :
      ProbabilityTheory.binomial m p
          (UpperEndpointFailure
            (p : ℝ)
            upperCP)
        ≤
      αUpper)
    (hBudget :
      αLower + αUpper ≤ α) :
    μ
        (CoordinateIntervalFailure
          pTrue
          lower
          upper
          i)
      ≤
    α := by

  apply
    coordinateIntervalFailure_measure_le_of_binomialCoverage
      μ
      m
      pTrue
      i
      p
      count
      lowerCP
      upperCP
      lower
      upper
      α
      hp
      hLowerEndpoint
      hUpperEndpoint
      hLaw

  exact
    binomial_countIntervalFailure_le_of_tailBounds
      m
      p
      lowerCP
      upperCP
      αLower
      αUpper
      α
      hLowerTail
      hUpperTail
      hBudget

end DROSafety.DRO
