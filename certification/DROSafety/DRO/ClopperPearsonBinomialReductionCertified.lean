import DROSafety.DRO.ClopperPearsonBinomialReduction
import DROSafety.DRO.ClopperPearsonExactCoverage
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Certified reduction from binomial counts to coordinate CP coverage

`ClopperPearsonBinomialReduction.lean` reduced coordinate coverage to two
premises:

1. the observed count has the correct binomial law;
2. the scalar binomial Clopper-Pearson interval has the desired coverage.

The second premise has now been proved unconditionally for the exact
Clopper-Pearson endpoints.

This file therefore removes the scalar `hCoverage` assumption.

The only remaining probabilistic premise is the model statement

    HasLaw count (binomial m p) μ.

That premise will later be derived from the i.i.d./multinomial mode-count
model.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
Coordinate-level exact Clopper-Pearson coverage with separate one-sided
tail budgets.

If the observed coordinate count has law `Binomial(m,p)`, then

    μ(CoordinateIntervalFailure i)
      ≤ αLower + αUpper.

No scalar Clopper-Pearson coverage assumption remains.
-/
theorem coordinateIntervalFailure_measure_le_exactClopperPearson
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (m : ℕ)
    (pTrue : ModeDistribution d)
    (i : Fin d)
    (p : unitInterval)
    (count : Ω → ℕ)
    (lower upper : Ω → Fin d → ℝ)
    (αLower αUpper : ENNReal)
    (hp :
      pTrue i = (p : ℝ))
    (hLower :
      ∀ ω : Ω,
        lower ω i
          =
        exactClopperPearsonLowerEndpointENNReal
          m
          αLower
          (count ω))
    (hUpper :
      ∀ ω : Ω,
        upper ω i
          =
        exactClopperPearsonUpperEndpointENNReal
          m
          αUpper
          (count ω))
    (hLaw :
      HasLaw
        count
        (ProbabilityTheory.binomial m p)
        μ)
    (hLowerBudget :
      αLower ≤ 1)
    (hUpperBudget :
      αUpper ≤ 1) :
    μ
        (
          CoordinateIntervalFailure
            pTrue
            lower
            upper
            i
        )
      ≤
    αLower + αUpper := by

  apply
    coordinateIntervalFailure_measure_le_of_binomialCoverage
      μ
      m
      pTrue
      i
      p
      count
      (
        exactClopperPearsonLowerEndpointENNReal
          m
          αLower
      )
      (
        exactClopperPearsonUpperEndpointENNReal
          m
          αUpper
      )
      lower
      upper
      (αLower + αUpper)

  · exact hp

  · exact hLower

  · exact hUpper

  · exact hLaw

  · exact
      binomial_countIntervalFailure_le_exactClopperPearson
        m
        p
        αLower
        αUpper
        hLowerBudget
        hUpperBudget

/--
Budgeted coordinate-level form.

If

    αLower + αUpper ≤ α,

then the coordinate interval failure probability is at most `α`.
-/
theorem coordinateIntervalFailure_measure_le_exactClopperPearson_budget
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (m : ℕ)
    (pTrue : ModeDistribution d)
    (i : Fin d)
    (p : unitInterval)
    (count : Ω → ℕ)
    (lower upper : Ω → Fin d → ℝ)
    (αLower αUpper α : ENNReal)
    (hp :
      pTrue i = (p : ℝ))
    (hLower :
      ∀ ω : Ω,
        lower ω i
          =
        exactClopperPearsonLowerEndpointENNReal
          m
          αLower
          (count ω))
    (hUpper :
      ∀ ω : Ω,
        upper ω i
          =
        exactClopperPearsonUpperEndpointENNReal
          m
          αUpper
          (count ω))
    (hLaw :
      HasLaw
        count
        (ProbabilityTheory.binomial m p)
        μ)
    (hLowerBudget :
      αLower ≤ 1)
    (hUpperBudget :
      αUpper ≤ 1)
    (hBudget :
      αLower + αUpper ≤ α) :
    μ
        (
          CoordinateIntervalFailure
            pTrue
            lower
            upper
            i
        )
      ≤
    α := by

  calc
    μ
        (
          CoordinateIntervalFailure
            pTrue
            lower
            upper
            i
        )
        ≤
      αLower + αUpper :=
        coordinateIntervalFailure_measure_le_exactClopperPearson
          μ
          m
          pTrue
          i
          p
          count
          lower
          upper
          αLower
          αUpper
          hp
          hLower
          hUpper
          hLaw
          hLowerBudget
          hUpperBudget

    _ ≤ α :=
      hBudget

/--
Equal-tailed coordinate-level exact Clopper-Pearson coverage.

Each one-sided tail receives the same budget `αTail`.
-/
theorem coordinateIntervalFailure_measure_le_exactClopperPearson_equalTails
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (m : ℕ)
    (pTrue : ModeDistribution d)
    (i : Fin d)
    (p : unitInterval)
    (count : Ω → ℕ)
    (lower upper : Ω → Fin d → ℝ)
    (αTail : ENNReal)
    (hp :
      pTrue i = (p : ℝ))
    (hLower :
      ∀ ω : Ω,
        lower ω i
          =
        exactClopperPearsonLowerEndpointENNReal
          m
          αTail
          (count ω))
    (hUpper :
      ∀ ω : Ω,
        upper ω i
          =
        exactClopperPearsonUpperEndpointENNReal
          m
          αTail
          (count ω))
    (hLaw :
      HasLaw
        count
        (ProbabilityTheory.binomial m p)
        μ)
    (hTail :
      αTail ≤ 1) :
    μ
        (
          CoordinateIntervalFailure
            pTrue
            lower
            upper
            i
        )
      ≤
    αTail + αTail := by

  exact
    coordinateIntervalFailure_measure_le_exactClopperPearson
      μ
      m
      pTrue
      i
      p
      count
      lower
      upper
      αTail
      αTail
      hp
      hLower
      hUpper
      hLaw
      hTail
      hTail

end DROSafety.DRO
