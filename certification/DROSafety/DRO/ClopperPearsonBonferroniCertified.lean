import DROSafety.DRO.ClopperPearsonBonferroni
import DROSafety.DRO.ClopperPearsonBinomialReductionCertified
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Certified Clopper-Pearson Bonferroni coverage

The original Bonferroni theorem assumed coordinate-wise
Clopper-Pearson coverage:

    ∀ i,
      μ (CoordinateIntervalFailure pTrue lower upper i) ≤ α i.

That assumption has now been discharged for the exact
Clopper-Pearson endpoints.

This file combines:

1. the exact scalar Clopper-Pearson theorem;
2. the binomial-count reduction;
3. the coordinate-wise coverage theorem; and
4. the existing Bonferroni union bound.

The only remaining statistical premise is that each coordinate count
has the appropriate binomial marginal law.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
Simultaneous exact Clopper-Pearson confidence-box coverage with
coordinate-dependent lower and upper tail budgets.

For each mode coordinate `i`, suppose

    count_i ~ Binomial(m, p_i),

and the interval endpoints are the certified exact Clopper-Pearson
endpoints with one-sided budgets

    αLower i
    αUpper i.

Then the probability that at least one coordinate interval misses its
true probability is bounded by

    ∑ᵢ (αLower i + αUpper i).

No coordinate-wise Clopper-Pearson coverage assumption remains.
-/
theorem confidenceBoxFailure_measure_le_exactClopperPearson
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (m : ℕ)
    (pTrue : ModeDistribution d)
    (pCoord : Fin d → unitInterval)
    (count : Ω → Fin d → ℕ)
    (lower upper : Ω → Fin d → ℝ)
    (αLower αUpper : Fin d → ENNReal)
    (hp :
      ∀ i : Fin d,
        pTrue i = ((pCoord i : unitInterval) : ℝ))
    (hLower :
      ∀ (ω : Ω) (i : Fin d),
        lower ω i
          =
        exactClopperPearsonLowerEndpointENNReal
          m
          (αLower i)
          (count ω i))
    (hUpper :
      ∀ (ω : Ω) (i : Fin d),
        upper ω i
          =
        exactClopperPearsonUpperEndpointENNReal
          m
          (αUpper i)
          (count ω i))
    (hLaw :
      ∀ i : Fin d,
        HasLaw
          (fun ω : Ω => count ω i)
          (ProbabilityTheory.binomial m (pCoord i))
          μ)
    (hLowerBudget :
      ∀ i : Fin d,
        αLower i ≤ 1)
    (hUpperBudget :
      ∀ i : Fin d,
        αUpper i ≤ 1) :
    μ
        (
          ConfidenceBoxFailure
            pTrue
            lower
            upper
        )
      ≤
    ∑' i : Fin d,
      (αLower i + αUpper i) := by

  apply
    confidenceBoxFailure_measure_le
      μ
      pTrue
      lower
      upper
      (fun i : Fin d =>
        αLower i + αUpper i)

  intro i

  exact
    coordinateIntervalFailure_measure_le_exactClopperPearson
      μ
      m
      pTrue
      i
      (pCoord i)
      (fun ω : Ω => count ω i)
      lower
      upper
      (αLower i)
      (αUpper i)
      (hp i)
      (fun ω : Ω =>
        hLower ω i)
      (fun ω : Ω =>
        hUpper ω i)
      (hLaw i)
      (hLowerBudget i)
      (hUpperBudget i)

/--
Budgeted simultaneous exact Clopper-Pearson confidence-box coverage.

If the sum of all coordinate tail budgets is at most `βDRO`, then

    μ (ConfidenceBoxFailure ...) ≤ βDRO.

This is the certified replacement for
`clopperPearsonBonferroni_failure_bound`: there is no
`hCoordinateCP` premise.
-/
theorem confidenceBoxFailure_measure_le_exactClopperPearson_budget
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (m : ℕ)
    (pTrue : ModeDistribution d)
    (pCoord : Fin d → unitInterval)
    (count : Ω → Fin d → ℕ)
    (lower upper : Ω → Fin d → ℝ)
    (αLower αUpper : Fin d → ENNReal)
    (βDRO : ENNReal)
    (hp :
      ∀ i : Fin d,
        pTrue i = ((pCoord i : unitInterval) : ℝ))
    (hLower :
      ∀ (ω : Ω) (i : Fin d),
        lower ω i
          =
        exactClopperPearsonLowerEndpointENNReal
          m
          (αLower i)
          (count ω i))
    (hUpper :
      ∀ (ω : Ω) (i : Fin d),
        upper ω i
          =
        exactClopperPearsonUpperEndpointENNReal
          m
          (αUpper i)
          (count ω i))
    (hLaw :
      ∀ i : Fin d,
        HasLaw
          (fun ω : Ω => count ω i)
          (ProbabilityTheory.binomial m (pCoord i))
          μ)
    (hLowerBudget :
      ∀ i : Fin d,
        αLower i ≤ 1)
    (hUpperBudget :
      ∀ i : Fin d,
        αUpper i ≤ 1)
    (hAllocation :
      (∑' i : Fin d,
        (αLower i + αUpper i))
        ≤ βDRO) :
    μ
        (
          ConfidenceBoxFailure
            pTrue
            lower
            upper
        )
      ≤
    βDRO := by

  exact
    confidenceBoxFailure_measure_le_beta
      μ
      pTrue
      lower
      upper
      (fun i : Fin d =>
        αLower i + αUpper i)
      βDRO
      (by
        intro i

        exact
          coordinateIntervalFailure_measure_le_exactClopperPearson
            μ
            m
            pTrue
            i
            (pCoord i)
            (fun ω : Ω => count ω i)
            lower
            upper
            (αLower i)
            (αUpper i)
            (hp i)
            (fun ω : Ω =>
              hLower ω i)
            (fun ω : Ω =>
              hUpper ω i)
            (hLaw i)
            (hLowerBudget i)
            (hUpperBudget i))
      hAllocation

/--
Equivalent direct proof of the budgeted result through the certified
simultaneous bound.

This form is useful downstream because it exposes the two logical
steps explicitly:

    exact coordinate CP
        ->
    Bonferroni sum
        ->
    total DRO budget.
-/
theorem clopperPearsonBonferroni_exact_failure_bound
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (m : ℕ)
    (pTrue : ModeDistribution d)
    (pCoord : Fin d → unitInterval)
    (count : Ω → Fin d → ℕ)
    (lower upper : Ω → Fin d → ℝ)
    (αLower αUpper : Fin d → ENNReal)
    (βDRO : ENNReal)
    (hp :
      ∀ i : Fin d,
        pTrue i = ((pCoord i : unitInterval) : ℝ))
    (hLower :
      ∀ (ω : Ω) (i : Fin d),
        lower ω i
          =
        exactClopperPearsonLowerEndpointENNReal
          m
          (αLower i)
          (count ω i))
    (hUpper :
      ∀ (ω : Ω) (i : Fin d),
        upper ω i
          =
        exactClopperPearsonUpperEndpointENNReal
          m
          (αUpper i)
          (count ω i))
    (hLaw :
      ∀ i : Fin d,
        HasLaw
          (fun ω : Ω => count ω i)
          (ProbabilityTheory.binomial m (pCoord i))
          μ)
    (hLowerBudget :
      ∀ i : Fin d,
        αLower i ≤ 1)
    (hUpperBudget :
      ∀ i : Fin d,
        αUpper i ≤ 1)
    (hAllocation :
      (∑' i : Fin d,
        (αLower i + αUpper i))
        ≤ βDRO) :
    μ
        (
          ConfidenceBoxFailure
            pTrue
            lower
            upper
        )
      ≤
    βDRO := by

  calc
    μ
        (
          ConfidenceBoxFailure
            pTrue
            lower
            upper
        )
        ≤
      ∑' i : Fin d,
        (αLower i + αUpper i) :=
          confidenceBoxFailure_measure_le_exactClopperPearson
            μ
            m
            pTrue
            pCoord
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

    _ ≤ βDRO :=
      hAllocation

end DROSafety.DRO
