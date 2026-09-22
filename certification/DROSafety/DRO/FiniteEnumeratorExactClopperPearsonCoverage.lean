import DROSafety.DRO.FiniteEnumeratorCombinedSafety
import DROSafety.DRO.ClopperPearsonBonferroniCertified
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Finite-enumerator coverage with exact Clopper-Pearson calibration

The existing finite-enumerator Wasserstein coverage theorem assumes
coordinate-wise Clopper-Pearson coverage:

    ∀ i,
      μ (CoordinateIntervalFailure pTrue lower upper i) ≤ α i.

That assumption has now been discharged by the exact
Clopper-Pearson development.

This file replaces it with the primitive statistical assumptions that

* each true coordinate is represented by a `unitInterval`;
* each observed coordinate count has its correct binomial marginal law;
* `lower` and `upper` are the exact Clopper-Pearson endpoints.

The resulting Wasserstein coverage theorem therefore no longer exposes
`hCoordinateCP`.

The same replacement is then propagated into the final combined
true-risk theorem.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
If `lower` is the exact Clopper-Pearson lower endpoint family and every
one-sided lower-tail budget is at most one, then every lower endpoint
is nonnegative.

This discharges the technical nonnegativity premise required by the
finite-enumerator radius construction.
-/
theorem exactClopperPearsonLowerFamily_nonneg
    {Ω : Type*}
    {d : ℕ}
    (m : ℕ)
    (count : Ω → Fin d → ℕ)
    (lower : Ω → Fin d → ℝ)
    (αLower : Fin d → ENNReal)
    (hLower :
      ∀ (ω : Ω) (i : Fin d),
        lower ω i
          =
        exactClopperPearsonLowerEndpointENNReal
          m
          (αLower i)
          (count ω i))
    (hLowerBudget :
      ∀ i : Fin d,
        αLower i ≤ 1) :
    ∀ (ω : Ω) (i : Fin d),
      0 ≤ lower ω i := by

  intro ω i

  rw [hLower ω i]

  unfold exactClopperPearsonLowerEndpointENNReal

  exact
    exactClopperPearsonLowerEndpoint_nonneg
      m
      (count ω i)
      (αLower i).toReal
      ENNReal.toReal_nonneg
      (
        ennreal_toReal_le_one
          (αLower i)
          (hLowerBudget i)
      )

/--
Finite-enumerator Wasserstein ambiguity-set coverage using the fully
certified exact Clopper-Pearson intervals.

No coordinate-wise CP coverage hypothesis remains.

The only statistical premise is

    count_i ~ Binomial(m, p_i)

for every coordinate `i`.
-/
theorem exactClopperPearson_finiteEnumeratorCoverageFailure_bound
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (pCoord : Fin d → unitInterval)
    (count : Ω → Fin d → ℕ)
    (lower upper : Ω → Fin d → ℝ)
    (data :
      (ω : Ω) →
        FiniteEnumeratorTransportData
          D
          (pNominal ω)
          (lower ω)
          (upper ω))
    (m : ℕ)
    (αLower αUpper : Fin d → ENNReal)
    (βDRO : ENNReal)
    (hd :
      0 < d)
    (hpTrue :
      IsProbabilityVector pTrue)
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
        ≤
      βDRO) :
    μ
        (
          CoverageFailure
            D
            pNominal
            pTrue
            (
              finiteEnumeratorRadiusFamily
                data
            )
        )
      ≤
    βDRO := by

  have hLowerNonneg :
      ∀ (ω : Ω) (i : Fin d),
        0 ≤ lower ω i :=
    exactClopperPearsonLowerFamily_nonneg
      m
      count
      lower
      αLower
      hLower
      hLowerBudget

  apply
    clopperPearson_finiteEnumeratorCoverageFailure_bound
      μ
      D
      pNominal
      pTrue
      lower
      upper
      data
      (fun i : Fin d =>
        αLower i + αUpper i)
      βDRO
      hd
      hLowerNonneg
      hpTrue

  · intro i

    exact
      coordinateIntervalFailure_measure_le_exactClopperPearson
        μ
        m
        pTrue
        i
        (pCoord i)
        (fun ω : Ω =>
          count ω i)
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

  · exact hAllocation

/--
Equivalent formulation emphasizing the ambiguity-set interpretation.

Except with probability at most `βDRO`, the true mode distribution lies
inside the finite Wasserstein ambiguity set whose radius is generated by
the exact Clopper-Pearson finite-enumerator construction.
-/
theorem exactClopperPearson_finiteEnumerator_trueDistributionCoverage
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (pCoord : Fin d → unitInterval)
    (count : Ω → Fin d → ℕ)
    (lower upper : Ω → Fin d → ℝ)
    (data :
      (ω : Ω) →
        FiniteEnumeratorTransportData
          D
          (pNominal ω)
          (lower ω)
          (upper ω))
    (m : ℕ)
    (αLower αUpper : Fin d → ENNReal)
    (βDRO : ENNReal)
    (hd :
      0 < d)
    (hpTrue :
      IsProbabilityVector pTrue)
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
        ≤
      βDRO) :
    μ
        (
          CoverageFailure
            D
            pNominal
            pTrue
            (
              finiteEnumeratorRadiusFamily
                data
            )
        )
      ≤
    βDRO := by

  exact
    exactClopperPearson_finiteEnumeratorCoverageFailure_bound
      μ
      D
      pNominal
      pTrue
      pCoord
      count
      lower
      upper
      data
      m
      αLower
      αUpper
      βDRO
      hd
      hpTrue
      hp
      hLower
      hUpper
      hLaw
      hLowerBudget
      hUpperBudget
      hAllocation

/--
Final combined true-risk theorem with exact Clopper-Pearson calibration.

Compared with
`combined_trueRiskViolation_bound_of_finiteEnumeratorClopperPearson`,
the abstract coordinate-wise CP hypothesis has disappeared.

The DRO calibration side now requires only the marginal binomial laws
of the observed counts.
-/
theorem combined_trueRiskViolation_bound_of_finiteEnumeratorExactClopperPearson
    {Ω : Type*}
    [MeasurableSpace Ω]
    {d : ℕ}
    (μ : Measure Ω)
    (D : FiniteGroundCost d)
    (pNominal : Ω → ModeDistribution d)
    (pTrue : ModeDistribution d)
    (pCoord : Fin d → unitInterval)
    (count : Ω → Fin d → ℕ)
    (qStar risk : Ω → ModeDistribution d)
    (lower upper : Ω → Fin d → ℝ)
    (data :
      (ω : Ω) →
        FiniteEnumeratorTransportData
          D
          (pNominal ω)
          (lower ω)
          (upper ω))
    (ε : Ω → ℝ)
    (m : ℕ)
    (αLower αUpper : Fin d → ENNReal)
    (βDRO βSH : ENNReal)
    (hd :
      0 < d)
    (hpTrue :
      IsProbabilityVector pTrue)
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
        ≤
      βDRO)
    (hWorst :
      ∀ ω : Ω,
        IsWorstCase
          (
            FiniteWassersteinAmbiguity
              D
              (pNominal ω)
              (
                finiteEnumeratorRadiusFamily
                  data
                  ω
              )
          )
          (risk ω)
          (qStar ω))
    (hSafeHorizon :
      μ
          (
            WorstCaseRiskViolation
              qStar
              risk
              ε
          )
        ≤
      βSH) :
    μ
        (
          TrueRiskViolation
            pTrue
            risk
            ε
        )
      ≤
    βDRO + βSH := by

  have hLowerNonneg :
      ∀ (ω : Ω) (i : Fin d),
        0 ≤ lower ω i :=
    exactClopperPearsonLowerFamily_nonneg
      m
      count
      lower
      αLower
      hLower
      hLowerBudget

  apply
    combined_trueRiskViolation_bound_of_finiteEnumeratorClopperPearson
      μ
      D
      pNominal
      pTrue
      qStar
      risk
      lower
      upper
      data
      ε
      (fun i : Fin d =>
        αLower i + αUpper i)
      βDRO
      βSH
      hd
      hLowerNonneg
      hpTrue

  · intro i

    exact
      coordinateIntervalFailure_measure_le_exactClopperPearson
        μ
        m
        pTrue
        i
        (pCoord i)
        (fun ω : Ω =>
          count ω i)
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

  · exact hAllocation

  · exact hWorst

  · exact hSafeHorizon

end DROSafety.DRO
