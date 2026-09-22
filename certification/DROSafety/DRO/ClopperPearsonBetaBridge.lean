import DROSafety.DRO.ClopperPearsonEndpointInversion
import Mathlib.Probability.Distributions.Beta
import Mathlib.Probability.Distributions.Binomial
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Beta-distribution bridge for exact Clopper-Pearson endpoints

The C++ implementation computes exact Clopper-Pearson endpoints through
Beta-distribution quantiles.

For `X ~ Binomial(m,p)`, the classical Beta/binomial identities are

    P_p[X >= k]
      =
    Beta(k, m-k+1) ((-∞, p])

for `1 <= k <= m`, and

    P_p[X <= k]
      =
    Beta(k+1, m-k) ([p, ∞))

for `k < m`.

The exact Clopper-Pearson endpoints satisfy

    Beta(k, m-k+1) ((-∞, L_k]) = alphaLower

and

    Beta(k+1, m-k) ([U_k, ∞)) = alphaUpper.

Therefore

    p < L_k
      =>
    P_p[X >= k] <= alphaLower,

and

    U_k < p
      =>
    P_p[X <= k] <= alphaUpper.

This file formalizes that order argument.

The remaining analytic obligations are isolated in
`BinomialBetaTailBridge` and `BetaClopperPearsonEndpointSpec`.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
The Beta/binomial tail identities required by the Clopper-Pearson proof.

These are mathematical identities, independent of the numerical
implementation used to evaluate Beta quantiles.
-/
structure BinomialBetaTailBridge
    (m : ℕ)
    (p : unitInterval) : Prop where

  /--
  For `1 <= k <= m`, the upper binomial tail is the lower Beta tail.
  -/
  upperBinomialTail :
    ∀ k : ℕ,
      0 < k →
      k ≤ m →
      ProbabilityTheory.binomial m p
          (Set.Ici k)
        =
      ProbabilityTheory.betaMeasure
          (k : ℝ)
          (m - k + 1 : ℝ)
          (Set.Iic (p : ℝ))

  /--
  For `k < m`, the lower binomial tail is the upper Beta tail.
  -/
  lowerBinomialTail :
    ∀ k : ℕ,
      k < m →
      ProbabilityTheory.binomial m p
          (Set.Iic k)
        =
      ProbabilityTheory.betaMeasure
          (k + 1 : ℝ)
          (m - k : ℝ)
          (Set.Ici (p : ℝ))

  /--
  Counts strictly larger than the number of trials have zero probability.
  -/
  upperTail_above_support :
    ∀ k : ℕ,
      m < k →
      ProbabilityTheory.binomial m p
          (Set.Ici k)
        =
      0

/--
Specification of exact two-sided Clopper-Pearson endpoints in Beta-tail
form.

For the lower endpoint:

* `L_0 = 0`;
* for `1 <= k <= m`,
  `Beta(k,m-k+1)((-∞,L_k]) = alphaLower`.

For the upper endpoint:

* for `k < m`,
  `Beta(k+1,m-k)([U_k,∞)) = alphaUpper`;
* for `k >= m`, `U_k = 1`.

The monotonicity properties are included because they are required by the
event-level endpoint-inversion theorem.
-/
structure BetaClopperPearsonEndpointSpec
    (m : ℕ)
    (lowerCP upperCP : ℕ → ℝ)
    (αLower αUpper : ENNReal) : Prop where

  lower_zero :
    lowerCP 0 = 0

  lower_quantile :
    ∀ k : ℕ,
      0 < k →
      k ≤ m →
      ProbabilityTheory.betaMeasure
          (k : ℝ)
          (m - k + 1 : ℝ)
          (Set.Iic (lowerCP k))
        =
      αLower

  upper_quantile :
    ∀ k : ℕ,
      k < m →
      ProbabilityTheory.betaMeasure
          (k + 1 : ℝ)
          (m - k : ℝ)
          (Set.Ici (upperCP k))
        =
      αUpper

  upper_one :
    ∀ k : ℕ,
      m ≤ k →
      upperCP k = 1

  lower_monotone :
    Monotone lowerCP

  upper_monotone :
    Monotone upperCP

/--
If `p < L_k`, then the Beta lower-tail probability at `p` is no larger
than the Beta lower-tail probability at `L_k`.
-/
theorem betaMeasure_Iic_mono_of_lt
    (a b p q : ℝ)
    (hpq :
      p < q) :
    ProbabilityTheory.betaMeasure
        a
        b
        (Set.Iic p)
      ≤
    ProbabilityTheory.betaMeasure
        a
        b
        (Set.Iic q) := by

  apply MeasureTheory.measure_mono

  intro x hx

  exact
    le_trans
      hx
      (le_of_lt hpq)

/--
If `U_k < p`, then the Beta upper-tail probability at `p` is no larger
than the Beta upper-tail probability at `U_k`.
-/
theorem betaMeasure_Ici_antitone_of_lt
    (a b p q : ℝ)
    (hqp :
      q < p) :
    ProbabilityTheory.betaMeasure
        a
        b
        (Set.Ici p)
      ≤
    ProbabilityTheory.betaMeasure
        a
        b
        (Set.Ici q) := by

  apply MeasureTheory.measure_mono

  intro x hx

  exact
    le_trans
      (le_of_lt hqp)
      hx

/--
The Beta lower-quantile specification and Beta/binomial tail identity imply
the lower-endpoint inversion property.
-/
theorem lowerEndpointBinomialInversion_of_betaSpec
    (m : ℕ)
    (p : unitInterval)
    (lowerCP upperCP : ℕ → ℝ)
    (αLower αUpper : ENNReal)
    (hBridge :
      BinomialBetaTailBridge
        m
        p)
    (hSpec :
      BetaClopperPearsonEndpointSpec
        m
        lowerCP
        upperCP
        αLower
        αUpper) :
    LowerEndpointBinomialInversion
      m
      p
      lowerCP
      αLower := by

  intro k hkFail

  by_cases hkZero :
      k = 0

  · subst k

    rw [hSpec.lower_zero] at hkFail

    have hpNonneg :
        0 ≤ (p : ℝ) :=
      p.property.1

    linarith

  · have hkPos :
        0 < k := by
      omega

    by_cases hkm :
        k ≤ m

    · have hTail :
          ProbabilityTheory.binomial m p
              (Set.Ici k)
            =
          ProbabilityTheory.betaMeasure
              (k : ℝ)
              (m - k + 1 : ℝ)
              (Set.Iic (p : ℝ)) :=
        hBridge.upperBinomialTail
          k
          hkPos
          hkm

      have hBetaMono :
          ProbabilityTheory.betaMeasure
              (k : ℝ)
              (m - k + 1 : ℝ)
              (Set.Iic (p : ℝ))
            ≤
          ProbabilityTheory.betaMeasure
              (k : ℝ)
              (m - k + 1 : ℝ)
              (Set.Iic (lowerCP k)) :=
        betaMeasure_Iic_mono_of_lt
          (k : ℝ)
          (m - k + 1 : ℝ)
          (p : ℝ)
          (lowerCP k)
          hkFail

      have hQuantile :
          ProbabilityTheory.betaMeasure
              (k : ℝ)
              (m - k + 1 : ℝ)
              (Set.Iic (lowerCP k))
            =
          αLower :=
        hSpec.lower_quantile
          k
          hkPos
          hkm

      calc
        ProbabilityTheory.binomial m p
            (Set.Ici k)
            =
          ProbabilityTheory.betaMeasure
            (k : ℝ)
            (m - k + 1 : ℝ)
            (Set.Iic (p : ℝ)) :=
          hTail

        _ ≤
          ProbabilityTheory.betaMeasure
            (k : ℝ)
            (m - k + 1 : ℝ)
            (Set.Iic (lowerCP k)) :=
          hBetaMono

        _ = αLower :=
          hQuantile

    · have hmk :
          m < k := by
        omega

      rw [
        hBridge.upperTail_above_support
          k
          hmk
      ]

      exact bot_le

/--
The Beta upper-quantile specification and Beta/binomial tail identity imply
the upper-endpoint inversion property.
-/
theorem upperEndpointBinomialInversion_of_betaSpec
    (m : ℕ)
    (p : unitInterval)
    (lowerCP upperCP : ℕ → ℝ)
    (αLower αUpper : ENNReal)
    (hBridge :
      BinomialBetaTailBridge
        m
        p)
    (hSpec :
      BetaClopperPearsonEndpointSpec
        m
        lowerCP
        upperCP
        αLower
        αUpper) :
    UpperEndpointBinomialInversion
      m
      p
      upperCP
      αUpper := by

  intro k hkFail

  by_cases hkm :
      k < m

  · have hTail :
        ProbabilityTheory.binomial m p
            (Set.Iic k)
          =
        ProbabilityTheory.betaMeasure
            (k + 1 : ℝ)
            (m - k : ℝ)
            (Set.Ici (p : ℝ)) :=
      hBridge.lowerBinomialTail
        k
        hkm

    have hBetaMono :
        ProbabilityTheory.betaMeasure
            (k + 1 : ℝ)
            (m - k : ℝ)
            (Set.Ici (p : ℝ))
          ≤
        ProbabilityTheory.betaMeasure
            (k + 1 : ℝ)
            (m - k : ℝ)
            (Set.Ici (upperCP k)) :=
      betaMeasure_Ici_antitone_of_lt
        (k + 1 : ℝ)
        (m - k : ℝ)
        (p : ℝ)
        (upperCP k)
        hkFail

    have hQuantile :
        ProbabilityTheory.betaMeasure
            (k + 1 : ℝ)
            (m - k : ℝ)
            (Set.Ici (upperCP k))
          =
        αUpper :=
      hSpec.upper_quantile
        k
        hkm

    calc
      ProbabilityTheory.binomial m p
          (Set.Iic k)
          =
        ProbabilityTheory.betaMeasure
          (k + 1 : ℝ)
          (m - k : ℝ)
          (Set.Ici (p : ℝ)) :=
        hTail

      _ ≤
        ProbabilityTheory.betaMeasure
          (k + 1 : ℝ)
          (m - k : ℝ)
          (Set.Ici (upperCP k)) :=
        hBetaMono

      _ = αUpper :=
        hQuantile

  · have hmk :
        m ≤ k := by
      omega

    have hUpperOne :
        upperCP k = 1 :=
      hSpec.upper_one
        k
        hmk

    rw [hUpperOne] at hkFail

    have hpLeOne :
        (p : ℝ) ≤ 1 :=
      p.property.2

    linarith

/--
The Beta endpoint specification automatically provides a count witnessing
the upper-endpoint nonfailure condition required by the general inversion
theorem.

Take `k = m`, where the exact Clopper-Pearson upper endpoint is `1`.
-/
theorem upperEndpoint_goodExists_of_betaSpec
    (m : ℕ)
    (p : unitInterval)
    (lowerCP upperCP : ℕ → ℝ)
    (αLower αUpper : ENNReal)
    (hSpec :
      BetaClopperPearsonEndpointSpec
        m
        lowerCP
        upperCP
        αLower
        αUpper) :
    ∃ k : ℕ,
      (p : ℝ) ≤ upperCP k := by

  refine ⟨m, ?_⟩

  rw [
    hSpec.upper_one
      m
      le_rfl
  ]

  exact
    p.property.2

/--
Main Beta-form Clopper-Pearson coverage theorem.

Once the Beta/binomial identities and exact endpoint quantile equations are
available, the two-sided binomial confidence failure probability is bounded
by the allocated total failure budget.
-/
theorem binomial_countIntervalFailure_le_of_betaClopperPearson
    (m : ℕ)
    (p : unitInterval)
    (lowerCP upperCP : ℕ → ℝ)
    (αLower αUpper α : ENNReal)
    (hBridge :
      BinomialBetaTailBridge
        m
        p)
    (hSpec :
      BetaClopperPearsonEndpointSpec
        m
        lowerCP
        upperCP
        αLower
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

  exact
    binomial_countIntervalFailure_le_of_endpointInversion
      m
      p
      lowerCP
      upperCP
      αLower
      αUpper
      α
      hSpec.lower_monotone
      (lowerEndpointBinomialInversion_of_betaSpec
        m
        p
        lowerCP
        upperCP
        αLower
        αUpper
        hBridge
        hSpec)
      hSpec.upper_monotone
      (upperEndpoint_goodExists_of_betaSpec
        m
        p
        lowerCP
        upperCP
        αLower
        αUpper
        hSpec)
      (upperEndpointBinomialInversion_of_betaSpec
        m
        p
        lowerCP
        upperCP
        αLower
        αUpper
        hBridge
        hSpec)
      hBudget

end DROSafety.DRO
