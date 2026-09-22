import DROSafety.DRO.ClopperPearsonExactQuantileENNReal
import DROSafety.DRO.ClopperPearsonBetaBridge
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Certified exact Clopper-Pearson endpoint specification

All six fields required by `BetaClopperPearsonEndpointSpec` have now
been proved:

1. lower endpoint is zero at count zero;
2. lower endpoint satisfies the exact lower Beta-tail equation;
3. upper endpoint satisfies the exact upper Beta-tail equation;
4. upper endpoint is one at and above the maximal count;
5. lower endpoint is monotone;
6. upper endpoint is monotone.

This file packages those results into a fully certified
`BetaClopperPearsonEndpointSpec`.

The tail budgets are represented as `ENNReal`, while the exact endpoint
functions use their real representatives via `toReal`.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory
open Set

/--
Exact lower Clopper-Pearson endpoint associated with an `ENNReal`
one-sided tail budget.
-/
noncomputable def exactClopperPearsonLowerEndpointENNReal
    (m : ℕ)
    (α : ENNReal) :
    ℕ → ℝ :=
  exactClopperPearsonLowerEndpoint
    m
    α.toReal

/--
Exact upper Clopper-Pearson endpoint associated with an `ENNReal`
one-sided tail budget.
-/
noncomputable def exactClopperPearsonUpperEndpointENNReal
    (m : ℕ)
    (α : ENNReal) :
    ℕ → ℝ :=
  exactClopperPearsonUpperEndpoint
    m
    α.toReal

/--
The lower exact endpoint at count zero is zero.
-/
theorem exactClopperPearsonLowerEndpointENNReal_zero
    (m : ℕ)
    (α : ENNReal) :
    exactClopperPearsonLowerEndpointENNReal
        m
        α
        0
      =
    0 := by

  unfold exactClopperPearsonLowerEndpointENNReal

  exact
    exactClopperPearsonLowerEndpoint_zero
      m
      α.toReal

/--
The exact upper endpoint is one at and above the maximal count.
-/
theorem exactClopperPearsonUpperEndpointENNReal_one
    (m k : ℕ)
    (α : ENNReal)
    (hmk :
      m ≤ k) :
    exactClopperPearsonUpperEndpointENNReal
        m
        α
        k
      =
    1 := by

  unfold exactClopperPearsonUpperEndpointENNReal

  exact
    exactClopperPearsonUpperEndpoint_eq_one
      m
      k
      α.toReal
      hmk

/--
If `α ≤ 1`, then the exact lower endpoint is monotone in the observed
count.
-/
theorem exactClopperPearsonLowerEndpointENNReal_monotone
    (m : ℕ)
    (α : ENNReal)
    (hα :
      α ≤ 1) :
    Monotone
      (
        exactClopperPearsonLowerEndpointENNReal
          m
          α
      ) := by

  unfold exactClopperPearsonLowerEndpointENNReal

  have hα0 :
      0 ≤ α.toReal :=
    ENNReal.toReal_nonneg

  have hα1 :
      α.toReal ≤ 1 :=
    ennreal_toReal_le_one
      α
      hα

  exact
    exactClopperPearsonLowerEndpoint_monotone
      m
      α.toReal
      hα0
      hα1

/--
If `α ≤ 1`, then the exact upper endpoint is monotone in the observed
count.
-/
theorem exactClopperPearsonUpperEndpointENNReal_monotone
    (m : ℕ)
    (α : ENNReal)
    (hα :
      α ≤ 1) :
    Monotone
      (
        exactClopperPearsonUpperEndpointENNReal
          m
          α
      ) := by

  unfold exactClopperPearsonUpperEndpointENNReal

  have hα0 :
      0 ≤ α.toReal :=
    ENNReal.toReal_nonneg

  have hα1 :
      α.toReal ≤ 1 :=
    ennreal_toReal_le_one
      α
      hα

  exact
    exactClopperPearsonUpperEndpoint_monotone
      m
      α.toReal
      hα0
      hα1

/--
The exact lower endpoint satisfies the required lower Beta-tail
quantile equation.
-/
theorem exactClopperPearsonLowerEndpointENNReal_quantile
    (m k : ℕ)
    (α : ENNReal)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hα :
      α ≤ 1) :
    ProbabilityTheory.betaMeasure
        (k : ℝ)
        (
          (m : ℝ)
            -
          (k : ℝ)
            +
          1
        )
        (
          Set.Iic
            (
              exactClopperPearsonLowerEndpointENNReal
                m
                α
                k
            )
        )
      =
    α := by

  unfold exactClopperPearsonLowerEndpointENNReal

  exact
    betaMeasure_Iic_exactLowerEndpoint
      m
      k
      α
      hk
      hkm
      hα

/--
The exact upper endpoint satisfies the required upper Beta-tail
quantile equation.
-/
theorem exactClopperPearsonUpperEndpointENNReal_quantile
    (m k : ℕ)
    (α : ENNReal)
    (hkm :
      k < m)
    (hα :
      α ≤ 1) :
    ProbabilityTheory.betaMeasure
        (
          (k : ℝ)
            +
          1
        )
        (
          (m : ℝ)
            -
          (k : ℝ)
        )
        (
          Set.Ici
            (
              exactClopperPearsonUpperEndpointENNReal
                m
                α
                k
            )
        )
      =
    α := by

  unfold exactClopperPearsonUpperEndpointENNReal

  exact
    betaMeasure_Ici_exactUpperEndpoint
      m
      k
      α
      hkm
      hα

/--
Fully certified exact Clopper-Pearson endpoint specification.

No Beta-quantile or monotonicity assumptions remain.

The only restrictions are the natural probability-budget conditions

    αLower ≤ 1
    αUpper ≤ 1.
-/
theorem certifiedBetaClopperPearsonEndpointSpec
    (m : ℕ)
    (αLower αUpper : ENNReal)
    (hLower :
      αLower ≤ 1)
    (hUpper :
      αUpper ≤ 1) :
    BetaClopperPearsonEndpointSpec
      m
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
      αLower
      αUpper := by

  refine
    {
      lower_zero := ?_
      lower_quantile := ?_
      upper_quantile := ?_
      upper_one := ?_
      lower_monotone := ?_
      upper_monotone := ?_
    }

  · exact
      exactClopperPearsonLowerEndpointENNReal_zero
        m
        αLower

  · intro k hk hkm

    exact
      exactClopperPearsonLowerEndpointENNReal_quantile
        m
        k
        αLower
        hk
        hkm
        hLower

  · intro k hkm

    exact
      exactClopperPearsonUpperEndpointENNReal_quantile
        m
        k
        αUpper
        hkm
        hUpper

  · intro k hmk

    exact
      exactClopperPearsonUpperEndpointENNReal_one
        m
        k
        αUpper
        hmk

  · exact
      exactClopperPearsonLowerEndpointENNReal_monotone
        m
        αLower
        hLower

  · exact
      exactClopperPearsonUpperEndpointENNReal_monotone
        m
        αUpper
        hUpper

end DROSafety.DRO
