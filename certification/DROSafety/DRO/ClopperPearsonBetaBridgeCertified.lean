import DROSafety.DRO.BinomialBetaTailBridgeCertified
import DROSafety.DRO.ClopperPearsonBetaBridge
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Clopper-Pearson coverage with the certified binomial-Beta bridge

The original theorem in `ClopperPearsonBetaBridge.lean` required

    hBridge : BinomialBetaTailBridge m p

as an explicit analytic assumption.

That bridge has now been proved unconditionally in
`BinomialBetaTailBridgeCertified.lean`.

This file therefore exposes Clopper-Pearson coverage theorems whose only
remaining analytic assumption is

    BetaClopperPearsonEndpointSpec.

In particular, the binomial-Beta tail identity is no longer part of the
public theorem assumptions.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
Exact two-sided Clopper-Pearson failure bound using the certified
binomial-Beta bridge.

The only remaining endpoint-specific assumption is
`BetaClopperPearsonEndpointSpec`.
-/
theorem binomial_countIntervalFailure_le_of_certified_betaClopperPearson
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
    ProbabilityTheory.binomial
        m
        p
        (
          CountIntervalFailure
            (p : ℝ)
            lowerCP
            upperCP
        )
      ≤
    αLower + αUpper := by

  apply
    binomial_countIntervalFailure_le_of_betaClopperPearson

  all_goals
    first
    | exact
        certifiedBinomialBetaTailBridge
          m
          p
    | exact hSpec
    | exact le_rfl

/--
Budgeted form of the certified Clopper-Pearson failure theorem.

If the two one-sided tail budgets satisfy

    αLower + αUpper ≤ α,

then the total interval failure probability is at most `α`.
-/
theorem binomial_countIntervalFailure_le_of_certified_betaClopperPearson_budget
    (m : ℕ)
    (p : unitInterval)
    (lowerCP upperCP : ℕ → ℝ)
    (αLower αUpper α : ENNReal)
    (hSpec :
      BetaClopperPearsonEndpointSpec
        m
        lowerCP
        upperCP
        αLower
        αUpper)
    (hBudget :
      αLower + αUpper ≤ α) :
    ProbabilityTheory.binomial
        m
        p
        (
          CountIntervalFailure
            (p : ℝ)
            lowerCP
            upperCP
        )
      ≤
    α := by

  calc
    ProbabilityTheory.binomial
        m
        p
        (
          CountIntervalFailure
            (p : ℝ)
            lowerCP
            upperCP
        )
        ≤
      αLower + αUpper :=
        binomial_countIntervalFailure_le_of_certified_betaClopperPearson
          m
          p
          lowerCP
          upperCP
          αLower
          αUpper
          hSpec

    _ ≤ α :=
      hBudget

end DROSafety.DRO
