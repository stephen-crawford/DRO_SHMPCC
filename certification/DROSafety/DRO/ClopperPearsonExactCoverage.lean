import DROSafety.DRO.ClopperPearsonExactEndpointSpec
import DROSafety.DRO.ClopperPearsonBetaBridgeCertified
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Fully certified scalar Clopper-Pearson coverage

At this point both analytic assumptions used by the earlier
Clopper-Pearson development have been discharged:

* `BinomialBetaTailBridge` is certified;
* `BetaClopperPearsonEndpointSpec` is certified.

Therefore the exact Clopper-Pearson interval can now be given a scalar
binomial coverage theorem with no remaining Beta-tail or quantile
assumptions.

The only remaining hypotheses are ordinary probability-budget
conditions.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
The exact lower and upper endpoints have total failure probability at
most the sum of their one-sided budgets.

That is,

    P_p[p ∉ [L(X), U(X)]]
      ≤ αLower + αUpper.

No binomial-Beta bridge or endpoint-spec assumption remains.
-/
theorem binomial_countIntervalFailure_le_exactClopperPearson
    (m : ℕ)
    (p : unitInterval)
    (αLower αUpper : ENNReal)
    (hLower :
      αLower ≤ 1)
    (hUpper :
      αUpper ≤ 1) :
    ProbabilityTheory.binomial
        m
        p
        (
          CountIntervalFailure
            (p : ℝ)
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
        )
      ≤
    αLower + αUpper := by

  exact
    binomial_countIntervalFailure_le_of_certified_betaClopperPearson
      m
      p
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
      αUpper
      (
        certifiedBetaClopperPearsonEndpointSpec
          m
          αLower
          αUpper
          hLower
          hUpper
      )

/--
Budgeted version.

If

    αLower + αUpper ≤ α,

then the exact two-sided Clopper-Pearson failure probability is at most
`α`.
-/
theorem binomial_countIntervalFailure_le_exactClopperPearson_budget
    (m : ℕ)
    (p : unitInterval)
    (αLower αUpper α : ENNReal)
    (hLower :
      αLower ≤ 1)
    (hUpper :
      αUpper ≤ 1)
    (hBudget :
      αLower + αUpper ≤ α) :
    ProbabilityTheory.binomial
        m
        p
        (
          CountIntervalFailure
            (p : ℝ)
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
        )
        ≤
      αLower + αUpper :=
        binomial_countIntervalFailure_le_exactClopperPearson
          m
          p
          αLower
          αUpper
          hLower
          hUpper

    _ ≤ α :=
      hBudget

/--
Equal-tailed exact Clopper-Pearson coverage.

If each one-sided tail gets budget `αTail`, then the total failure
probability is at most `αTail + αTail`.
-/
theorem binomial_countIntervalFailure_le_exactClopperPearson_equalTails
    (m : ℕ)
    (p : unitInterval)
    (αTail : ENNReal)
    (hTail :
      αTail ≤ 1) :
    ProbabilityTheory.binomial
        m
        p
        (
          CountIntervalFailure
            (p : ℝ)
            (
              exactClopperPearsonLowerEndpointENNReal
                m
                αTail
            )
            (
              exactClopperPearsonUpperEndpointENNReal
                m
                αTail
            )
        )
      ≤
    αTail + αTail := by

  exact
    binomial_countIntervalFailure_le_exactClopperPearson
      m
      p
      αTail
      αTail
      hTail
      hTail

end DROSafety.DRO
