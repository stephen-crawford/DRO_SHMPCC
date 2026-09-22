import DROSafety.SingleMode.ScenarioMonotonicity
import DROSafety.SingleMode.ScenarioMonotonicityInterior
import DROSafety.SingleMode.ScenarioMonotonicityStep
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Certified scenario-epsilon monotonicity corollaries

This file closes the conditional monotonicity interface used by the
single-mode deterministic-support theorem.

For `0 ≤ β ≤ 1`, the previously proved monotonicity theorem discharges
`ScenarioEpsilonMonotoneUpTo`, so callers only need the ordinary
support-count bound.
-/

namespace DROSafety.SingleMode

/--
A pointwise support-count upper bound implies
`SupportThresholdDominated` for a valid confidence parameter.
-/
theorem supportThresholdDominated_of_support_bound_of_confidence
    {S : ℕ}
    {Ξ : Type}
    (supportSize : Sample S Ξ → ℕ)
    (nBar : ℕ)
    (β : ℝ)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1)
    (hSupportBound :
      ∀ z,
        supportSize z ≤ nBar) :
    SupportThresholdDominated
      supportSize
      nBar
      β := by

    exact
    supportThresholdDominated_of_support_bound
      supportSize
      nBar
      β
      hSupportBound
      (scenarioEpsilonMonotoneUpTo_of_confidence
        S nBar β hβ0 hβ1)


/--
Deterministic-support single-mode scenario theorem using only the usual
support-count upper bound.

The former `ScenarioEpsilonMonotoneUpTo` assumption is now derived from
`0 ≤ β ≤ 1`.
-/
theorem fixed_single_mode_scenario_bound_of_support_bound_of_confidence
    {S : ℕ}
    {Ξ : Type}
    {Θ : Type}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (solver : Solver S Ξ Θ)
    (reconstruct : Reconstructor S Ξ Θ)
    (supportSize : Sample S Ξ → ℕ)
    (nBar : ℕ)
    (β : ℝ)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1)
    (hReg :
      ScenarioRegularity
        P
        violates
        reconstruct)
    (hSupportLt :
      ∀ z,
        supportSize z < S)
    (hCompression :
      ∀ z,
        HasCompressionCertificate
          solver
          reconstruct
          violates
          (supportSize z)
          z)
    (hSupportBound :
      ∀ z,
        supportSize z ≤ nBar) :
    iidSampleMeasure P S
        (FixedScenarioFailure
          P
          violates
          solver
          nBar
          β)
      ≤
    ENNReal.ofReal β := by

    exact
    fixed_single_mode_scenario_bound_of_support_bound
      P
      violates
      solver
      reconstruct
      supportSize
      nBar
      β
      hβ0
      hβ1
      hReg
      hSupportLt
      hCompression
      hSupportBound
      (scenarioEpsilonMonotoneUpTo_of_interior
        S
        nBar
        β
        hβ0
        (scenarioEpsilonInteriorStepMonotone_of_confidence
          S β hβ0 hβ1))

/--
Deterministic-support scenario certificate from the practically available
support bound `supportSize z ≤ nBar`.

The technical hypothesis `supportSize z < S` is derived automatically from
`nBar < S`, so callers no longer need to provide it separately.
-/
theorem fixed_single_mode_scenario_bound_certified
    {S : ℕ}
    {Ξ : Type}
    {Θ : Type}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (solver : Solver S Ξ Θ)
    (reconstruct : Reconstructor S Ξ Θ)
    (supportSize : Sample S Ξ → ℕ)
    (nBar : ℕ)
    (β : ℝ)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1)
    (hnBarLtS : nBar < S)
    (hReg :
      ScenarioRegularity
        P
        violates
        reconstruct)
    (hCompression :
      ∀ z,
        HasCompressionCertificate
          solver
          reconstruct
          violates
          (supportSize z)
          z)
    (hSupportBound :
      ∀ z,
        supportSize z ≤ nBar) :
    iidSampleMeasure P S
        (FixedScenarioFailure
          P
          violates
          solver
          nBar
          β)
      ≤
    ENNReal.ofReal β := by

  have hSupportLt :
      ∀ z,
        supportSize z < S := by
    intro z
    exact
      lt_of_le_of_lt
        (hSupportBound z)
        hnBarLtS

  exact
    fixed_single_mode_scenario_bound_of_support_bound_of_confidence
      P
      violates
      solver
      reconstruct
      supportSize
      nBar
      β
      hβ0
      hβ1
      hReg
      hSupportLt
      hCompression
      hSupportBound

end DROSafety.SingleMode
