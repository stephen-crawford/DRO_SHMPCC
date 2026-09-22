import DROSafety.SingleMode.DeterministicSupport

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Scenario-Bound Monotonicity Interface

The deterministic-support scenario theorem currently assumes

    scenarioEpsilon S (supportSize z) β
      ≤ scenarioEpsilon S nBar β.

Operationally, SH-MPCC normally supplies the simpler fact

    supportSize z ≤ nBar.

This file isolates the remaining algebraic property needed to turn
that support-count bound into threshold domination.
-/

namespace DROSafety.SingleMode

/--
`scenarioEpsilon` is monotone in the support count up to `nBar`.
-/
def ScenarioEpsilonMonotoneUpTo
    (S nBar : ℕ)
    (β : ℝ) :
    Prop :=
  ∀ n,
    n ≤ nBar →
    DROSafety.scenarioEpsilon S n β ≤
      DROSafety.scenarioEpsilon S nBar β

/--
A pointwise upper bound on the realized support size implies
`SupportThresholdDominated`, provided `scenarioEpsilon` is monotone
over the relevant support-count range.
-/
theorem supportThresholdDominated_of_support_bound
    {S : ℕ}
    {Ξ : Type*}
    (supportSize : Sample S Ξ → ℕ)
    (nBar : ℕ)
    (β : ℝ)
    (hSupportBound :
      ∀ z,
        supportSize z ≤ nBar)
    (hMono :
      ScenarioEpsilonMonotoneUpTo
        S nBar β) :
    SupportThresholdDominated
      supportSize
      nBar
      β := by

  intro z

  exact
    hMono
      (supportSize z)
      (hSupportBound z)

/--
Deterministic-support single-mode scenario theorem stated using the
usual support-count upper bound.

The only remaining generic algebraic obligation is monotonicity of
`scenarioEpsilon` in the support count.
-/
theorem fixed_single_mode_scenario_bound_of_support_bound
    {S : ℕ}
    {Ξ Θ : Type*}
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
        P violates reconstruct)
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
        supportSize z ≤ nBar)
    (hMono :
      ScenarioEpsilonMonotoneUpTo
        S nBar β) :
    iidSampleMeasure P S
        (FixedScenarioFailure
          P
          violates
          solver
          nBar
          β)
      ≤
    ENNReal.ofReal β := by

  apply
    fixed_single_mode_scenario_bound
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

  exact
    supportThresholdDominated_of_support_bound
      supportSize
      nBar
      β
      hSupportBound
      hMono

end DROSafety.SingleMode
