import DROSafety.SingleMode.CompressionTheorem
import Mathlib.Data.ENNReal.Real

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Deterministic Support-Size Bound

The compression theorem uses the realized support size

    supportSize z.

The existing SH-MPCC development instead uses a deterministic support
bound `nBar`.

This file bridges those formulations. We explicitly require that the
scenario threshold corresponding to the realized support size is no
larger than the deterministic threshold.

Later, this condition can be discharged from an upper bound on the
realized support size once the relevant monotonicity property of
`scenarioEpsilon` has been proved.
-/

namespace DROSafety.SingleMode

/--
Failure event using a deterministic support-size bound `nBar`.
-/
def FixedScenarioFailure
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (solver : Solver S Ξ Θ)
    (nBar : ℕ)
    (β : ℝ) :
    Set (Sample S Ξ) :=
  {z |
    ENNReal.ofReal
        (DROSafety.scenarioEpsilon
          S nBar β)
      <
    violationProbability
      P
      violates
      (solver z)}

/--
The deterministic threshold dominates every realized-support
threshold.
-/
def SupportThresholdDominated
    {S : ℕ}
    {Ξ : Type*}
    (supportSize : Sample S Ξ → ℕ)
    (nBar : ℕ)
    (β : ℝ) :
    Prop :=
  ∀ z,
    DROSafety.scenarioEpsilon
        S
        (supportSize z)
        β
      ≤
    DROSafety.scenarioEpsilon
      S
      nBar
      β

/--
A failure relative to the deterministic support bound is also a
failure relative to the realized support size whenever the
deterministic threshold dominates the realized threshold.
-/
theorem fixedScenarioFailure_subset_adaptive
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (solver : Solver S Ξ Θ)
    (supportSize : Sample S Ξ → ℕ)
    (nBar : ℕ)
    (β : ℝ)
    (hThreshold :
      SupportThresholdDominated
        supportSize
        nBar
        β) :
    FixedScenarioFailure
        P
        violates
        solver
        nBar
        β
      ⊆
    AdaptiveScenarioFailure
      P
      violates
      solver
      supportSize
      β := by

  intro z hz

  change
    ENNReal.ofReal
        (DROSafety.scenarioEpsilon
          S nBar β)
      <
    violationProbability
      P violates (solver z)
    at hz

  change
    ENNReal.ofReal
        (DROSafety.scenarioEpsilon
          S (supportSize z) β)
      <
    violationProbability
      P violates (solver z)

  exact
    lt_of_le_of_lt
      (ENNReal.ofReal_le_ofReal
        (hThreshold z))
      hz

/--
Single-mode scenario guarantee with a deterministic support-size
threshold `nBar`.

This follows directly from the adaptive-support scenario theorem once
the deterministic threshold is known to dominate every realized
support-size threshold.
-/
theorem fixed_single_mode_scenario_bound
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
    (hThreshold :
      SupportThresholdDominated
        supportSize
        nBar
        β) :
    iidSampleMeasure P S
        (FixedScenarioFailure
          P
          violates
          solver
          nBar
          β)
      ≤
    ENNReal.ofReal β := by

  calc
    iidSampleMeasure P S
        (FixedScenarioFailure
          P
          violates
          solver
          nBar
          β)
      ≤
    iidSampleMeasure P S
        (AdaptiveScenarioFailure
          P
          violates
          solver
          supportSize
          β) := by

      apply MeasureTheory.measure_mono

      exact
        fixedScenarioFailure_subset_adaptive
          P
          violates
          solver
          supportSize
          nBar
          β
          hThreshold

    _ ≤ ENNReal.ofReal β :=
      adaptive_single_mode_scenario_bound
        P
        violates
        solver
        reconstruct
        supportSize
        β
        hβ0
        hβ1
        hReg
        hSupportLt
        hCompression

end DROSafety.SingleMode
