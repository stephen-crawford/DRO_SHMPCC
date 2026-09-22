import DROSafety.SingleMode.SupportSizeUnion
import DROSafety.SingleMode.Compression

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Compression-Based Single-Mode Scenario Theorem

The previous files prove that the union of all fixed-support-size
bad events has probability at most `β`.

This file connects that union to an actual solver.

For every sample `z`, assume:

* `supportSize z < S`;
* the solver output has a compression certificate of exactly
  `supportSize z` samples.

Then a solver output whose true violation probability exceeds the
corresponding scenario bound must belong to the previously bounded
union event.

Therefore the solver failure probability is at most `β`.
-/

namespace DROSafety.SingleMode

/--
Failure event for a solver whose support size may depend on the
realized sample.

A sample is bad when the true violation probability of the solver
output exceeds the scenario bound associated with its realized
support size.
-/
def AdaptiveScenarioFailure
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (solver : Solver S Ξ Θ)
    (supportSize : Sample S Ξ → ℕ)
    (β : ℝ) :
    Set (Sample S Ξ) :=
  {z |
    ENNReal.ofReal
        (DROSafety.scenarioEpsilon
          S
          (supportSize z)
          β)
      <
    violationProbability
      P
      violates
      (solver z)}

/--
Any actual solver failure with a valid compression certificate belongs
to the union of all support-size bad events.
-/
theorem adaptiveScenarioFailure_subset_allSupportSizes
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (solver : Solver S Ξ Θ)
    (reconstruct : Reconstructor S Ξ Θ)
    (supportSize : Sample S Ξ → ℕ)
    (β : ℝ)
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
          z) :
    AdaptiveScenarioFailure
        P
        violates
        solver
        supportSize
        β
      ⊆
    AllSupportSizesBadEvent
      P
      violates
      reconstruct
      β := by

  intro z hz

  let n := supportSize z

  have hnS : n < S := by
    exact hSupportLt z

  have hCert :
      HasCompressionCertificate
        solver
        reconstruct
        violates
        n
        z := by
    exact hCompression z

  rcases hCert with
    ⟨I, hCard, hReconstruct, hConsistent⟩

  have hBad :
      ENNReal.ofReal
          (DROSafety.scenarioEpsilon S n β)
        <
      violationProbability
        P
        violates
        (solver z) := by
    exact hz

  have hI :
      I ∈ supportSets S n := by
    simp [supportSets, hCard]

  have hFixed :
      z ∈
        FixedSupportBadEvent
          P
          violates
          reconstruct
          I
          (DROSafety.scenarioEpsilon S n β) := by

    unfold FixedSupportBadEvent

    constructor

    · simpa [hReconstruct] using hBad

    · simpa [hReconstruct] using hConsistent

  have hSize :
      z ∈
        SupportSizeBadEvent
          P
          violates
          reconstruct
          n
          (DROSafety.scenarioEpsilon S n β) := by

    rw [
      supportSizeBadEvent_eq_union
        P
        violates
        reconstruct
        n
        (DROSafety.scenarioEpsilon S n β)
    ]

    simp only [Set.mem_iUnion]

    refine ⟨I, ?_⟩
    refine ⟨hI, ?_⟩

    exact hFixed

  unfold AllSupportSizesBadEvent

  simp only [Set.mem_iUnion]

  refine ⟨n, ?_⟩
  refine ⟨Finset.mem_range.mpr hnS, ?_⟩

  exact hSize

/--
Generic adaptive-support single-mode scenario theorem.

If every solver output admits a compression certificate whose support
size is strictly smaller than the sample count, then the probability
that its true violation probability exceeds its support-dependent
scenario bound is at most `β`.
-/
theorem adaptive_single_mode_scenario_bound
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (solver : Solver S Ξ Θ)
    (reconstruct : Reconstructor S Ξ Θ)
    (supportSize : Sample S Ξ → ℕ)
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
          z) :
    iidSampleMeasure P S
        (AdaptiveScenarioFailure
          P
          violates
          solver
          supportSize
          β)
      ≤
    ENNReal.ofReal β := by

  calc
    iidSampleMeasure P S
        (AdaptiveScenarioFailure
          P
          violates
          solver
          supportSize
          β)
      ≤
    iidSampleMeasure P S
        (AllSupportSizesBadEvent
          P
          violates
          reconstruct
          β) := by
      apply MeasureTheory.measure_mono

      exact
        adaptiveScenarioFailure_subset_allSupportSizes
          P
          violates
          solver
          reconstruct
          supportSize
          β
          hSupportLt
          hCompression

    _ ≤ ENNReal.ofReal β :=
      allSupportSizesBadEvent_measure_le
        P
        violates
        reconstruct
        β
        hβ0
        hβ1
        hReg

end DROSafety.SingleMode
