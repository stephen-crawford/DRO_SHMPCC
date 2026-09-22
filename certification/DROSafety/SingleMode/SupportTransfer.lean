import DROSafety.SingleMode.FixedSupport
import DROSafety.SingleMode.SupportIntegral

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Transfer Back to the Full Sample Space

The fixed-support probability bound has been proved on the product space

    support coordinates × outside coordinates.

This file proves that the original fixed-support bad event is exactly
the preimage of the split bad event under `splitSampleEquiv`.

Since the split is measure preserving, the product-space probability
bound transfers directly back to the original IID sample space.
-/

namespace DROSafety.SingleMode

/--
Splitting a full sample gives exactly its support and outside parts.
-/
theorem splitSampleEquiv_apply
    {S : ℕ}
    {Ξ : Type*}
    [MeasurableSpace Ξ]
    (I : Finset (Fin S))
    (z : Sample S Ξ) :
    splitSampleEquiv (Ξ := Ξ) I z =
      (supportPart I z, outsidePart I z) := by

  change
    ((fun x : SupportIndex I => z x.1),
     (fun x : OutsideIndex I => z x.1)) =
      (supportPart I z, outsidePart I z)

  apply Prod.ext

  · funext x
    rfl

  · funext x
    rfl

/--
The original fixed-support bad event is exactly the preimage of the
split product-space bad event.
-/
theorem fixedSupportBadEvent_eq_preimage_split
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (ε : ℝ) :
    FixedSupportBadEvent
        P violates reconstruct I ε =
      (splitSampleEquiv (Ξ := Ξ) I) ⁻¹'
        SplitSupportBadEvent
          P violates reconstruct I ε := by
  ext z

  simp only [
    FixedSupportBadEvent,
    SplitSupportBadEvent,
    SupportFiberBadEvent,
    Set.mem_setOf_eq,
    Set.mem_preimage,
    splitSampleEquiv_apply,
    Prod.fst,
    Prod.snd
  ]

  rw [
    reconstructedDecision_eq_supportDecision
      reconstruct I z
  ]

  rw [
    consistentOutside_iff_outsideSafe
      violates
      I
      z
      (supportDecision
        reconstruct
        I
        (supportPart I z))
  ]

/--
The original fixed-support bad event has probability at most

    (1 - ε)^(S - |I|).

This is the complete fixed-support scenario probability bound,
conditional only on measurability of the split bad event.
-/
theorem fixedSupportBadEvent_measure_le
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (ε : ℝ)
    (hε : 0 ≤ ε)
    (hViolationMeas :
      ViolationSetsMeasurable violates)
    (hSplitMeas :
      MeasurableSet
        (SplitSupportBadEvent
          P violates reconstruct I ε)) :
    iidSampleMeasure P S
        (FixedSupportBadEvent
          P violates reconstruct I ε)
      ≤
    (ENNReal.ofReal (1 - ε)) ^
      (S - I.card) := by

  rw [
    fixedSupportBadEvent_eq_preimage_split
      P
      violates
      reconstruct
      I
      ε
  ]

  have hPreserving :=
    splitSample_measurePreserving
      P
      I

  have hMeasureEq :
      iidSampleMeasure P S
          ((splitSampleEquiv (Ξ := Ξ) I) ⁻¹'
            SplitSupportBadEvent
              P violates reconstruct I ε)
        =
      ((iidPredicateMeasure
          P
          (supportPredicate I)).prod
        (iidPredicateMeasure
          P
          (fun i : Fin S =>
            ¬ supportPredicate I i)))
        (SplitSupportBadEvent
          P violates reconstruct I ε) := by
    exact
      hPreserving.measure_preimage_equiv
        (SplitSupportBadEvent
          P violates reconstruct I ε)

  rw [hMeasureEq]

  exact
    splitSupportBadEvent_measure_le
      P
      violates
      reconstruct
      I
      ε
      hε
      hViolationMeas
      hSplitMeas

/--
The previously defined proposition `FixedSupportProbabilityBound`
follows from the fixed-support theorem.
-/
theorem fixedSupportProbabilityBound_of_measurable
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (ε : ℝ)
    (hε : 0 ≤ ε)
    (hViolationMeas :
      ViolationSetsMeasurable violates)
    (hSplitMeas :
      MeasurableSet
        (SplitSupportBadEvent
          P violates reconstruct I ε)) :
    FixedSupportProbabilityBound
      P violates reconstruct I ε := by
  unfold FixedSupportProbabilityBound

  exact
    fixedSupportBadEvent_measure_le
      P
      violates
      reconstruct
      I
      ε
      hε
      hViolationMeas
      hSplitMeas

end DROSafety.SingleMode
