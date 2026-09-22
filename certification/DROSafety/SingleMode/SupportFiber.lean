import DROSafety.SingleMode.SupportDependence
import DROSafety.SingleMode.SupportSplit
import DROSafety.SingleMode.FixedDecision

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Fixed-Support Fiber Bound

Fix a support set `I` and the values of the scenarios indexed by `I`.

The reconstructed decision is then fixed. The only remaining randomness
comes from the scenarios outside `I`.

This file proves that, for every fixed support realization, the
probability that

1. the reconstructed decision has violation probability greater than
   `ε`, and
2. every outside scenario nevertheless satisfies the decision,

is bounded by

    (1 - ε)^(S - |I|).
-/

namespace DROSafety.SingleMode

/--
For a fixed decision `θ`, the event that every outside coordinate is
safe for `θ`.
-/
def OutsideAllSafeEvent
    {S : ℕ}
    {Ξ Θ : Type*}
    (violates : Θ → Ξ → Prop)
    (I : Finset (Fin S))
    (θ : Θ) :
    Set (OutsideIndex I → Ξ) :=
  Set.univ.pi
    (fun _ : OutsideIndex I =>
      safeSet violates θ)

/--
The probability that all outside coordinates are safe is the
one-sample safe probability raised to the number of outside
coordinates.
-/
theorem outsideAllSafeEvent_measure
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (I : Finset (Fin S))
    (θ : Θ) :
    iidPredicateMeasure
        P
        (fun i : Fin S =>
          ¬ supportPredicate I i)
        (OutsideAllSafeEvent
          violates I θ) =
      (P (safeSet violates θ)) ^
        Fintype.card (OutsideIndex I) := by
  unfold iidPredicateMeasure
  unfold OutsideAllSafeEvent
  rw [MeasureTheory.Measure.pi_pi]
  simp

/--
For a fixed decision whose true violation probability exceeds `ε`,
the probability that every outside sample is safe is bounded by

    (1 - ε)^(S - |I|).
-/
theorem fixedDecision_outsideAllSafe_le
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (violates : Θ → Ξ → Prop)
    (I : Finset (Fin S))
    (θ : Θ)
    (ε : ℝ)
    (hε : 0 ≤ ε)
    (hMeas :
      MeasurableSet
        (violationSet violates θ))
    (hBad :
      ENNReal.ofReal ε <
        violationProbability
          P violates θ) :
    iidPredicateMeasure
        P
        (fun i : Fin S =>
          ¬ supportPredicate I i)
        (OutsideAllSafeEvent
          violates I θ) ≤
      (ENNReal.ofReal (1 - ε)) ^
        (S - I.card) := by

  rw [outsideAllSafeEvent_measure]

  rw [outsideIndex_card I]

  exact
    pow_le_pow_left'
      (safeProbability_le_one_sub
        P
        violates
        θ
        ε
        hε
        hMeas
        hBad)
      (S - I.card)

/--
For fixed support coordinates `x`, the bad outside fiber consists of
outside samples `y` such that

1. the decision reconstructed from `x` has true violation probability
   greater than `ε`, and
2. every outside coordinate is safe for that decision.
-/
def SupportFiberBadEvent
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (ε : ℝ)
    (x : SupportIndex I → Ξ) :
    Set (OutsideIndex I → Ξ) :=
  {y |
    ENNReal.ofReal ε <
        violationProbability
          P
          violates
          (supportDecision
            reconstruct I x) ∧
    OutsideSafe
      violates
      I
      (supportDecision reconstruct I x)
      y}

/--
If the reconstructed decision is bad, then its bad outside fiber is
exactly the all-safe outside event.
-/
theorem supportFiberBadEvent_eq_allSafe
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (ε : ℝ)
    (x : SupportIndex I → Ξ)
    (hBad :
      ENNReal.ofReal ε <
        violationProbability
          P
          violates
          (supportDecision
            reconstruct I x)) :
    SupportFiberBadEvent
        P violates reconstruct I ε x =
      OutsideAllSafeEvent
        violates
        I
        (supportDecision reconstruct I x) := by
  ext y
  simp [
    SupportFiberBadEvent,
    OutsideAllSafeEvent,
    OutsideSafe,
    safeSet,
    violationSet,
    hBad
  ]

/--
If the reconstructed decision is not bad, then the corresponding bad
fiber is empty.
-/
theorem supportFiberBadEvent_eq_empty
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (ε : ℝ)
    (x : SupportIndex I → Ξ)
    (hNotBad :
      ¬ ENNReal.ofReal ε <
        violationProbability
          P
          violates
          (supportDecision
            reconstruct I x)) :
    SupportFiberBadEvent
        P violates reconstruct I ε x =
      ∅ := by
  ext y
  simp [
    SupportFiberBadEvent,
    hNotBad
  ]

/--
Uniform fixed-support fiber bound.

For every realization of the support coordinates, the probability
of the corresponding bad outside fiber is bounded by

    (1 - ε)^(S - |I|).
-/
theorem supportFiberBad_measure_le
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
    (hMeas :
      ViolationSetsMeasurable violates)
    (x : SupportIndex I → Ξ) :
    iidPredicateMeasure
        P
        (fun i : Fin S =>
          ¬ supportPredicate I i)
        (SupportFiberBadEvent
          P violates reconstruct I ε x) ≤
      (ENNReal.ofReal (1 - ε)) ^
        (S - I.card) := by

  by_cases hBad :
      ENNReal.ofReal ε <
        violationProbability
          P
          violates
          (supportDecision
            reconstruct I x)

  · rw [
      supportFiberBadEvent_eq_allSafe
        P
        violates
        reconstruct
        I
        ε
        x
        hBad
    ]

    exact
      fixedDecision_outsideAllSafe_le
        P
        violates
        I
        (supportDecision reconstruct I x)
        ε
        hε
        (hMeas
          (supportDecision reconstruct I x))
        hBad

  · rw [
      supportFiberBadEvent_eq_empty
        P
        violates
        reconstruct
        I
        ε
        x
        hBad
    ]

    simp

end DROSafety.SingleMode
