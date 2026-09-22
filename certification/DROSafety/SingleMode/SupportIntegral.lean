import DROSafety.SingleMode.SupportFiber
import DROSafety.SingleMode.SupportSplit
import Mathlib.MeasureTheory.Measure.Prod
import Mathlib.MeasureTheory.Integral.Lebesgue.Basic

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Fixed-Support Product-Measure Bound

For a fixed support set `I`, this file integrates the previously proved
uniform fiber bound over all possible support-coordinate realizations.

The resulting product-space bad-event probability is bounded by

    (1 - ε)^(S - |I|).

The measurability of the complete split bad event is currently supplied
explicitly. It will be derived from regularity assumptions in the next
formalization step.
-/

namespace DROSafety.SingleMode

/--
The bad event on the support/outside product space.

A pair `(x,y)` is bad when

1. the decision reconstructed from support coordinates `x`
   has true violation probability greater than `ε`, and
2. all outside samples `y` happen to be safe for that decision.
-/
def SplitSupportBadEvent
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (ε : ℝ) :
    Set (
      (SupportIndex I → Ξ) ×
      (OutsideIndex I → Ξ)
    ) :=
  {xy |
    xy.2 ∈
      SupportFiberBadEvent
        P
        violates
        reconstruct
        I
        ε
        xy.1}

/--
The vertical fiber of the split bad event at support realization `x`
is exactly `SupportFiberBadEvent ... x`.
-/
theorem splitSupportBadEvent_fiber
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (ε : ℝ)
    (x : SupportIndex I → Ξ) :
    (fun y : OutsideIndex I → Ξ => (x, y)) ⁻¹'
        SplitSupportBadEvent
          P violates reconstruct I ε =
      SupportFiberBadEvent
        P violates reconstruct I ε x := by
  rfl

/--
Integrating the uniform fiber estimate gives the complete
product-space bound

    P^I × P^(Iᶜ) [bad]
      ≤ (1 - ε)^(S - |I|).
-/
theorem splitSupportBadEvent_measure_le
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
    ((iidPredicateMeasure
        P
        (supportPredicate I)).prod
      (iidPredicateMeasure
        P
        (fun i : Fin S =>
          ¬ supportPredicate I i)))
      (SplitSupportBadEvent
        P violates reconstruct I ε)
      ≤
    (ENNReal.ofReal (1 - ε)) ^
      (S - I.card) := by

  let μSupport :
      MeasureTheory.Measure
        (SupportIndex I → Ξ) :=
    iidPredicateMeasure
      P
      (supportPredicate I)

  let μOutside :
      MeasureTheory.Measure
        (OutsideIndex I → Ξ) :=
    iidPredicateMeasure
      P
      (fun i : Fin S =>
        ¬ supportPredicate I i)

  haveI hSupportProb :
      MeasureTheory.IsProbabilityMeasure μSupport := by
    dsimp [μSupport]
    exact
      iidPredicateMeasure_isProbability
        P
        (supportPredicate I)

  haveI hOutsideProb :
      MeasureTheory.IsProbabilityMeasure μOutside := by
    dsimp [μOutside]
    exact
      iidPredicateMeasure_isProbability
        P
        (fun i : Fin S =>
          ¬ supportPredicate I i)

  change
    (μSupport.prod μOutside)
        (SplitSupportBadEvent
          P violates reconstruct I ε)
      ≤
    (ENNReal.ofReal (1 - ε)) ^
      (S - I.card)

  rw [
    MeasureTheory.Measure.prod_apply
      hSplitMeas
  ]

  simp_rw [
    splitSupportBadEvent_fiber
      P
      violates
      reconstruct
      I
      ε
  ]

  calc
    (∫⁻ x,
        μOutside
          (SupportFiberBadEvent
            P violates reconstruct I ε x)
        ∂μSupport)
      ≤
        ∫⁻ _x,
          (ENNReal.ofReal (1 - ε)) ^
            (S - I.card)
          ∂μSupport := by
      apply MeasureTheory.lintegral_mono
      intro x
      dsimp [μOutside]
      exact
        supportFiberBad_measure_le
          P
          violates
          reconstruct
          I
          ε
          hε
          hViolationMeas
          x

    _ =
        (ENNReal.ofReal (1 - ε)) ^
          (S - I.card) := by
      simp

end DROSafety.SingleMode
