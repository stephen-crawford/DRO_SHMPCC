import DROSafety.SingleMode.IID
import DROSafety.SingleMode.Compression
import Mathlib.Data.Fintype.Card
import Mathlib.MeasureTheory.Constructions.Pi

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Support / Complement Sample Split

For a fixed support set `I ⊆ Fin S`, this file splits a full IID
scenario sample into

1. the scenarios indexed by `I`, and
2. the scenarios indexed by the complement of `I`.

The split is first formulated for an arbitrary predicate. This avoids
ambiguity between different `Fintype` instances for finite subtypes.
-/

namespace DROSafety.SingleMode

/--
IID product measure over coordinates satisfying a predicate `p`.
-/
noncomputable def iidPredicateMeasure
    {S : ℕ}
    {Ξ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (p : Fin S → Prop)
    [DecidablePred p] :
    MeasureTheory.Measure ({i : Fin S // p i} → Ξ) :=
  MeasureTheory.Measure.pi
    (fun _ : {i : Fin S // p i} => P)

/--
An IID product indexed by a finite predicate subtype is again a
probability measure.
-/
theorem iidPredicateMeasure_isProbability
    {S : ℕ}
    {Ξ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (p : Fin S → Prop)
    [DecidablePred p] :
    MeasureTheory.IsProbabilityMeasure
      (iidPredicateMeasure P p) := by
  unfold iidPredicateMeasure
  infer_instance

/--
A full sample can be split measurably according to an arbitrary
predicate `p`.
-/
noncomputable def splitPredicateEquiv
    {S : ℕ}
    {Ξ : Type*}
    [MeasurableSpace Ξ]
    (p : Fin S → Prop)
    [DecidablePred p] :
    Sample S Ξ ≃ᵐ
      (({i : Fin S // p i} → Ξ) ×
       ({i : Fin S // ¬ p i} → Ξ)) :=
  MeasurableEquiv.piEquivPiSubtypeProd
    (fun _ : Fin S => Ξ)
    p

/--
Splitting an IID sample according to a predicate preserves the
product measure.

This is the generic form of the support/complement decomposition.
-/
theorem splitPredicate_measurePreserving
    {S : ℕ}
    {Ξ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (p : Fin S → Prop)
    [DecidablePred p] :
    MeasureTheory.MeasurePreserving
      (splitPredicateEquiv (Ξ := Ξ) p)
      (iidSampleMeasure P S)
      ((iidPredicateMeasure P p).prod
       (iidPredicateMeasure P (fun i => ¬ p i))) := by
  simpa [
    splitPredicateEquiv,
    iidSampleMeasure,
    iidPredicateMeasure
  ] using
    (MeasureTheory.measurePreserving_piEquivPiSubtypeProd
      (fun _ : Fin S => P)
      p)

/--
Predicate selecting the indices belonging to a support set.
-/
abbrev supportPredicate
    {S : ℕ}
    (I : Finset (Fin S))
    (i : Fin S) :
    Prop :=
  i ∈ I

/--
Support-coordinate index type.
-/
abbrev SupportIndex
    {S : ℕ}
    (I : Finset (Fin S)) :=
  {i : Fin S // supportPredicate I i}

/--
Non-support-coordinate index type.
-/
abbrev OutsideIndex
    {S : ℕ}
    (I : Finset (Fin S)) :=
  {i : Fin S // ¬ supportPredicate I i}

/--
The measurable support/complement split for a particular support set.
-/
noncomputable def splitSampleEquiv
    {S : ℕ}
    {Ξ : Type*}
    [MeasurableSpace Ξ]
    (I : Finset (Fin S)) :
    Sample S Ξ ≃ᵐ
      ((SupportIndex I → Ξ) ×
       (OutsideIndex I → Ξ)) :=
  splitPredicateEquiv
    (Ξ := Ξ)
    (supportPredicate I)

/--
The support/complement split preserves the IID product measure.
-/
theorem splitSample_measurePreserving
    {S : ℕ}
    {Ξ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    [MeasureTheory.IsProbabilityMeasure P]
    (I : Finset (Fin S)) :
    MeasureTheory.MeasurePreserving
      (splitSampleEquiv (Ξ := Ξ) I)
      (iidSampleMeasure P S)
      ((iidPredicateMeasure P
          (supportPredicate I)).prod
       (iidPredicateMeasure P
          (fun i => ¬ supportPredicate I i))) := by
  exact
    splitPredicate_measurePreserving
      P
      (supportPredicate I)

/--
The number of support coordinates is exactly `I.card`.
-/
theorem supportIndex_card
    {S : ℕ}
    (I : Finset (Fin S)) :
    Fintype.card (SupportIndex I) =
      I.card := by
  classical
  simpa [SupportIndex, supportPredicate] using
    (Fintype.subtype_card
      I
      (fun i : Fin S => Iff.rfl))

/--
The number of non-support coordinates is `S - I.card`.
-/
theorem outsideIndex_card
    {S : ℕ}
    (I : Finset (Fin S)) :
    Fintype.card (OutsideIndex I) =
      S - I.card := by
  classical

  have h :=
    Fintype.card_subtype_compl
      (supportPredicate I)

  rw [Fintype.card_fin] at h
  rw [supportIndex_card I] at h

  exact h

end DROSafety.SingleMode
