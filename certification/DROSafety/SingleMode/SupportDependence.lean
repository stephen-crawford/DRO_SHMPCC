import DROSafety.SingleMode.SupportSplit
import DROSafety.SingleMode.Compression
import DROSafety.SingleMode.FixedDecision

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Support Dependence

This file proves that a reconstructed scenario decision depends only
on the samples contained in its support set.

That fact is needed before conditioning on the support coordinates
in the fixed-support probability proof.
-/

namespace DROSafety.SingleMode

/--
The support coordinates of a full sample.
-/
def supportPart
    {S : ℕ}
    {Ξ : Type*}
    (I : Finset (Fin S))
    (z : Sample S Ξ) :
    SupportIndex I → Ξ :=
  fun i => z i.1

/--
The non-support coordinates of a full sample.
-/
def outsidePart
    {S : ℕ}
    {Ξ : Type*}
    (I : Finset (Fin S))
    (z : Sample S Ξ) :
    OutsideIndex I → Ξ :=
  fun i => z i.1

/--
The decision obtained directly from support coordinates.
-/
def supportDecision
    {S : ℕ}
    {Ξ Θ : Type*}
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (x : SupportIndex I → Ξ) :
    Θ :=
  reconstruct I x

/--
If two full samples agree on every support coordinate, then their
restricted support samples are equal.
-/
theorem restrictSample_eq_of_eq_on_support
    {S : ℕ}
    {Ξ : Type*}
    (I : Finset (Fin S))
    (z₁ z₂ : Sample S Ξ)
    (h :
      ∀ i, i ∈ I →
        z₁ i = z₂ i) :
    restrictSample z₁ I =
      restrictSample z₂ I := by
  funext i
  exact h i.1 i.2

/--
A reconstructed decision is unchanged when only non-support
coordinates are changed.
-/
theorem reconstructedDecision_eq_of_eq_on_support
    {S : ℕ}
    {Ξ Θ : Type*}
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (z₁ z₂ : Sample S Ξ)
    (h :
      ∀ i, i ∈ I →
        z₁ i = z₂ i) :
    reconstructedDecision reconstruct I z₁ =
      reconstructedDecision reconstruct I z₂ := by
  unfold reconstructedDecision
  exact congrArg
    (reconstruct I)
    (restrictSample_eq_of_eq_on_support
      I z₁ z₂ h)

/--
The decision reconstructed from a full sample is exactly the decision
obtained from its support coordinates.
-/
theorem reconstructedDecision_eq_supportDecision
    {S : ℕ}
    {Ξ Θ : Type*}
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (z : Sample S Ξ) :
    reconstructedDecision reconstruct I z =
      supportDecision reconstruct I
        (supportPart I z) := by
  rfl

/--
All outside coordinates satisfy a fixed decision.
-/
def OutsideSafe
    {S : ℕ}
    {Ξ Θ : Type*}
    (violates : Θ → Ξ → Prop)
    (I : Finset (Fin S))
    (θ : Θ)
    (y : OutsideIndex I → Ξ) :
    Prop :=
  ∀ j, ¬ violates θ (y j)

/--
Consistency outside the support set is exactly safety of the outside
coordinate block.
-/
theorem consistentOutside_iff_outsideSafe
    {S : ℕ}
    {Ξ Θ : Type*}
    (violates : Θ → Ξ → Prop)
    (I : Finset (Fin S))
    (z : Sample S Ξ)
    (θ : Θ) :
    ConsistentOutside violates I z θ ↔
      OutsideSafe violates I θ
        (outsidePart I z) := by
  constructor
  · intro h j
    exact h j.1 j.2
  · intro h j hj
    exact h ⟨j, hj⟩

end DROSafety.SingleMode
