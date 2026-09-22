import DROSafety.SingleMode.IID
import DROSafety.SingleMode.Compression
import DROSafety.SingleMode.FixedDecision
import DROSafety.SingleMode.Compression

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Fixed-Support Bad Event

For one fixed support subset `I`, define the event that

1. the reconstructed decision has true violation probability greater
   than `ε`, but
2. nevertheless satisfies every sampled scenario outside `I`.

The main probability lemma will eventually show that this event has
probability at most

    (1 - ε)^(S - |I|).
-/

namespace DROSafety.SingleMode

/--
Bad event corresponding to one fixed support subset.
-/
def FixedSupportBadEvent
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (ε : ℝ) :
    Set (Sample S Ξ) :=
  {z |
    ENNReal.ofReal ε <
        violationProbability
          P
          violates
          (reconstructedDecision reconstruct I z) ∧
    ConsistentOutside
      violates
      I
      z
      (reconstructedDecision reconstruct I z)}

/--
The desired fixed-support probability statement.

This is packaged as a proposition first. We will prove that it follows
from IID sampling and the required measurability assumptions.
-/
def FixedSupportProbabilityBound
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ)
    (I : Finset (Fin S))
    (ε : ℝ) :
    Prop :=
  iidSampleMeasure P S
      (FixedSupportBadEvent
        P violates reconstruct I ε)
    ≤
      (ENNReal.ofReal (1 - ε)) ^
        (S - I.card)

end DROSafety.SingleMode
