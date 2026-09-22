import DROSafety.SingleMode.SupportTransfer

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Single-Mode Regularity Assumptions

The generic scenario theorem requires the relevant violation and
support-dependent bad events to be measurable.

These are mathematical regularity assumptions, not safety assumptions.
They can later be discharged for a concrete MPC problem from regularity
of its constraint and reconstruction maps.
-/

namespace DROSafety.SingleMode

/--
Regularity assumptions required by the generic single-mode
scenario theorem.
-/
structure ScenarioRegularity
    {S : ℕ}
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (reconstruct : Reconstructor S Ξ Θ) : Prop where

  violationSets :
    ViolationSetsMeasurable violates

  splitBadEvent :
    ∀ (I : Finset (Fin S)) (ε : ℝ),
      MeasurableSet
        (SplitSupportBadEvent
          P
          violates
          reconstruct
          I
          ε)

end DROSafety.SingleMode
