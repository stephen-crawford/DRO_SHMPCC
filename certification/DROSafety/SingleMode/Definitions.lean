import DROSafety.ScenarioBound
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Single-Mode Scenario Definitions

Basic definitions for the single-mode scenario theorem.

`Ξ` is the uncertainty/scenario space.
`Θ` is the decision space.
`violates θ ξ` means that decision `θ` violates the constraint
under uncertainty realization `ξ`.
-/

namespace DROSafety.SingleMode

/--
A sample of `S` uncertainty realizations.
-/
abbrev Sample
    (S : ℕ)
    (Ξ : Type*) :=
  Fin S → Ξ

/--
The set of uncertainty realizations that violate a decision.
-/
def violationSet
    {Ξ Θ : Type*}
    (violates : Θ → Ξ → Prop)
    (θ : Θ) :
    Set Ξ :=
  {ξ | violates θ ξ}

/--
The true violation probability of a decision.
-/
def violationProbability
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (P : MeasureTheory.Measure Ξ)
    (violates : Θ → Ξ → Prop)
    (θ : Θ) :
    ENNReal :=
  P (violationSet violates θ)

/--
All violation sets produced by the decision family are measurable.
-/
def ViolationSetsMeasurable
    {Ξ Θ : Type*}
    [MeasurableSpace Ξ]
    (violates : Θ → Ξ → Prop) :
    Prop :=
  ∀ θ, MeasurableSet (violationSet violates θ)

end DROSafety.SingleMode
