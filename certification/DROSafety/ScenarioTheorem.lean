import DROSafety.ScenarioBound
import DROSafety.Confidence

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Mode-wise Scenario Theorem

This file formalizes the scenario-theoretic component of the
mode-stratified DRO safety guarantee.

The ultimate result to establish for every mode `m` is

    Pr[Vₘ > ε(Sₘ,nₘ,βₘ)] ≤ βₘ.

This is the remaining bridge between finite scenario samples and
the deterministic mode-wise certificate consumed by the DRO
aggregation theorem.
-/

namespace DROSafety

variable {K : ℕ}

/--
A valid scenario confidence parameter lies strictly between zero
and one.
-/
def IsScenarioConfidence (β : ℝ) : Prop :=
  0 < β ∧ β ≤ 1

/--
All mode-wise confidence parameters are valid.
-/
def IsScenarioConfidenceVector
    (β : ModeVec K) : Prop :=
  ∀ m, IsScenarioConfidence (β m)

/--
The support count for every mode must be strictly smaller than the
number of scenarios used for that mode.
-/
def ValidScenarioCounts
    (S n : Fin K → ℕ) : Prop :=
  ∀ m, n m < S m

/--
Extract positivity of a mode-wise confidence parameter.
-/
theorem scenarioConfidence_pos
    {β : ModeVec K}
    (hβ : IsScenarioConfidenceVector β)
    (m : Fin K) :
    0 < β m := by
  exact (hβ m).1

/--
Extract the upper bound `βₘ ≤ 1`.
-/
theorem scenarioConfidence_le_one
    {β : ModeVec K}
    (hβ : IsScenarioConfidenceVector β)
    (m : Fin K) :
    β m ≤ 1 := by
  exact (hβ m).2

section Probability

variable {Ω : Type*}
variable [MeasurableSpace Ω]

/--
The mode-wise probabilistic scenario guarantee.

This packages the result supplied by the standard Safe-Horizon /
scenario theorem:

* every confidence parameter is valid,
* every support cap is smaller than its sample count,
* for each mode, the probability that the conditional violation
  exceeds the scenario bound is at most `βₘ`.

The full proof of the underlying single-mode scenario theorem is
not yet formalized here. This structure makes that dependency
explicit rather than introducing it as an axiom.
-/
structure ModewiseScenarioGuarantee
    (μ : MeasureTheory.Measure Ω)
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (V : Ω → ModeVec K) : Prop where
  validConfidence :
    IsScenarioConfidenceVector β

  validCounts :
    ValidScenarioCounts S nBar

  failureBound :
    ∀ m,
      μ (ScenarioFailure S nBar β V m) ≤
        ENNReal.ofReal (β m)

/--
Extract the per-mode probability bound from a
`ModewiseScenarioGuarantee`.
-/
theorem ModewiseScenarioGuarantee.failure_le
    {μ : MeasureTheory.Measure Ω}
    {S nBar : Fin K → ℕ}
    {β : ModeVec K}
    {V : Ω → ModeVec K}
    (h : ModewiseScenarioGuarantee μ S nBar β V)
    (m : Fin K) :
    μ (ScenarioFailure S nBar β V m) ≤
      ENNReal.ofReal (β m) := by
  exact h.failureBound m

end Probability

end DROSafety
