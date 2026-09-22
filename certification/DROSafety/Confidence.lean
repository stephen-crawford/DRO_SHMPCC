import DROSafety.Ambiguity
import DROSafety.ScenarioBound
import Mathlib.MeasureTheory.OuterMeasure.Basic

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Confidence Composition

This file formalizes the confidence bookkeeping for the
mode-stratified DRO safety certificate.

There are two possible sources of certification failure:

1. The true mode distribution is not contained in the DRO ambiguity set.
2. One or more mode-wise scenario certificates fail.

The probability of any certification failure is bounded using the
finite union bound.
-/

open Set MeasureTheory
open scoped BigOperators ENNReal

namespace DROSafety

variable {K : ℕ}
variable {Ω : Type*} [MeasurableSpace Ω]

/--
The event that at least one mode-wise scenario certificate fails.
-/
def AnyModeFailure
    (bad : Fin K → Set Ω) :
    Set Ω :=
  ⋃ m, bad m

/--
The total certification failure event consists of either

* failure of ambiguity-set coverage, or
* failure of at least one mode-wise scenario certificate.
-/
def TotalFailure
    (badDRO : Set Ω)
    (badMode : Fin K → Set Ω) :
    Set Ω :=
  badDRO ∪ AnyModeFailure badMode

/--
Finite union bound over the mode-wise failure events.

    μ(⋃ₘ badₘ) ≤ Σₘ μ(badₘ)
-/
theorem anyModeFailure_measure_le
    (μ : Measure Ω)
    (bad : Fin K → Set Ω) :
    μ (AnyModeFailure bad) ≤
      ∑ m, μ (bad m) := by
  unfold AnyModeFailure
  exact MeasureTheory.measure_iUnion_fintype_le μ bad

/--
If each mode-wise failure event has probability at most `budget m`,
then the probability that any mode fails is at most the sum of the
individual confidence budgets.
-/
theorem anyModeFailure_measure_le_budget
    (μ : Measure Ω)
    (bad : Fin K → Set Ω)
    (budget : Fin K → ℝ≥0∞)
    (hBudget :
      ∀ m, μ (bad m) ≤ budget m) :
    μ (AnyModeFailure bad) ≤
      ∑ m, budget m := by
  calc
    μ (AnyModeFailure bad)
        ≤ ∑ m, μ (bad m) :=
      anyModeFailure_measure_le μ bad
    _ ≤ ∑ m, budget m := by
      apply Finset.sum_le_sum
      intro m hm
      exact hBudget m

/--
Total confidence composition.

If

    μ(badDRO) ≤ βDRO

and

    μ(badMode m) ≤ βMode m

for every mode, then

    μ(total failure)
      ≤ βDRO + Σₘ βModeₘ.
-/
theorem totalFailure_measure_le_budget
    (μ : Measure Ω)
    (badDRO : Set Ω)
    (badMode : Fin K → Set Ω)
    (βDRO : ℝ≥0∞)
    (βMode : Fin K → ℝ≥0∞)
    (hDRO :
      μ badDRO ≤ βDRO)
    (hMode :
      ∀ m, μ (badMode m) ≤ βMode m) :
    μ (TotalFailure badDRO badMode) ≤
      βDRO + ∑ m, βMode m := by
  unfold TotalFailure
  calc
    μ (badDRO ∪ AnyModeFailure badMode)
        ≤ μ badDRO + μ (AnyModeFailure badMode) :=
      MeasureTheory.measure_union_le
        (μ := μ)
        badDRO
        (AnyModeFailure badMode)
    _ ≤ βDRO + ∑ m, βMode m := by
      apply add_le_add
      · exact hDRO
      · exact
          anyModeFailure_measure_le_budget
            μ badMode βMode hMode

/--
A version with an explicit total confidence budget.

If

    βDRO + Σₘ βModeₘ ≤ βTotal,

then the complete certification failure probability is at most
`βTotal`.
-/
theorem totalFailure_measure_le_target
    (μ : Measure Ω)
    (badDRO : Set Ω)
    (badMode : Fin K → Set Ω)
    (βDRO βTotal : ℝ≥0∞)
    (βMode : Fin K → ℝ≥0∞)
    (hDRO :
      μ badDRO ≤ βDRO)
    (hMode :
      ∀ m, μ (badMode m) ≤ βMode m)
    (hTotal :
      βDRO + ∑ m, βMode m ≤ βTotal) :
    μ (TotalFailure badDRO badMode) ≤ βTotal := by
  exact le_trans
    (totalFailure_measure_le_budget
      μ badDRO badMode βDRO βMode hDRO hMode)
    hTotal

/--
Failure of the DRO ambiguity-set coverage guarantee.

The true mode distribution `p` is fixed, while the ambiguity set
`U ω` may depend on the observed data represented by outcome `ω`.
-/
def AmbiguityFailure
    (U : Ω → AmbiguitySet K)
    (p : ModeVec K) :
    Set Ω :=
  {ω | p ∉ U ω}

/--
Failure of the scenario certificate for one mode.

A failure occurs when the true conditional violation probability
for the decision generated under outcome `ω` exceeds the
support-dependent scenario bound.
-/
def ScenarioFailure
    (S n : Fin K → ℕ)
    (β : ModeVec K)
    (V : Ω → ModeVec K)
    (m : Fin K) :
    Set Ω :=
  {ω |
    scenarioEpsilon (S m) (n m) (β m) <
      V ω m}

/--
Outside the total failure event,

1. the true mode distribution belongs to the ambiguity set, and
2. every mode satisfies its scenario certificate.
-/
theorem goodOutcome_has_certificates
    (U : Ω → AmbiguitySet K)
    (p : ModeVec K)
    (V : Ω → ModeVec K)
    (S n : Fin K → ℕ)
    (β : ModeVec K)
    (ω : Ω)
    (hGood :
      ω ∉ TotalFailure
        (AmbiguityFailure U p)
        (ScenarioFailure S n β V)) :
    p ∈ U ω ∧
      ModewiseScenarioCertified
        S n β (V ω) := by
  constructor
  · have hNotAmb :
        ω ∉ AmbiguityFailure U p := by
      intro hAmb
      apply hGood
      exact Or.inl hAmb
    simpa [AmbiguityFailure] using hNotAmb
  · intro m
    have hNotModes :
        ω ∉ AnyModeFailure
          (ScenarioFailure S n β V) := by
      intro hModes
      apply hGood
      exact Or.inr hModes
    have hNotM :
        ω ∉ ScenarioFailure S n β V m := by
      intro hm
      apply hNotModes
      unfold AnyModeFailure
      exact Set.mem_iUnion.2 ⟨m, hm⟩
    have hNotLt :
        ¬ scenarioEpsilon
            (S m) (n m) (β m) <
          V ω m := by
      simpa [ScenarioFailure] using hNotM
    exact not_lt.mp hNotLt

end DROSafety
