import DROSafety.SingleMode.ScenarioMonotonicity

import Mathlib.Order.Monotone.Basic

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!

# Reduction of Scenario-Epsilon Monotonicity to Adjacent Support Sizes

To prove monotonicity of

    n ↦ scenarioEpsilon S n β,

it is sufficient to prove the adjacent inequality

    scenarioEpsilon S n β
      ≤ scenarioEpsilon S (n + 1) β

for every `n`.

For `n + 1 ≥ S`, the result is trivial because the right-hand side
is the conservative value `1`.

Thus the only nontrivial algebraic case is

    n + 1 < S.

-/

namespace DROSafety.SingleMode

/--
The nontrivial adjacent-step property for support counts strictly
below the sample count.
-/
def ScenarioEpsilonInteriorStepMonotone
    (S : ℕ)
    (β : ℝ) :
    Prop :=
  ∀ n,
    n + 1 < S →
    DROSafety.scenarioEpsilon S n β ≤
      DROSafety.scenarioEpsilon S (n + 1) β

/--
`scenarioEpsilon` never exceeds one when the confidence parameter
is nonnegative.
-/
theorem scenarioEpsilon_le_one
    (S n : ℕ)
    (β : ℝ)
    (hβ : 0 ≤ β) :
    DROSafety.scenarioEpsilon S n β ≤ 1 := by

  by_cases hnS : n < S

  · rw [DROSafety.scenarioEpsilon]

    -- Reduce the conditional itself, rather than merely rewriting
    -- the proposition `n < S` to `True`.
    simp only [if_pos hnS]

    have hBaseNonneg :
        0 ≤
          β /
            ((S : ℝ) *
              (Nat.choose S n : ℝ)) := by

      have hnle : n ≤ S :=
        Nat.le_of_lt hnS

      have hSpos : 0 < S := by
        omega

      have hChoosePos :
          0 < Nat.choose S n :=
        Nat.choose_pos hnle

      positivity

    have hrpow :
        0 ≤
          Real.rpow
            (β /
              ((S : ℝ) *
                (Nat.choose S n : ℝ)))
            ((1 : ℝ) /
              ((S - n : ℕ) : ℝ)) := by

      exact
        Real.rpow_nonneg
          hBaseNonneg
          _

    linarith

  · rw [
      DROSafety.scenarioEpsilon_of_not_lt
        S n β hnS
    ]

/--
An interior adjacent-step proof implies the adjacent inequality for
every natural support count.

Outside the interior region the successor threshold is the trivial
value `1`.
-/
theorem scenarioEpsilon_step_of_interior
    (S : ℕ)
    (β : ℝ)
    (hβ : 0 ≤ β)
    (hInterior :
      ScenarioEpsilonInteriorStepMonotone S β) :
    ∀ n,
      DROSafety.scenarioEpsilon S n β ≤
        DROSafety.scenarioEpsilon S (n + 1) β := by

  intro n

  by_cases hNext : n + 1 < S

  · exact hInterior n hNext

  · have hRight :
        DROSafety.scenarioEpsilon S (n + 1) β = 1 := by

      apply
        DROSafety.scenarioEpsilon_of_not_lt
          S
          (n + 1)
          β

      exact hNext

    rw [hRight]

    exact
      scenarioEpsilon_le_one
        S n β hβ

/--
Interior adjacent monotonicity implies global monotonicity of
`scenarioEpsilon` as a function of the support count.
-/
theorem scenarioEpsilon_monotone_of_interior
    (S : ℕ)
    (β : ℝ)
    (hβ : 0 ≤ β)
    (hInterior :
      ScenarioEpsilonInteriorStepMonotone S β) :
    Monotone
      (fun n =>
        DROSafety.scenarioEpsilon S n β) := by

  apply monotone_nat_of_le_succ

  exact
    scenarioEpsilon_step_of_interior
      S
      β
      hβ
      hInterior

/--
The global monotonicity theorem immediately supplies the existing
`ScenarioEpsilonMonotoneUpTo` interface.
-/
theorem scenarioEpsilonMonotoneUpTo_of_interior
    (S nBar : ℕ)
    (β : ℝ)
    (hβ : 0 ≤ β)
    (hInterior :
      ScenarioEpsilonInteriorStepMonotone S β) :
    ScenarioEpsilonMonotoneUpTo
      S nBar β := by

  have hMono :
      Monotone
        (fun n =>
          DROSafety.scenarioEpsilon S n β) :=
    scenarioEpsilon_monotone_of_interior
      S β hβ hInterior

  intro n hn

  exact hMono hn

end DROSafety.SingleMode
