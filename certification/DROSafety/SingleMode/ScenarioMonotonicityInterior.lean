import DROSafety.SingleMode.ScenarioMonotonicityAlgebra

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Interior adjacent-step monotonicity for scenario epsilon

This file converts the algebraic estimate from
`ScenarioMonotonicityAlgebra` into the required `Real.rpow` inequality and then
uses `ScenarioMonotonicityStep` to obtain global monotonicity.
-/

namespace DROSafety.SingleMode

/--
The nontrivial interior adjacent inequality for `scenarioEpsilon`.
-/
theorem scenarioEpsilon_interior_step
    (S n : ℕ)
    (β : ℝ)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1)
    (hNext : n + 1 < S) :
    DROSafety.scenarioEpsilon S n β ≤
      DROSafety.scenarioEpsilon S (n + 1) β := by
  have hnS : n < S := by
    omega

  let k : ℕ := S - n
  let l : ℕ := S - (n + 1)

  have hkPos : 0 < k := by
    dsimp [k]
    omega

  have hlPos : 0 < l := by
    dsimp [l]
    omega

  have hkl : k = l + 1 := by
    dsimp [k, l]
    omega

  let a : ℝ :=
    β / ((S : ℝ) * (Nat.choose S n : ℝ))

  let b : ℝ :=
    β / ((S : ℝ) * (Nat.choose S (n + 1) : ℝ))

  let r : ℝ :=
    ((n + 1 : ℕ) : ℝ) / ((S - n : ℕ) : ℝ)

  have haNonneg : 0 ≤ a := by
    dsimp [a]
    positivity

  have hbNonneg : 0 ≤ b := by
    dsimp [b]
    positivity

  have hrNonneg : 0 ≤ r := by
    dsimp [r]
    positivity

  have hCore :
      a * r ^ k ≤ 1 := by
    simpa [a, r, k] using
      scenario_base_mul_ratio_pow_le_one
        S n β hβ0 hβ1 hnS

  have hBaseStep :
      b = a * r := by
    simpa [a, b, r] using
      scenario_next_base_eq S n β hNext

  have hPower :
      b ^ k ≤ a ^ l := by
    calc
      b ^ k = (a * r) ^ k := by
        rw [hBaseStep]
      _ = a ^ k * r ^ k := by
        rw [mul_pow]
      _ = (a ^ l * a) * r ^ k := by
        rw [hkl, pow_succ]
      _ = a ^ l * (a * r ^ k) := by
        ring
      _ ≤ a ^ l * 1 := by
        exact mul_le_mul_of_nonneg_left hCore (by positivity)
      _ = a ^ l := by
        ring

  have hRootB :
      (Real.rpow b (1 / (l : ℝ))) ^ l = b := by
    simpa [one_div] using
      (Real.rpow_inv_natCast_pow
        hbNonneg
        (Nat.ne_of_gt hlPos))

  have hRootA :
      (Real.rpow a (1 / (k : ℝ))) ^ k = a := by
    simpa [one_div] using
      (Real.rpow_inv_natCast_pow
        haNonneg
        (Nat.ne_of_gt hkPos))

  have hLeftNonneg :
      0 ≤ Real.rpow b (1 / (l : ℝ)) :=
    Real.rpow_nonneg hbNonneg _

  have hRightNonneg :
      0 ≤ Real.rpow a (1 / (k : ℝ)) :=
    Real.rpow_nonneg haNonneg _

  have hExponentNe : k * l ≠ 0 :=
    Nat.mul_ne_zero
      (Nat.ne_of_gt hkPos)
      (Nat.ne_of_gt hlPos)

  have hRoot :
      Real.rpow b (1 / (l : ℝ)) ≤
        Real.rpow a (1 / (k : ℝ)) := by
    apply
      (pow_le_pow_iff_left₀
        hLeftNonneg
        hRightNonneg
        hExponentNe).mp

    calc
      (Real.rpow b (1 / (l : ℝ))) ^ (k * l) =
          b ^ k := by
            rw [Nat.mul_comm k l, pow_mul, hRootB]
      _ ≤ a ^ l := hPower
      _ = (Real.rpow a (1 / (k : ℝ))) ^ (k * l) := by
            symm
            rw [pow_mul, hRootA]

  have hRootExpanded :
      Real.rpow
          (β / ((S : ℝ) * (Nat.choose S (n + 1) : ℝ)))
          ((1 : ℝ) / ((S - (n + 1) : ℕ) : ℝ)) ≤
        Real.rpow
          (β / ((S : ℝ) * (Nat.choose S n : ℝ)))
          ((1 : ℝ) / ((S - n : ℕ) : ℝ)) := by
    simpa [a, b, k, l] using hRoot

  simp only [
    DROSafety.scenarioEpsilon,
    if_pos hnS,
    if_pos hNext
  ]

  linarith

/--
For a genuine confidence parameter `0 ≤ β ≤ 1`, the interior-step property
required by `ScenarioMonotonicityStep` holds.
-/
theorem scenarioEpsilonInteriorStepMonotone_of_confidence
    (S : ℕ)
    (β : ℝ)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1) :
    ScenarioEpsilonInteriorStepMonotone S β := by
  intro n hNext
  exact
    scenarioEpsilon_interior_step
      S n β hβ0 hβ1 hNext

/--
Global monotonicity of `scenarioEpsilon` in the support count.
-/
theorem scenarioEpsilon_monotone_of_confidence
    (S : ℕ)
    (β : ℝ)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1) :
    Monotone
      (fun n => DROSafety.scenarioEpsilon S n β) := by
  exact
    scenarioEpsilon_monotone_of_interior
      S
      β
      hβ0
      (scenarioEpsilonInteriorStepMonotone_of_confidence
        S β hβ0 hβ1)

/--
The existing bounded-support monotonicity interface follows immediately.
-/
theorem scenarioEpsilonMonotoneUpTo_of_confidence
    (S nBar : ℕ)
    (β : ℝ)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1) :
    ScenarioEpsilonMonotoneUpTo S nBar β := by
  exact
    scenarioEpsilonMonotoneUpTo_of_interior
      S
      nBar
      β
      hβ0
      (scenarioEpsilonInteriorStepMonotone_of_confidence
        S β hβ0 hβ1)

end DROSafety.SingleMode
