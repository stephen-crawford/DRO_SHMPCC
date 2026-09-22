import DROSafety.SingleMode.ScenarioMonotonicityStep

import Mathlib.Data.Nat.Choose.Bounds
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Algebraic lemmas for scenario-epsilon monotonicity

This file isolates the finite-dimensional algebra needed for the
interior adjacent-support monotonicity proof.
-/

namespace DROSafety.SingleMode

/--
The standard adjacent-binomial identity, cast to the reals.
-/
theorem choose_succ_real_identity
    (S n : ℕ) :
    (Nat.choose S (n + 1) : ℝ) * ((n + 1 : ℕ) : ℝ) =
      (Nat.choose S n : ℝ) * ((S - n : ℕ) : ℝ) := by
  exact_mod_cast Nat.choose_succ_right_eq S n

/--
For an interior support count, the ratio appearing after applying the
adjacent-binomial identity is controlled by the current binomial coefficient:

    ((n+1)/(S-n))^(S-n) ≤ choose S n.

The proof uses symmetry of `choose`, Mathlib's lower bound
`Nat.pow_le_choose`, and `Nat.factorial_le_pow`.
-/
theorem scenario_ratio_pow_le_choose
    (S n : ℕ)
    (hnS : n < S) :
    ((((n + 1 : ℕ) : ℝ) / ((S - n : ℕ) : ℝ)) ^ (S - n)) ≤
      (Nat.choose S n : ℝ) := by
  have hGapPos : 0 < S - n := by
    omega

  have hnle : n ≤ S :=
    Nat.le_of_lt hnS

  have hChooseSymmNat :
      Nat.choose S (S - n) = Nat.choose S n := by
    exact Nat.choose_symm hnle

  have hSubIdentity :
      S + 1 - (S - n) = n + 1 := by
    omega

  have hPowChoose :
      ((((n + 1 : ℕ) : ℝ) ^ (S - n)) /
          (Nat.factorial (S - n) : ℝ)) ≤
        (Nat.choose S n : ℝ) := by
    calc
      ((((n + 1 : ℕ) : ℝ) ^ (S - n)) /
          (Nat.factorial (S - n) : ℝ)) =
          ((((S + 1 - (S - n) : ℕ) : ℝ) ^ (S - n)) /
            (Nat.factorial (S - n) : ℝ)) := by
              rw [hSubIdentity]
      _ ≤ (Nat.choose S (S - n) : ℝ) := by
            exact Nat.pow_le_choose (α := ℝ) (S - n) S
      _ = (Nat.choose S n : ℝ) := by
            exact_mod_cast hChooseSymmNat

  have hFactorialLe :
      (Nat.factorial (S - n) : ℝ) ≤
        (((S - n : ℕ) : ℝ) ^ (S - n)) := by
    exact_mod_cast (Nat.factorial_le_pow (S - n))

  have hFactorialPos :
      0 < (Nat.factorial (S - n) : ℝ) := by
    positivity

  have hNumeratorNonneg :
      0 ≤ (((n + 1 : ℕ) : ℝ) ^ (S - n)) := by
    positivity

  have hDivide :
      ((((n + 1 : ℕ) : ℝ) ^ (S - n)) /
          (((S - n : ℕ) : ℝ) ^ (S - n))) ≤
        ((((n + 1 : ℕ) : ℝ) ^ (S - n)) /
          (Nat.factorial (S - n) : ℝ)) := by
    exact
      div_le_div_of_nonneg_left
        hNumeratorNonneg
        hFactorialPos
        hFactorialLe

  calc
    ((((n + 1 : ℕ) : ℝ) / ((S - n : ℕ) : ℝ)) ^ (S - n)) =
        ((((n + 1 : ℕ) : ℝ) ^ (S - n)) /
          (((S - n : ℕ) : ℝ) ^ (S - n))) := by
            rw [div_pow]
    _ ≤ ((((n + 1 : ℕ) : ℝ) ^ (S - n)) /
          (Nat.factorial (S - n) : ℝ)) := hDivide
    _ ≤ (Nat.choose S n : ℝ) := hPowChoose

/--
The dimensionless algebraic estimate that drives the adjacent-step proof:

    [β / (S * choose S n)] * ((n+1)/(S-n))^(S-n) ≤ 1.

The assumptions `0 ≤ β ≤ 1` are exactly the confidence-parameter bounds used
by the scenario certificate.
-/
theorem scenario_base_mul_ratio_pow_le_one
    (S n : ℕ)
    (β : ℝ)
    (hβ0 : 0 ≤ β)
    (hβ1 : β ≤ 1)
    (hnS : n < S) :
    (β / ((S : ℝ) * (Nat.choose S n : ℝ))) *
        ((((n + 1 : ℕ) : ℝ) / ((S - n : ℕ) : ℝ)) ^ (S - n)) ≤
      1 := by
  have hSPosNat : 0 < S := by
    omega

  have hSPos : 0 < (S : ℝ) := by
    exact_mod_cast hSPosNat

  have hnle : n ≤ S :=
    Nat.le_of_lt hnS

  have hChoosePosNat : 0 < Nat.choose S n :=
    Nat.choose_pos hnle

  have hChoosePos : 0 < (Nat.choose S n : ℝ) := by
    exact_mod_cast hChoosePosNat

  have hBaseNonneg :
      0 ≤ β / ((S : ℝ) * (Nat.choose S n : ℝ)) := by
    positivity

  have hRatio :=
    scenario_ratio_pow_le_choose S n hnS

  have hMul :
      (β / ((S : ℝ) * (Nat.choose S n : ℝ))) *
          ((((n + 1 : ℕ) : ℝ) / ((S - n : ℕ) : ℝ)) ^ (S - n)) ≤
        (β / ((S : ℝ) * (Nat.choose S n : ℝ))) *
          (Nat.choose S n : ℝ) := by
    exact mul_le_mul_of_nonneg_left hRatio hBaseNonneg

  have hCancel :
      (β / ((S : ℝ) * (Nat.choose S n : ℝ))) *
          (Nat.choose S n : ℝ) =
        β / (S : ℝ) := by
    field_simp [ne_of_gt hSPos, ne_of_gt hChoosePos]

  have hβS : β ≤ (S : ℝ) := by
    have hOneLeS : (1 : ℝ) ≤ (S : ℝ) := by
      exact_mod_cast (Nat.succ_le_iff.mpr hSPosNat)
    linarith

  have hDivLeOne : β / (S : ℝ) ≤ 1 := by
    exact (div_le_one hSPos).2 hβS

  calc
    (β / ((S : ℝ) * (Nat.choose S n : ℝ))) *
        ((((n + 1 : ℕ) : ℝ) / ((S - n : ℕ) : ℝ)) ^ (S - n)) ≤
      (β / ((S : ℝ) * (Nat.choose S n : ℝ))) *
        (Nat.choose S n : ℝ) := hMul
    _ = β / (S : ℝ) := hCancel
    _ ≤ 1 := hDivLeOne

/--
The adjacent scenario bases are related by the exact binomial recurrence:

    base(n+1) = base(n) * (n+1)/(S-n).
-/
theorem scenario_next_base_eq
    (S n : ℕ)
    (β : ℝ)
    (hNext : n + 1 < S) :
    β / ((S : ℝ) * (Nat.choose S (n + 1) : ℝ)) =
      (β / ((S : ℝ) * (Nat.choose S n : ℝ))) *
        (((n + 1 : ℕ) : ℝ) / ((S - n : ℕ) : ℝ)) := by
  have hnS : n < S := by
    omega

  have hSPosNat : 0 < S := by
    omega

  have hSPos : 0 < (S : ℝ) := by
    exact_mod_cast hSPosNat

  have hnle : n ≤ S :=
    Nat.le_of_lt hnS

  have hn1le : n + 1 ≤ S := by
    omega

  have hChooseNPos : 0 < (Nat.choose S n : ℝ) := by
    exact_mod_cast (Nat.choose_pos hnle)

  have hChooseNextPos :
      0 < (Nat.choose S (n + 1) : ℝ) := by
    exact_mod_cast (Nat.choose_pos hn1le)

  have hGapPosNat : 0 < S - n := by
    omega

  have hGapPos : 0 < ((S - n : ℕ) : ℝ) := by
    exact_mod_cast hGapPosNat

  have hChoose :=
    choose_succ_real_identity S n

  field_simp [
    ne_of_gt hSPos,
    ne_of_gt hChooseNPos,
    ne_of_gt hChooseNextPos,
    ne_of_gt hGapPos
  ]

  ring_nf at hChoose ⊢
  rw [hChoose]

end DROSafety.SingleMode
