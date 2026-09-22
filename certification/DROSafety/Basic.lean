import Mathlib

/-!
# Basic DRO safety definitions

Finite-mode vectors, probability distributions, and elementary
weighted-risk inequalities used by the certification framework.
-/

open scoped BigOperators

namespace DROSafety

variable {K : ℕ}

/-- A real-valued vector indexed by the finite set of motion modes. -/
abbrev ModeVec (K : ℕ) :=
  Fin K → ℝ

/-- Dot product over the finite set of modes. -/
def dot (x y : ModeVec K) : ℝ :=
  ∑ i : Fin K, x i * y i

/-- A vector is a probability distribution when it is nonnegative
    and sums to one. -/
def IsProbabilityVector (p : ModeVec K) : Prop :=
  (∀ i, 0 ≤ p i) ∧
  (∑ i : Fin K, p i) = 1

/--
If Vₘ ≤ εₘ for every mode and p is nonnegative, then

    pᵀ V ≤ pᵀ ε.
-/
theorem dot_le_dot_of_nonneg_left
    (p V ε : ModeVec K)
    (hp : ∀ i, 0 ≤ p i)
    (hV : ∀ i, V i ≤ ε i) :
    dot p V ≤ dot p ε := by

  unfold dot

  apply Finset.sum_le_sum

  intro i hi

  exact mul_le_mul_of_nonneg_left
    (hV i)
    (hp i)

end DROSafety
