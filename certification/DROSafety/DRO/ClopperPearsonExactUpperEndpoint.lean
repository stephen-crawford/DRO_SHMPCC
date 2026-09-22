import DROSafety.DRO.ClopperPearsonExactLowerEndpoint
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Exact upper Clopper-Pearson endpoint

For a fixed number of trials `m` and upper-tail error budget
`α ∈ [0,1]`, define

    U(k) =
      r_{m,k+1}(1-α),     if k < m,
      1,                  if m ≤ k,

where `r_{m,k+1}(1-α)` is the unique root in `[0,1]` of

    upperBinomialTailPolynomial m (k+1) x = 1 - α.

Why this is the Clopper-Pearson upper endpoint:

    Bin(m,x) [k+1,∞)
      =
    Beta(k+1,m-k) (-∞,x].

Thus choosing the lower Beta tail to equal `1-α` makes the complementary
upper Beta tail equal `α`.

Using the already-proved monotonicity of upper-tail roots in the count
index, we obtain global monotonicity of `U`.
-/

namespace DROSafety.DRO

/--
Total exact upper Clopper-Pearson endpoint.

For `k < m`, use the exact upper-tail root at threshold `k+1` and level
`1-α`.  For `m ≤ k`, extend the endpoint by `1`.
-/
noncomputable def exactClopperPearsonUpperEndpoint
    (m : ℕ)
    (α : ℝ)
    (k : ℕ) :
    ℝ :=
  if k < m then
    upperBinomialTailLevelRoot
      m
      (k + 1)
      (1 - α)
  else
    1

/--
For `k < m`, the exact upper endpoint is the upper-tail root at
threshold `k+1` and level `1-α`.
-/
theorem exactClopperPearsonUpperEndpoint_eq_root
    (m k : ℕ)
    (α : ℝ)
    (hkm :
      k < m) :
    exactClopperPearsonUpperEndpoint
        m
        α
        k
      =
    upperBinomialTailLevelRoot
      m
      (k + 1)
      (1 - α) := by

  simp [
    exactClopperPearsonUpperEndpoint,
    hkm
  ]

/--
At and above the maximal possible count, the total upper endpoint is
exactly `1`.
-/
theorem exactClopperPearsonUpperEndpoint_eq_one
    (m k : ℕ)
    (α : ℝ)
    (hmk :
      m ≤ k) :
    exactClopperPearsonUpperEndpoint
        m
        α
        k
      =
    1 := by

  have hNot :
      ¬ k < m :=
    Nat.not_lt.mpr
      hmk

  simp [
    exactClopperPearsonUpperEndpoint,
    hNot
  ]

/--
If `α ∈ [0,1]`, then the root level `1-α` also belongs to `[0,1]`.
-/
theorem one_sub_mem_Icc
    (α : ℝ)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    1 - α ∈ Set.Icc (0 : ℝ) 1 := by

  constructor <;>
    linarith

/--
For `α ∈ [0,1]`, the total upper endpoint is nonnegative.
-/
theorem exactClopperPearsonUpperEndpoint_nonneg
    (m k : ℕ)
    (α : ℝ)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    0 ≤
      exactClopperPearsonUpperEndpoint
        m
        α
        k := by

  by_cases hkm :
      k < m

  · rw [
      exactClopperPearsonUpperEndpoint_eq_root
        m
        k
        α
        hkm
    ]

    have hkSuccPos :
        0 < k + 1 :=
      Nat.succ_pos k

    have hkSuccM :
        k + 1 ≤ m := by
      omega

    have hLevel :
        1 - α ∈ Set.Icc (0 : ℝ) 1 :=
      one_sub_mem_Icc
        α
        hα0
        hα1

    exact
      upperBinomialTailLevelRoot_nonneg
        m
        (k + 1)
        (1 - α)
        hkSuccPos
        hkSuccM
        hLevel.1
        hLevel.2

  · have hmk :
        m ≤ k :=
      Nat.le_of_not_gt
        hkm

    rw [
      exactClopperPearsonUpperEndpoint_eq_one
        m
        k
        α
        hmk
    ]

    norm_num

/--
For `α ∈ [0,1]`, the total upper endpoint is at most one.
-/
theorem exactClopperPearsonUpperEndpoint_le_one
    (m k : ℕ)
    (α : ℝ)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    exactClopperPearsonUpperEndpoint
        m
        α
        k
      ≤
    1 := by

  by_cases hkm :
      k < m

  · rw [
      exactClopperPearsonUpperEndpoint_eq_root
        m
        k
        α
        hkm
    ]

    have hkSuccPos :
        0 < k + 1 :=
      Nat.succ_pos k

    have hkSuccM :
        k + 1 ≤ m := by
      omega

    have hLevel :
        1 - α ∈ Set.Icc (0 : ℝ) 1 :=
      one_sub_mem_Icc
        α
        hα0
        hα1

    exact
      upperBinomialTailLevelRoot_le_one
        m
        (k + 1)
        (1 - α)
        hkSuccPos
        hkSuccM
        hLevel.1
        hLevel.2

  · have hmk :
        m ≤ k :=
      Nat.le_of_not_gt
        hkm

    rw [
      exactClopperPearsonUpperEndpoint_eq_one
        m
        k
        α
        hmk
    ]

/--
For `k < m`, the exact upper endpoint satisfies its defining
upper-tail equation:

    T_{m,k+1}(U(k)) = 1 - α.
-/
theorem upperBinomialTailPolynomial_exactUpperEndpoint
    (m k : ℕ)
    (α : ℝ)
    (hkm :
      k < m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    upperBinomialTailPolynomial
        m
        (k + 1)
        (
          exactClopperPearsonUpperEndpoint
            m
            α
            k
        )
      =
    1 - α := by

  rw [
    exactClopperPearsonUpperEndpoint_eq_root
      m
      k
      α
      hkm
  ]

  have hkSuccPos :
      0 < k + 1 :=
    Nat.succ_pos k

  have hkSuccM :
      k + 1 ≤ m := by
    omega

  have hLevel :
      1 - α ∈ Set.Icc (0 : ℝ) 1 :=
    one_sub_mem_Icc
      α
      hα0
      hα1

  exact
    upperBinomialTailLevelRoot_eq
      m
      (k + 1)
      (1 - α)
      hkSuccPos
      hkSuccM
      hLevel.1
      hLevel.2

/--
For `α ∈ [0,1]`, the total exact upper Clopper-Pearson endpoint is
monotone in the observed count.
-/
theorem exactClopperPearsonUpperEndpoint_monotone
    (m : ℕ)
    (α : ℝ)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    Monotone
      (
        exactClopperPearsonUpperEndpoint
          m
          α
      ) := by

  intro k₁ k₂ hk₁₂

  by_cases hk₂m :
      k₂ < m

  · have hk₁m :
        k₁ < m :=
      lt_of_le_of_lt
        hk₁₂
        hk₂m

    rw [
      exactClopperPearsonUpperEndpoint_eq_root
        m
        k₁
        α
        hk₁m,
      exactClopperPearsonUpperEndpoint_eq_root
        m
        k₂
        α
        hk₂m
    ]

    have hLevel :
        1 - α ∈ Set.Icc (0 : ℝ) 1 :=
      one_sub_mem_Icc
        α
        hα0
        hα1

    have hkSucc :
        k₁ + 1 ≤ k₂ + 1 :=
      Nat.succ_le_succ
        hk₁₂

    have hk₂SuccM :
        k₂ + 1 ≤ m := by
      omega

    exact
      upperBinomialTailLevelRoot_mono_index
        m
        (k₁ + 1)
        (k₂ + 1)
        (1 - α)
        (Nat.succ_pos k₁)
        hkSucc
        hk₂SuccM
        hLevel.1
        hLevel.2

  · have hmk₂ :
        m ≤ k₂ :=
      Nat.le_of_not_gt
        hk₂m

    rw [
      exactClopperPearsonUpperEndpoint_eq_one
        m
        k₂
        α
        hmk₂
    ]

    exact
      exactClopperPearsonUpperEndpoint_le_one
        m
        k₁
        α
        hα0
        hα1

end DROSafety.DRO
