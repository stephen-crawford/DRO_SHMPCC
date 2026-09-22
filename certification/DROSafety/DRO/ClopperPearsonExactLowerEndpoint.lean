import DROSafety.DRO.BinomialUpperTailIndexMonotonicity
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Exact lower Clopper-Pearson endpoint

For a fixed number of trials `m` and lower-tail budget `α ∈ [0,1]`,
define the total lower endpoint

    L(k) =
      0,                         if k = 0,
      r_{m,k}(α),                if 0 < k ≤ m,
      1,                         if m < k,

where `r_{m,k}(α)` is the unique root in `[0,1]` of

    upperBinomialTailPolynomial m k x = α.

For the statistically relevant range `0 ≤ k ≤ m`, this is exactly the
usual lower Clopper-Pearson construction.

The extension to `k > m` is only to make the endpoint a total monotone
function `ℕ → ℝ`, as required by `BetaClopperPearsonEndpointSpec`.
-/

namespace DROSafety.DRO

/--
Total exact lower Clopper-Pearson endpoint.

For valid counts:

* `k = 0` gives `0`;
* `0 < k ≤ m` gives the exact upper-tail level root.

Above the binomial support we extend the function by `1`.
-/
noncomputable def exactClopperPearsonLowerEndpoint
    (m : ℕ)
    (α : ℝ)
    (k : ℕ) :
    ℝ :=
  if k = 0 then
    0
  else if k ≤ m then
    upperBinomialTailLevelRoot
      m
      k
      α
  else
    1

/--
The lower endpoint at count zero is exactly zero.
-/
theorem exactClopperPearsonLowerEndpoint_zero
    (m : ℕ)
    (α : ℝ) :
    exactClopperPearsonLowerEndpoint
        m
        α
        0
      =
    0 := by

  simp [
    exactClopperPearsonLowerEndpoint
  ]

/--
For a positive valid count, the total lower endpoint is exactly the
previously constructed upper-tail level root.
-/
theorem exactClopperPearsonLowerEndpoint_eq_root
    (m k : ℕ)
    (α : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m) :
    exactClopperPearsonLowerEndpoint
        m
        α
        k
      =
    upperBinomialTailLevelRoot
      m
      k
      α := by

  have hkNe :
      k ≠ 0 :=
    Nat.ne_of_gt
      hk

  simp [
    exactClopperPearsonLowerEndpoint,
    hkNe,
    hkm
  ]

/--
Above the binomial support, the total lower endpoint is extended by `1`.
-/
theorem exactClopperPearsonLowerEndpoint_eq_one_of_lt
    (m k : ℕ)
    (α : ℝ)
    (hmk :
      m < k) :
    exactClopperPearsonLowerEndpoint
        m
        α
        k
      =
    1 := by

  have hkPos :
      0 < k :=
    lt_of_le_of_lt
      (Nat.zero_le m)
      hmk

  have hkNe :
      k ≠ 0 :=
    Nat.ne_of_gt
      hkPos

  have hNotLe :
      ¬ k ≤ m :=
    Nat.not_le.mpr
      hmk

  simp [
    exactClopperPearsonLowerEndpoint,
    hkNe,
    hNotLe
  ]

/--
For `α ∈ [0,1]`, the total lower endpoint is nonnegative.
-/
theorem exactClopperPearsonLowerEndpoint_nonneg
    (m k : ℕ)
    (α : ℝ)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    0 ≤
      exactClopperPearsonLowerEndpoint
        m
        α
        k := by

  by_cases hkZero :
      k = 0

  · subst k

    rw [
      exactClopperPearsonLowerEndpoint_zero
        m
        α
    ]

  · have hkPos :
        0 < k :=
      Nat.pos_of_ne_zero
        hkZero

    by_cases hkm :
        k ≤ m

    · rw [
        exactClopperPearsonLowerEndpoint_eq_root
          m
          k
          α
          hkPos
          hkm
      ]

      exact
        upperBinomialTailLevelRoot_nonneg
          m
          k
          α
          hkPos
          hkm
          hα0
          hα1

    · have hmk :
          m < k :=
        Nat.lt_of_not_ge
          hkm

      rw [
        exactClopperPearsonLowerEndpoint_eq_one_of_lt
          m
          k
          α
          hmk
      ]

      norm_num

/--
For `α ∈ [0,1]`, the total lower endpoint is at most one.
-/
theorem exactClopperPearsonLowerEndpoint_le_one
    (m k : ℕ)
    (α : ℝ)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    exactClopperPearsonLowerEndpoint
        m
        α
        k
      ≤
    1 := by

  by_cases hkZero :
      k = 0

  · subst k

    rw [
      exactClopperPearsonLowerEndpoint_zero
        m
        α
    ]

    norm_num

  · have hkPos :
        0 < k :=
      Nat.pos_of_ne_zero
        hkZero

    by_cases hkm :
        k ≤ m

    · rw [
        exactClopperPearsonLowerEndpoint_eq_root
          m
          k
          α
          hkPos
          hkm
      ]

      exact
        upperBinomialTailLevelRoot_le_one
          m
          k
          α
          hkPos
          hkm
          hα0
          hα1

    · have hmk :
          m < k :=
        Nat.lt_of_not_ge
          hkm

      rw [
        exactClopperPearsonLowerEndpoint_eq_one_of_lt
          m
          k
          α
          hmk
      ]

/--
For a valid positive count, the lower endpoint satisfies the defining
upper-binomial-tail equation exactly.
-/
theorem upperBinomialTailPolynomial_exactLowerEndpoint
    (m k : ℕ)
    (α : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    upperBinomialTailPolynomial
        m
        k
        (
          exactClopperPearsonLowerEndpoint
            m
            α
            k
        )
      =
    α := by

  rw [
    exactClopperPearsonLowerEndpoint_eq_root
      m
      k
      α
      hk
      hkm
  ]

  exact
    upperBinomialTailLevelRoot_eq
      m
      k
      α
      hk
      hkm
      hα0
      hα1

/--
For `α ∈ [0,1]`, the total exact lower Clopper-Pearson endpoint is
monotone in the observed count.
-/
theorem exactClopperPearsonLowerEndpoint_monotone
    (m : ℕ)
    (α : ℝ)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    Monotone
      (
        exactClopperPearsonLowerEndpoint
          m
          α
      ) := by

  intro k₁ k₂ hk₁₂

  by_cases hk₁Zero :
      k₁ = 0

  · subst k₁

    rw [
      exactClopperPearsonLowerEndpoint_zero
        m
        α
    ]

    exact
      exactClopperPearsonLowerEndpoint_nonneg
        m
        k₂
        α
        hα0
        hα1

  · have hk₁Pos :
        0 < k₁ :=
      Nat.pos_of_ne_zero
        hk₁Zero

    by_cases hk₂m :
        k₂ ≤ m

    · have hk₁m :
          k₁ ≤ m :=
        le_trans
          hk₁₂
          hk₂m

      have hk₂Pos :
          0 < k₂ :=
        lt_of_lt_of_le
          hk₁Pos
          hk₁₂

      rw [
        exactClopperPearsonLowerEndpoint_eq_root
          m
          k₁
          α
          hk₁Pos
          hk₁m,
        exactClopperPearsonLowerEndpoint_eq_root
          m
          k₂
          α
          hk₂Pos
          hk₂m
      ]

      exact
        upperBinomialTailLevelRoot_mono_index
          m
          k₁
          k₂
          α
          hk₁Pos
          hk₁₂
          hk₂m
          hα0
          hα1

    · have hmk₂ :
          m < k₂ :=
        Nat.lt_of_not_ge
          hk₂m

      rw [
        exactClopperPearsonLowerEndpoint_eq_one_of_lt
          m
          k₂
          α
          hmk₂
      ]

      exact
        exactClopperPearsonLowerEndpoint_le_one
          m
          k₁
          α
          hα0
          hα1

end DROSafety.DRO
