import DROSafety.DRO.BinomialTailExactRoots
import DROSafety.DRO.BinomialUpperTailDerivative
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Strict monotonicity of the upper binomial tail

For `0 < k ≤ m`, the upper-tail polynomial

    T_{m,k}(x)
      =
    sum_{j=k}^m C(m,j) x^j (1-x)^(m-j)

has derivative

    T'_{m,k}(x)
      =
    k C(m,k) x^(k-1) (1-x)^(m-k).

Every factor is strictly positive for `0 < x < 1`, so the derivative is
strictly positive on the interior of `[0,1]`.

Therefore `T_{m,k}` is strictly increasing on `[0,1]`.

As a consequence, every admissible tail level has a unique root in
`[0,1]`, and the noncomputable root selected in
`BinomialTailExactRoots.lean` is that unique root.
-/

namespace DROSafety.DRO

/--
For `0 < k ≤ m` and `0 < x < 1`, the upper-tail derivative kernel is
strictly positive.
-/
theorem upperBinomialTailDerivativeKernel_pos
    (m k : ℕ)
    (x : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hx0 :
      0 < x)
    (hx1 :
      x < 1) :
    0 <
      upperBinomialTailDerivativeKernel
        m
        k
        x := by

  unfold upperBinomialTailDerivativeKernel

  have hkReal :
      0 < (k : ℝ) := by
    exact_mod_cast hk

  have hChooseNat :
      0 < m.choose k :=
    Nat.choose_pos
      hkm

  have hChooseReal :
      0 < (m.choose k : ℝ) := by
    exact_mod_cast hChooseNat

  have hxPow :
      0 < x ^ (k - 1) :=
    pow_pos
      hx0
      (k - 1)

  have hOneMinusX :
      0 < 1 - x :=
    sub_pos.mpr
      hx1

  have hOneMinusXPow :
      0 < (1 - x) ^ (m - k) :=
    pow_pos
      hOneMinusX
      (m - k)

  have h1 :
      0 <
        (k : ℝ)
          *
        (m.choose k : ℝ) :=
    mul_pos
      hkReal
      hChooseReal

  have h2 :
      0 <
        (
          (k : ℝ)
            *
          (m.choose k : ℝ)
        )
          *
        x ^ (k - 1) :=
    mul_pos
      h1
      hxPow

  exact
    mul_pos
      h2
      hOneMinusXPow

/--
For `0 < k ≤ m`, the derivative of the upper-tail polynomial is
strictly positive at every point of `(0,1)`.
-/
theorem deriv_upperBinomialTailPolynomial_pos
    (m k : ℕ)
    (x : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hx0 :
      0 < x)
    (hx1 :
      x < 1) :
    0 <
      deriv
        (upperBinomialTailPolynomial m k)
        x := by

  rw [
    deriv_upperBinomialTailPolynomial
      m
      k
      x
      hkm
  ]

  exact
    upperBinomialTailDerivativeKernel_pos
      m
      k
      x
      hk
      hkm
      hx0
      hx1

/--
For `0 < k ≤ m`, the upper-binomial-tail polynomial is strictly
increasing on the closed unit interval.
-/
theorem strictMonoOn_upperBinomialTailPolynomial
    (m k : ℕ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m) :
    StrictMonoOn
      (upperBinomialTailPolynomial m k)
      (Set.Icc (0 : ℝ) 1) := by

  apply
    strictMonoOn_of_deriv_pos
      (convex_Icc (0 : ℝ) 1)
      (
        (
          continuous_upperBinomialTailPolynomial'
            m
            k
        ).continuousOn
      )

  intro x hx

  have hxIoo :
      x ∈ Set.Ioo (0 : ℝ) 1 := by
    simpa using hx

  exact
    deriv_upperBinomialTailPolynomial_pos
      m
      k
      x
      hk
      hkm
      hxIoo.1
      hxIoo.2

/--
For an admissible level, the selected upper-tail root is the unique
point of `[0,1]` at which the upper-tail polynomial equals `α`.
-/
theorem upperBinomialTailLevelRoot_unique
    (m k : ℕ)
    (α x : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1)
    (hx :
      x ∈ Set.Icc (0 : ℝ) 1)
    (hxLevel :
      upperBinomialTailPolynomial
          m
          k
          x
        =
      α) :
    x
      =
    upperBinomialTailLevelRoot
      m
      k
      α := by

  have hRootMem :
      upperBinomialTailLevelRoot
          m
          k
          α
        ∈
      Set.Icc (0 : ℝ) 1 :=
    upperBinomialTailLevelRoot_mem_Icc
      m
      k
      α
      hk
      hkm
      hα0
      hα1

  have hRootLevel :
      upperBinomialTailPolynomial
          m
          k
          (
            upperBinomialTailLevelRoot
              m
              k
              α
          )
        =
      α :=
    upperBinomialTailLevelRoot_eq
      m
      k
      α
      hk
      hkm
      hα0
      hα1

  have hEq :
      upperBinomialTailPolynomial
          m
          k
          x
        =
      upperBinomialTailPolynomial
          m
          k
          (
            upperBinomialTailLevelRoot
              m
              k
              α
          ) := by

    calc
      upperBinomialTailPolynomial
          m
          k
          x
          =
        α :=
        hxLevel

      _ =
        upperBinomialTailPolynomial
          m
          k
          (
            upperBinomialTailLevelRoot
              m
              k
              α
          ) :=
        hRootLevel.symm

  exact
    (
      strictMonoOn_upperBinomialTailPolynomial
        m
        k
        hk
        hkm
    ).injOn
      hx
      hRootMem
      hEq

/--
There is exactly one point in `[0,1]` satisfying the upper-tail level
equation.
-/
theorem existsUnique_upperBinomialTailPolynomial_eq
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
    ∃! x : ℝ,
      x ∈ Set.Icc (0 : ℝ) 1
        ∧
      upperBinomialTailPolynomial
          m
          k
          x
        =
      α := by

  refine
    ⟨
      upperBinomialTailLevelRoot
        m
        k
        α,
      ?_,
      ?_
    ⟩

  · exact
      ⟨
        upperBinomialTailLevelRoot_mem_Icc
          m
          k
          α
          hk
          hkm
          hα0
          hα1,
        upperBinomialTailLevelRoot_eq
          m
          k
          α
          hk
          hkm
          hα0
          hα1
      ⟩

  · intro x hx

    exact
      upperBinomialTailLevelRoot_unique
        m
        k
        α
        x
        hk
        hkm
        hα0
        hα1
        hx.1
        hx.2

end DROSafety.DRO
