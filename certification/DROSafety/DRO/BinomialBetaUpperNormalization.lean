import DROSafety.DRO.BinomialUpperTailDerivative
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Beta normalization for the upper binomial tail

For natural parameters satisfying

    0 < k ≤ m,

the Beta normalizing constant satisfies

    1 / Beta(k, m-k+1)
      =
    k * C(m,k).

This identifies the coefficient in the derivative of the upper
binomial-tail polynomial with the normalization coefficient of the
Beta(k, m-k+1) density.

Combined with `BinomialUpperTailDerivative.lean`, this gives

    T'_{m,k}(x)
      =
    (1 / Beta(k,m-k+1))
      * x^(k-1)
      * (1-x)^(m-k),

where the powers on the right are still written as natural powers.

A later file will reconcile these natural powers with the real powers
used by Mathlib's `betaPDFReal`.
-/

namespace DROSafety.DRO

open ProbabilityTheory

/--
For positive `k`,

    Gamma(k) = (k-1)!.
-/
theorem Gamma_nat_pos_eq_factorial_pred
    (k : ℕ)
    (hk :
      0 < k) :
    Real.Gamma (k : ℝ)
      =
    ((k - 1).factorial : ℝ) := by

  have hkOne :
      1 ≤ k :=
    Nat.succ_le_iff.mpr
      hk

  have hkNat :
      k - 1 + 1 = k :=
    Nat.sub_add_cancel
      hkOne

  have hkReal :
      (((k - 1 : ℕ) : ℝ) + 1)
        =
      (k : ℝ) := by
    exact_mod_cast hkNat

  rw [
    ← hkReal
  ]

  exact
    Real.Gamma_nat_eq_factorial
      (k - 1)

/--
For any natural `n`,

    Gamma(n+1) = n!.
-/
theorem Gamma_cast_add_one_eq_factorial
    (n : ℕ) :
    Real.Gamma
        (((n : ℕ) : ℝ) + 1)
      =
    (n.factorial : ℝ) := by

  exact
    Real.Gamma_nat_eq_factorial
      n

/--
For `k ≤ m`, the sum of the two upper-tail Beta parameters is `m+1`.
-/
theorem upperBetaParameters_sum
    (m k : ℕ)
    (hkm :
      k ≤ m) :
    (k : ℝ)
        +
      (
        ((m - k : ℕ) : ℝ)
          +
        1
      )
      =
    (m : ℝ) + 1 := by

  have hNat :
      k + (m - k) + 1
        =
      m + 1 := by
    omega

  exact_mod_cast hNat

/--
Explicit factorial formula for the Beta normalizing constant with
upper-binomial-tail parameters:

    Beta(k,m-k+1)
      =
    (k-1)! (m-k)! / m!.
-/
theorem beta_upper_nat_eq_factorial_ratio
    (m k : ℕ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m) :
    ProbabilityTheory.beta
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
      =
    ((k - 1).factorial : ℝ)
      *
    ((m - k).factorial : ℝ)
      /
    (m.factorial : ℝ) := by

  have hGammaK :
      Real.Gamma (k : ℝ)
        =
      ((k - 1).factorial : ℝ) :=
    Gamma_nat_pos_eq_factorial_pred
      k
      hk

  have hGammaRest :
      Real.Gamma
          (
            ((m - k : ℕ) : ℝ)
              +
            1
          )
        =
      ((m - k).factorial : ℝ) :=
    Real.Gamma_nat_eq_factorial
      (m - k)

  have hSum :
      (k : ℝ)
          +
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
        =
      (m : ℝ) + 1 :=
    upperBetaParameters_sum
      m
      k
      hkm

  have hGammaSum :
      Real.Gamma
          (
            (k : ℝ)
              +
            (
              ((m - k : ℕ) : ℝ)
                +
              1
            )
          )
        =
      (m.factorial : ℝ) := by

    rw [
      hSum
    ]

    exact
      Real.Gamma_nat_eq_factorial
        m

  unfold ProbabilityTheory.beta

  rw [
    hGammaK,
    hGammaRest,
    hGammaSum
  ]

/--
The real cast of `C(m,k)` has the standard factorial formula.
-/
theorem choose_cast_eq_factorial_ratio
    (m k : ℕ)
    (hkm :
      k ≤ m) :
    (m.choose k : ℝ)
      =
    (m.factorial : ℝ)
      /
    (
      (k.factorial : ℝ)
        *
      ((m - k).factorial : ℝ)
    ) := by

  exact
    Nat.cast_choose
      ℝ
      hkm

/--
For positive `k`,

    k! = k (k-1)!

after casting to `ℝ`.
-/
theorem factorial_cast_eq_mul_factorial_pred
    (k : ℕ)
    (hk :
      0 < k) :
    (k.factorial : ℝ)
      =
    (k : ℝ)
      *
    ((k - 1).factorial : ℝ) := by

  have hkNe :
      k ≠ 0 :=
    Nat.ne_of_gt
      hk

  have hNat :
      k
          *
        (k - 1).factorial
        =
      k.factorial :=
    Nat.mul_factorial_pred
      hkNe

  exact_mod_cast hNat.symm

/--
The reciprocal Beta normalizer for the upper binomial tail is exactly

    k * C(m,k).
-/
theorem one_div_beta_upper_nat_eq_choose
    (m k : ℕ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m) :
    1
        /
      ProbabilityTheory.beta
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
      =
    (k : ℝ)
      *
    (m.choose k : ℝ) := by

  rw [
    beta_upper_nat_eq_factorial_ratio
      m
      k
      hk
      hkm
  ]

  rw [
    choose_cast_eq_factorial_ratio
      m
      k
      hkm
  ]

  rw [
    factorial_cast_eq_mul_factorial_pred
      k
      hk
  ]

  have hkNeNat :
      k ≠ 0 :=
    Nat.ne_of_gt
      hk

  have hkNe :
      (k : ℝ) ≠ 0 := by
    exact_mod_cast hkNeNat

  have hPredFactNe :
      ((k - 1).factorial : ℝ) ≠ 0 := by
    exact_mod_cast
      Nat.factorial_ne_zero
        (k - 1)

  have hRestFactNe :
      ((m - k).factorial : ℝ) ≠ 0 := by
    exact_mod_cast
      Nat.factorial_ne_zero
        (m - k)

  have hMFactNe :
      (m.factorial : ℝ) ≠ 0 := by
    exact_mod_cast
      Nat.factorial_ne_zero
        m

  field_simp [
    hkNe,
    hPredFactNe,
    hRestFactNe,
    hMFactNe
  ]

  <;> ring

/--
The upper-binomial-tail derivative kernel is the naturally-powered Beta
kernel with parameters `(k,m-k+1)`.
-/
theorem upperBinomialTailDerivativeKernel_eq_betaKernel
    (m k : ℕ)
    (x : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m) :
    upperBinomialTailDerivativeKernel
        m
        k
        x
      =
    (
      1
        /
      ProbabilityTheory.beta
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
    )
      *
    x ^ (k - 1)
      *
    (1 - x) ^ (m - k) := by

  rw [
    one_div_beta_upper_nat_eq_choose
      m
      k
      hk
      hkm
  ]

  unfold upperBinomialTailDerivativeKernel

  ring

/--
Combining the polynomial derivative theorem with the Beta normalization
gives the upper-tail derivative directly in Beta-kernel form.
-/
theorem deriv_upperBinomialTailPolynomial_eq_betaKernel
    (m k : ℕ)
    (x : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m) :
    deriv
        (upperBinomialTailPolynomial m k)
        x
      =
    (
      1
        /
      ProbabilityTheory.beta
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
    )
      *
    x ^ (k - 1)
      *
    (1 - x) ^ (m - k) := by

  rw [
    deriv_upperBinomialTailPolynomial
      m
      k
      x
      hkm
  ]

  exact
    upperBinomialTailDerivativeKernel_eq_betaKernel
      m
      k
      x
      hk
      hkm

end DROSafety.DRO
