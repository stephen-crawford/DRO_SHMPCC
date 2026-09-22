import DROSafety.DRO.BinomialBetaUpperNormalization
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Bridge from the upper binomial-tail kernel to the Beta PDF

For `0 < k ≤ m` and `0 < x < 1`, the derivative kernel

    k * C(m,k) * x^(k-1) * (1-x)^(m-k)

is exactly Mathlib's real-valued Beta PDF with parameters

    α = k
    β = m-k+1.

Combining this with the previous derivative theorem gives

    d/dx upperBinomialTailPolynomial m k x
      =
    betaPDFReal k (m-k+1) x

on the open unit interval.
-/

namespace DROSafety.DRO

open ProbabilityTheory

/--
The second upper-tail Beta exponent simplifies to the natural number
`m-k`.
-/
theorem upper_beta_second_exponent
    (m k : ℕ) :
    (
      ((m - k : ℕ) : ℝ)
        +
      1
    )
      -
    1
      =
    ((m - k : ℕ) : ℝ) := by

  ring

/--
The first upper-tail Beta exponent is the cast of the natural exponent
`k - 1`.
-/
theorem upper_beta_first_exponent
    (k : ℕ)
    (hk :
      0 < k) :
    (k : ℝ) - 1
      =
    ((k - 1 : ℕ) : ℝ) := by

  have hkOne :
      1 ≤ k := by
    omega

  have hNat :
      k - 1 + 1 = k :=
    Nat.sub_add_cancel
      hkOne

  have hReal :
      ((k - 1 : ℕ) : ℝ) + 1
        =
      (k : ℝ) := by
    exact_mod_cast hNat

  linarith

/--
On the interior of the unit interval, the upper-tail Beta PDF is exactly
the natural-power Beta kernel.
-/
theorem betaPDFReal_upper_eq_betaKernel
    (m k : ℕ)
    (x : ℝ)
    (hk :
      0 < k)
    (hx0 :
      0 < x)
    (hx1 :
      x < 1) :
    ProbabilityTheory.betaPDFReal
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
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

  unfold ProbabilityTheory.betaPDFReal

  rw [
    if_pos
      ⟨hx0, hx1⟩
  ]

  have hFirst :
      (k : ℝ) - 1
        =
      ((k - 1 : ℕ) : ℝ) :=
    upper_beta_first_exponent
      k
      hk

  have hSecond :
      (
        ((m - k : ℕ) : ℝ)
          +
        1
      )
        -
      1
        =
      ((m - k : ℕ) : ℝ) :=
    upper_beta_second_exponent
      m
      k

  rw [
    hFirst,
    hSecond
  ]

  rw [
    Real.rpow_natCast,
    Real.rpow_natCast
  ]

/--
For `0 < k ≤ m`, the Beta PDF equals the already-defined upper
binomial-tail derivative kernel on `(0,1)`.
-/
theorem betaPDFReal_upper_eq_upperKernel
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
    ProbabilityTheory.betaPDFReal
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
        x
      =
    upperBinomialTailDerivativeKernel
        m
        k
        x := by

  rw [
    betaPDFReal_upper_eq_betaKernel
      m
      k
      x
      hk
      hx0
      hx1
  ]

  symm

  exact
    upperBinomialTailDerivativeKernel_eq_betaKernel
      m
      k
      x
      hk
      hkm

/--
The derivative of the upper binomial-tail polynomial is exactly the
Beta PDF on the open unit interval.
-/
theorem deriv_upperBinomialTailPolynomial_eq_betaPDFReal
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
    deriv
        (upperBinomialTailPolynomial m k)
        x
      =
    ProbabilityTheory.betaPDFReal
      (k : ℝ)
      (
        ((m - k : ℕ) : ℝ)
          +
        1
      )
      x := by

  rw [
    deriv_upperBinomialTailPolynomial
      m
      k
      x
      hkm
  ]

  symm

  exact
    betaPDFReal_upper_eq_upperKernel
      m
      k
      x
      hk
      hkm
      hx0
      hx1

/--
`HasDerivAt` formulation of the Beta-PDF derivative identity on `(0,1)`.
-/
theorem hasDerivAt_upperBinomialTailPolynomial_eq_betaPDFReal
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
    HasDerivAt
      (upperBinomialTailPolynomial m k)
      (
        ProbabilityTheory.betaPDFReal
          (k : ℝ)
          (
            ((m - k : ℕ) : ℝ)
              +
            1
          )
          x
      )
      x := by

  have h :
      HasDerivAt
        (upperBinomialTailPolynomial m k)
        (
          upperBinomialTailDerivativeKernel
            m
            k
            x
        )
        x :=
    hasDerivAt_upperBinomialTailPolynomial
      m
      k
      x
      hkm

  rw [
    ← betaPDFReal_upper_eq_upperKernel
      m
      k
      x
      hk
      hkm
      hx0
      hx1
  ] at h

  exact h

end DROSafety.DRO
