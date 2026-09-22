import DROSafety.DRO.BinomialTailSupport
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Binomial tail polynomials

The previous files proved that the real-valued binomial upper and lower
tails are finite sums of the binomial PMF.

This file packages those finite sums as real functions:

    upperBinomialTailPolynomial m k x
      = sum_{j=k}^m C(m,j) x^j (1-x)^(m-j),

and

    lowerBinomialTailPolynomial m k x
      = sum_{j=0}^k C(m,j) x^j (1-x)^(m-j).

The binomial probabilities are therefore obtained simply by evaluating
these functions at the true Bernoulli parameter.

The next file will prove the derivative identities

    d/dx upperTail
      =
    k * C(m,k) * x^(k-1) * (1-x)^(m-k),

and

    d/dx lowerTail
      =
    -(m-k) * C(m,k) * x^k * (1-x)^(m-k-1),

under the appropriate index assumptions.

Those derivatives are exactly the Beta-density kernels appearing in the
Clopper-Pearson construction.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
Real upper-binomial-tail polynomial

    sum_{j=k}^m C(m,j) x^j (1-x)^(m-j).
-/
def upperBinomialTailPolynomial
    (m k : ℕ)
    (x : ℝ) :
    ℝ :=
  ∑ j ∈ Finset.Icc k m,
    (m.choose j : ℝ)
      *
    x ^ j
      *
    (1 - x) ^ (m - j)

/--
Real lower-binomial-tail polynomial

    sum_{j=0}^k C(m,j) x^j (1-x)^(m-j).
-/
def lowerBinomialTailPolynomial
    (m k : ℕ)
    (x : ℝ) :
    ℝ :=
  ∑ j ∈ Finset.Iic k,
    (m.choose j : ℝ)
      *
    x ^ j
      *
    (1 - x) ^ (m - j)

/--
The upper binomial tail probability, converted to `ℝ`, is exactly the
upper-tail polynomial evaluated at `p`.
-/
theorem binomial_real_Ici_eq_upperTailPolynomial
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    (ProbabilityTheory.binomial m p).real
        (Set.Ici k)
      =
    upperBinomialTailPolynomial
      m
      k
      (p : ℝ) := by

  rw [
    binomial_real_Ici_eq_explicit_sum
      m
      p
      k
  ]

  rfl

/--
The lower binomial tail probability, converted to `ℝ`, is exactly the
lower-tail polynomial evaluated at `p`.
-/
theorem binomial_real_Iic_eq_lowerTailPolynomial
    (m : ℕ)
    (p : unitInterval)
    (k : ℕ) :
    (ProbabilityTheory.binomial m p).real
        (Set.Iic k)
      =
    lowerBinomialTailPolynomial
      m
      k
      (p : ℝ) := by

  rw [
    binomial_real_Iic_eq_explicit_sum
      m
      p
      k
  ]

  rfl

/--
If the lower endpoint of the upper tail is above `m`, the upper-tail
polynomial is identically zero.
-/
theorem upperBinomialTailPolynomial_eq_zero_of_lt
    (m k : ℕ)
    (x : ℝ)
    (hmk :
      m < k) :
    upperBinomialTailPolynomial
        m
        k
        x
      =
    0 := by

  unfold upperBinomialTailPolynomial

  have hEmpty :
      Finset.Icc k m = ∅ :=
    Finset.Icc_eq_empty_of_lt
      hmk

  rw [hEmpty]

  simp

/--
For positive `k`, the upper-tail polynomial vanishes at zero.
-/
theorem upperBinomialTailPolynomial_zero
    (m k : ℕ)
    (hk :
      0 < k) :
    upperBinomialTailPolynomial
        m
        k
        0
      =
    0 := by

  unfold upperBinomialTailPolynomial

  apply Finset.sum_eq_zero

  intro j hj

  have hjBounds :
      k ≤ j ∧ j ≤ m := by
    simpa using
      (Finset.mem_Icc.mp hj)

  have hjPos :
      0 < j :=
    lt_of_lt_of_le
      hk
      hjBounds.1

  have hjNe :
      j ≠ 0 :=
    Nat.ne_of_gt
      hjPos

  simp [hjNe]

/--
For `k < m`, the lower-tail polynomial vanishes at one.

Every term has a strictly positive power of `(1-x)`.
-/
theorem lowerBinomialTailPolynomial_one
    (m k : ℕ)
    (hkm :
      k < m) :
    lowerBinomialTailPolynomial
        m
        k
        1
      =
    0 := by

  unfold lowerBinomialTailPolynomial

  apply Finset.sum_eq_zero

  intro j hj

  have hjLe :
      j ≤ k := by
    simpa using
      (Finset.mem_Iic.mp hj)

  have hjm :
      j < m :=
    lt_of_le_of_lt
      hjLe
      hkm

  have hSubPos :
      0 < m - j :=
    Nat.sub_pos_of_lt
      hjm

  have hSubNe :
      m - j ≠ 0 :=
    Nat.ne_of_gt
      hSubPos

  simp [hSubNe]

/--
Kernel appearing in the derivative of the upper-tail polynomial.

For `0 < k ≤ m`,

    k * C(m,k) * x^(k-1) * (1-x)^(m-k)

is the normalized Beta `(k,m-k+1)` density kernel.
-/
def upperBinomialTailDerivativeKernel
    (m k : ℕ)
    (x : ℝ) :
    ℝ :=
  (k : ℝ)
    *
  (m.choose k : ℝ)
    *
  x ^ (k - 1)
    *
  (1 - x) ^ (m - k)

/--
Kernel appearing in minus the derivative of the lower-tail polynomial.

For `k < m`,

    (m-k) * C(m,k) * x^k * (1-x)^(m-k-1)

is the normalized Beta `(k+1,m-k)` density kernel.

The coefficient uses natural-number subtraction first and is then cast
to `ℝ`.
-/
def lowerBinomialTailDerivativeKernel
    (m k : ℕ)
    (x : ℝ) :
    ℝ :=
  ((m - k : ℕ) : ℝ)
    *
  (m.choose k : ℝ)
    *
  x ^ k
    *
  (1 - x) ^ (m - k - 1)

/--
The upper derivative kernel is nonnegative on `[0,1]`.
-/
theorem upperBinomialTailDerivativeKernel_nonneg
    (m k : ℕ)
    {x : ℝ}
    (hx0 :
      0 ≤ x)
    (hx1 :
      x ≤ 1) :
    0 ≤
      upperBinomialTailDerivativeKernel
        m
        k
        x := by

  unfold upperBinomialTailDerivativeKernel

  have hkNonneg :
      0 ≤ (k : ℝ) :=
    Nat.cast_nonneg k

  have hChooseNonneg :
      0 ≤ (m.choose k : ℝ) :=
    Nat.cast_nonneg
      (m.choose k)

  have hxPowNonneg :
      0 ≤ x ^ (k - 1) :=
    pow_nonneg
      hx0
      (k - 1)

  have hOneMinusX :
      0 ≤ 1 - x :=
    sub_nonneg.mpr
      hx1

  have hOneMinusXPowNonneg :
      0 ≤ (1 - x) ^ (m - k) :=
    pow_nonneg
      hOneMinusX
      (m - k)

  have h1 :
      0 ≤
        (k : ℝ)
          *
        (m.choose k : ℝ) :=
    mul_nonneg
      hkNonneg
      hChooseNonneg

  have h2 :
      0 ≤
        (
          (k : ℝ)
            *
          (m.choose k : ℝ)
        )
          *
        x ^ (k - 1) :=
    mul_nonneg
      h1
      hxPowNonneg

  exact
    mul_nonneg
      h2
      hOneMinusXPowNonneg

/--
The lower derivative kernel is nonnegative on `[0,1]`.
-/
theorem lowerBinomialTailDerivativeKernel_nonneg
    (m k : ℕ)
    {x : ℝ}
    (hx0 :
      0 ≤ x)
    (hx1 :
      x ≤ 1) :
    0 ≤
      lowerBinomialTailDerivativeKernel
        m
        k
        x := by

  unfold lowerBinomialTailDerivativeKernel

  have hSubNonneg :
      0 ≤ ((m - k : ℕ) : ℝ) :=
    Nat.cast_nonneg
      (m - k)

  have hChooseNonneg :
      0 ≤ (m.choose k : ℝ) :=
    Nat.cast_nonneg
      (m.choose k)

  have hxPowNonneg :
      0 ≤ x ^ k :=
    pow_nonneg
      hx0
      k

  have hOneMinusX :
      0 ≤ 1 - x :=
    sub_nonneg.mpr
      hx1

  have hOneMinusXPowNonneg :
      0 ≤ (1 - x) ^ (m - k - 1) :=
    pow_nonneg
      hOneMinusX
      (m - k - 1)

  have h1 :
      0 ≤
        ((m - k : ℕ) : ℝ)
          *
        (m.choose k : ℝ) :=
    mul_nonneg
      hSubNonneg
      hChooseNonneg

  have h2 :
      0 ≤
        (
          ((m - k : ℕ) : ℝ)
            *
          (m.choose k : ℝ)
        )
          *
        x ^ k :=
    mul_nonneg
      h1
      hxPowNonneg

  exact
    mul_nonneg
      h2
      hOneMinusXPowNonneg

end DROSafety.DRO
