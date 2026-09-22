import DROSafety.DRO.BinomialTailTelescoping
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Derivative of the upper binomial tail polynomial

We have already proved:

1. the derivative of each individual binomial PMF term;
2. that these derivatives have boundary-difference form;
3. that the boundary differences telescope.

This file combines those results to prove

    d/dx upperBinomialTailPolynomial m k x
      =
    upperBinomialTailDerivativeKernel m k x

whenever `k ≤ m`.

Equivalently,

    d/dx sum_{j=k}^m C(m,j) x^j (1-x)^(m-j)
      =
    k C(m,k) x^(k-1) (1-x)^(m-k).
-/

namespace DROSafety.DRO

/--
The upper-tail polynomial is exactly the finite sum of the packaged
binomial PMF terms.
-/
theorem upperBinomialTailPolynomial_eq_sum_binomialPMFTerm
    (m k : ℕ)
    (x : ℝ) :
    upperBinomialTailPolynomial
        m
        k
        x
      =
    ∑ j ∈ Finset.Icc k m,
      binomialPMFTerm
        m
        j
        x := by

  unfold
    upperBinomialTailPolynomial
    binomialPMFTerm

  apply Finset.sum_congr rfl

  intro j hj

  ring

/--
Function-valued version of
`upperBinomialTailPolynomial_eq_sum_binomialPMFTerm`.
-/
theorem upperBinomialTailPolynomial_fun_eq_sum_binomialPMFTerm
    (m k : ℕ) :
    upperBinomialTailPolynomial
        m
        k
      =
    fun x : ℝ =>
      ∑ j ∈ Finset.Icc k m,
        binomialPMFTerm
          m
          j
          x := by

  funext x

  exact
    upperBinomialTailPolynomial_eq_sum_binomialPMFTerm
      m
      k
      x

/--
Each packaged binomial PMF term is differentiable everywhere.
-/
theorem differentiableAt_binomialPMFTerm
    (m j : ℕ)
    (x : ℝ) :
    DifferentiableAt
      ℝ
      (binomialPMFTerm m j)
      x := by

  exact
    (
      hasDerivAt_binomialPMFTerm
        m
        j
        x
    ).differentiableAt

/--
Differentiate the finite sum of packaged PMF terms term-by-term.
-/
theorem deriv_sum_binomialPMFTerm
    (m k : ℕ)
    (x : ℝ) :
    deriv
      (fun y : ℝ =>
        ∑ j ∈ Finset.Icc k m,
          binomialPMFTerm
            m
            j
            y)
      x
      =
    ∑ j ∈ Finset.Icc k m,
      binomialPMFTermDerivative
        m
        j
        x := by

  have hDiff :
      ∀ j ∈ Finset.Icc k m,
        DifferentiableAt
          ℝ
          (binomialPMFTerm m j)
          x := by

    intro j hj

    exact
      differentiableAt_binomialPMFTerm
        m
        j
        x

  rw [
    deriv_fun_sum
      hDiff
  ]

  apply Finset.sum_congr rfl

  intro j hj

  exact
    deriv_binomialPMFTerm
      m
      j
      x

/--
Derivative of the full upper-binomial-tail polynomial.

For `k ≤ m`,

    T'_{m,k}(x)
      =
    k C(m,k) x^(k-1) (1-x)^(m-k).
-/
theorem deriv_upperBinomialTailPolynomial
    (m k : ℕ)
    (x : ℝ)
    (hkm :
      k ≤ m) :
    deriv
      (upperBinomialTailPolynomial m k)
      x
      =
    upperBinomialTailDerivativeKernel
      m
      k
      x := by

  rw [
    upperBinomialTailPolynomial_fun_eq_sum_binomialPMFTerm
      m
      k
  ]

  rw [
    deriv_sum_binomialPMFTerm
      m
      k
      x
  ]

  exact
    sum_binomialPMFTermDerivative_eq_upperKernel
      m
      k
      x
      hkm

/--
The upper-tail polynomial is differentiable everywhere.
-/
theorem differentiableAt_upperBinomialTailPolynomial
    (m k : ℕ)
    (x : ℝ) :
    DifferentiableAt
      ℝ
      (upperBinomialTailPolynomial m k)
      x := by

  unfold upperBinomialTailPolynomial

  fun_prop

/--
`HasDerivAt` formulation of the upper-tail derivative identity.

This form will be useful when connecting the polynomial derivative to the
Beta integral using the fundamental theorem of calculus.
-/
theorem hasDerivAt_upperBinomialTailPolynomial
    (m k : ℕ)
    (x : ℝ)
    (hkm :
      k ≤ m) :
    HasDerivAt
      (upperBinomialTailPolynomial m k)
      (upperBinomialTailDerivativeKernel m k x)
      x := by

  have hDiff :
      DifferentiableAt
        ℝ
        (upperBinomialTailPolynomial m k)
        x :=
    differentiableAt_upperBinomialTailPolynomial
      m
      k
      x

  have hDeriv :
      HasDerivAt
        (upperBinomialTailPolynomial m k)
        (
          deriv
            (upperBinomialTailPolynomial m k)
            x
        )
        x :=
    hDiff.hasDerivAt

  rw [
    deriv_upperBinomialTailPolynomial
      m
      k
      x
      hkm
  ] at hDeriv

  exact hDeriv

end DROSafety.DRO
