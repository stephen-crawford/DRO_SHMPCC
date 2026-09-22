import DROSafety.DRO.BinomialTailTermDerivative
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Boundary-difference form of the binomial PMF derivative

For the boundary term

    B_{m,j}(x)
      =
    j * C(m,j) * x^(j-1) * (1-x)^(m-j),

the derivative of one binomial PMF term satisfies

    b'_{m,j}(x)
      =
    B_{m,j}(x) - B_{m,j+1}(x).

This is the exact algebraic form needed for telescoping.

We also prove

    B_{m,m+1}(x) = 0,

so summing from `j = k` through `m` leaves only `B_{m,k}(x)`.
-/

namespace DROSafety.DRO

/--
Boundary term used in the telescoping derivative calculation:

    B_{m,j}(x)
      =
    j * C(m,j) * x^(j-1) * (1-x)^(m-j).
-/
def binomialTailBoundaryTerm
    (m j : ℕ)
    (x : ℝ) :
    ℝ :=
  (j : ℝ)
    *
  (m.choose j : ℝ)
    *
  x ^ (j - 1)
    *
  (1 - x) ^ (m - j)

/--
The successor boundary term can be written in the form appearing as the
negative part of the derivative of PMF term `j`.
-/
theorem binomialTailBoundaryTerm_succ
    (m j : ℕ)
    (x : ℝ) :
    binomialTailBoundaryTerm
        m
        (j + 1)
        x
      =
    ((m - j : ℕ) : ℝ)
      *
    (m.choose j : ℝ)
      *
    x ^ j
      *
    (1 - x) ^ (m - j - 1) := by

  unfold binomialTailBoundaryTerm

  rw [
    succ_sub_one_exponent
      j
  ]

  rw [
    sub_succ_exponent
      m
      j
  ]

  rw [
    succ_mul_choose_succ_eq_real
      m
      j
  ]

/--
The derivative expression of a single PMF term is exactly the difference
of two adjacent boundary terms:

    b'_{m,j}(x)
      =
    B_{m,j}(x) - B_{m,j+1}(x).
-/
theorem binomialPMFTermDerivative_eq_boundary_sub
    (m j : ℕ)
    (x : ℝ) :
    binomialPMFTermDerivative
        m
        j
        x
      =
    binomialTailBoundaryTerm
        m
        j
        x
      -
    binomialTailBoundaryTerm
        m
        (j + 1)
        x := by

  rw [
    binomialTailBoundaryTerm_succ
      m
      j
      x
  ]

  unfold
    binomialPMFTermDerivative
    binomialTailBoundaryTerm

  ring

/--
The terminal boundary term after `m` vanishes because

    C(m,m+1) = 0.
-/
theorem binomialTailBoundaryTerm_succ_trials_zero
    (m : ℕ)
    (x : ℝ) :
    binomialTailBoundaryTerm
        m
        (m + 1)
        x
      =
    0 := by

  unfold binomialTailBoundaryTerm

  simp

/--
The boundary term at `k` is exactly the upper-tail derivative kernel
introduced in `BinomialTailPolynomial.lean`.
-/
theorem binomialTailBoundaryTerm_eq_upperKernel
    (m k : ℕ)
    (x : ℝ) :
    binomialTailBoundaryTerm
        m
        k
        x
      =
    upperBinomialTailDerivativeKernel
        m
        k
        x := by

  rfl

/--
Finite sum of the individual PMF derivatives over the upper-tail
index interval `[k,m]`.
-/
def upperBinomialTailDerivativeSum
    (m k : ℕ)
    (x : ℝ) :
    ℝ :=
  ∑ j ∈ Finset.Icc k m,
    binomialPMFTermDerivative
      m
      j
      x

/--
Rewrite the entire upper-tail derivative sum as a sum of adjacent
boundary differences.
-/
theorem upperBinomialTailDerivativeSum_eq_boundaryDifferences
    (m k : ℕ)
    (x : ℝ) :
    upperBinomialTailDerivativeSum
        m
        k
        x
      =
    ∑ j ∈ Finset.Icc k m,
      (
        binomialTailBoundaryTerm
            m
            j
            x
          -
        binomialTailBoundaryTerm
            m
            (j + 1)
            x
      ) := by

  unfold upperBinomialTailDerivativeSum

  apply Finset.sum_congr rfl

  intro j hj

  exact
    binomialPMFTermDerivative_eq_boundary_sub
      m
      j
      x

end DROSafety.DRO
