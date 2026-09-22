import DROSafety.DRO.BinomialTailPolynomial
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Derivative of a single binomial PMF term

This file isolates the calculus and combinatorial identities needed for
the telescoping derivative proof of the binomial tail polynomial.

For

    b_{m,j}(x)
      =
    C(m,j) x^j (1-x)^(m-j),

we prove

    b'_{m,j}(x)
      =
    C(m,j) *
      (
        j x^(j-1) (1-x)^(m-j)
        -
        (m-j) x^j (1-x)^(m-j-1)
      ).

We also record the coefficient identity

    C(m,j+1) (j+1)
      =
    C(m,j) (m-j),

which is what makes adjacent derivative terms telescope.
-/

namespace DROSafety.DRO

/--
A single term of the binomial PMF polynomial.

The parentheses are chosen so that the constant binomial coefficient is
an outer constant factor.  This makes the derivative proof use
`deriv_const_mul_field` directly.
-/
def binomialPMFTerm
    (m j : ℕ)
    (x : ℝ) :
    ℝ :=
  (m.choose j : ℝ)
    *
  (
    x ^ j
      *
    (1 - x) ^ (m - j)
  )

/--
The expanded derivative expression for a single binomial PMF term.
-/
def binomialPMFTermDerivative
    (m j : ℕ)
    (x : ℝ) :
    ℝ :=
  (m.choose j : ℝ)
    *
  (
    (j : ℝ)
      *
    x ^ (j - 1)
      *
    (1 - x) ^ (m - j)
      -
    ((m - j : ℕ) : ℝ)
      *
    x ^ j
      *
    (1 - x) ^ (m - j - 1)
  )

/--
The map `x ↦ 1 - x` has derivative `-1`.

We derive this from the specialized field theorem for `c - x`, avoiding
the ambiguity between the different generic `Module ℝ ℝ` instances.
-/
theorem deriv_one_sub
    (x : ℝ) :
    deriv
      (fun y : ℝ => 1 - y)
      x
      =
    -1 := by

  exact
    deriv_const_sub_id
      (x := x)
      (1 : ℝ)

/--
The identity power `x ↦ x^j` has the usual derivative.
-/
theorem deriv_id_pow
    (j : ℕ)
    (x : ℝ) :
    deriv
      (fun y : ℝ => y ^ j)
      x
      =
    (j : ℝ)
      *
    x ^ (j - 1) := by

  exact
    deriv_pow_field
      j

/--
Derivative of `(1-x)^n`.
-/
theorem deriv_one_sub_pow
    (n : ℕ)
    (x : ℝ) :
    deriv
      (fun y : ℝ =>
        (1 - y) ^ n)
      x
      =
    (n : ℝ)
      *
    (1 - x) ^ (n - 1)
      *
    (-1) := by

  have hDiff :
      DifferentiableAt
        ℝ
        (fun y : ℝ => 1 - y)
        x := by
    fun_prop

  rw [
    deriv_fun_pow
      hDiff
      n
  ]

  rw [
    deriv_one_sub
      x
  ]

/--
Derivative of the product

    x^j (1-x)^(m-j).
-/
theorem deriv_binomialPowerProduct
    (m j : ℕ)
    (x : ℝ) :
    deriv
      (fun y : ℝ =>
        y ^ j
          *
        (1 - y) ^ (m - j))
      x
      =
    (j : ℝ)
      *
    x ^ (j - 1)
      *
    (1 - x) ^ (m - j)
      -
    ((m - j : ℕ) : ℝ)
      *
    x ^ j
      *
    (1 - x) ^ (m - j - 1) := by

  have hDiffLeft :
      DifferentiableAt
        ℝ
        (fun y : ℝ => y ^ j)
        x := by
    fun_prop

  have hDiffRight :
      DifferentiableAt
        ℝ
        (fun y : ℝ =>
          (1 - y) ^ (m - j))
        x := by
    fun_prop

  rw [
    deriv_fun_mul
      hDiffLeft
      hDiffRight
  ]

  rw [
    deriv_id_pow
      j
      x
  ]

  rw [
    deriv_one_sub_pow
      (m - j)
      x
  ]

  ring

/--
The derivative of a single binomial PMF polynomial term, stated first as
an equality of `deriv`.

This formulation is deliberately separated from `HasDerivAt`: all
calculus computation then takes place in `ℝ`, avoiding elaboration
ambiguities between generic normed-space structures.
-/
theorem deriv_binomialPMFTerm
    (m j : ℕ)
    (x : ℝ) :
    deriv
      (binomialPMFTerm m j)
      x
      =
    binomialPMFTermDerivative
      m
      j
      x := by

  unfold binomialPMFTerm

  have hDiffProduct :
      DifferentiableAt
        ℝ
        (fun y : ℝ =>
          y ^ j
            *
          (1 - y) ^ (m - j))
        x := by
    fun_prop

  rw [
    deriv_const_mul_field
  ]

  rw [
    deriv_binomialPowerProduct
      m
      j
      x
  ]

  unfold binomialPMFTermDerivative

  rfl

/--
Derivative of a single binomial PMF polynomial term.
-/
theorem hasDerivAt_binomialPMFTerm
    (m j : ℕ)
    (x : ℝ) :
    HasDerivAt
      (binomialPMFTerm m j)
      (binomialPMFTermDerivative m j x)
      x := by

  have hDiff :
      DifferentiableAt
        ℝ
        (binomialPMFTerm m j)
        x := by

    unfold binomialPMFTerm

    fun_prop

  have hDeriv :
      HasDerivAt
        (binomialPMFTerm m j)
        (deriv (binomialPMFTerm m j) x)
        x :=
    hDiff.hasDerivAt

  apply
    hDeriv.congr_deriv

  exact
    deriv_binomialPMFTerm
      m
      j
      x

/--
Natural-number adjacent binomial-coefficient identity in the orientation
needed for telescoping.
-/
theorem choose_succ_right_eq_nat
    (m j : ℕ) :
    m.choose (j + 1)
        *
      (j + 1)
      =
    m.choose j
        *
      (m - j) := by

  exact
    Nat.choose_succ_right_eq
      m
      j

/--
Real-cast version of the adjacent binomial-coefficient identity

    C(m,j+1) (j+1)
      =
    C(m,j) (m-j).
-/
theorem choose_succ_right_eq_real
    (m j : ℕ) :
    (m.choose (j + 1) : ℝ)
        *
      ((j + 1 : ℕ) : ℝ)
      =
    (m.choose j : ℝ)
        *
      ((m - j : ℕ) : ℝ) := by

  exact_mod_cast
    (choose_succ_right_eq_nat
      m
      j)

/--
The same coefficient identity with the factors ordered in the form used
by the derivative calculation.
-/
theorem succ_mul_choose_succ_eq_real
    (m j : ℕ) :
    ((j + 1 : ℕ) : ℝ)
        *
      (m.choose (j + 1) : ℝ)
      =
    ((m - j : ℕ) : ℝ)
        *
      (m.choose j : ℝ) := by

  have h :
      (m.choose (j + 1) : ℝ)
          *
        ((j + 1 : ℕ) : ℝ)
        =
      (m.choose j : ℝ)
          *
        ((m - j : ℕ) : ℝ) :=
    choose_succ_right_eq_real
      m
      j

  calc
    ((j + 1 : ℕ) : ℝ)
        *
      (m.choose (j + 1) : ℝ)
        =
      (m.choose (j + 1) : ℝ)
        *
      ((j + 1 : ℕ) : ℝ) := by
          ring

    _ =
      (m.choose j : ℝ)
        *
      ((m - j : ℕ) : ℝ) :=
        h

    _ =
      ((m - j : ℕ) : ℝ)
        *
      (m.choose j : ℝ) := by
          ring

/--
The exponent occurring in the negative part of term `j` is exactly the
`(1-x)` exponent occurring in the positive part of term `j+1`.
-/
theorem sub_succ_exponent
    (m j : ℕ) :
    m - (j + 1)
      =
    m - j - 1 := by

  omega

/--
The `x` exponent in the positive derivative contribution of term `j+1`
reduces to `j`.
-/
theorem succ_sub_one_exponent
    (j : ℕ) :
    j + 1 - 1
      =
    j := by

  omega

end DROSafety.DRO
