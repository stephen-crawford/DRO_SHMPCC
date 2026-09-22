import DROSafety.DRO.BinomialTailBoundaryDifference
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Telescoping the upper binomial-tail derivative

The previous file proved

    b'_{m,j}(x)
      =
    B_{m,j}(x) - B_{m,j+1}(x).

This file telescopes that identity over the finite interval `j ∈ [k,m]`.

For `k ≤ m`,

    sum_{j=k}^m
      (B_{m,j}(x) - B_{m,j+1}(x))
        =
      B_{m,k}(x) - B_{m,m+1}(x).

The terminal boundary term is zero, so

    sum_{j=k}^m b'_{m,j}(x)
      =
    B_{m,k}(x),

which is exactly the upper-binomial-tail derivative kernel.
-/

namespace DROSafety.DRO

/--
Generic telescoping identity over a natural-number interval.

For any fixed function `f`,

    sum_{j=k}^n (f j - f (j+1))
      =
    f k - f (n+1).

The function `f` is kept fixed during the induction. This is important
for the binomial application, where `m` is a parameter of the boundary
term and must not change while telescoping over the index.
-/
theorem sum_Icc_sub_succ
    (f : ℕ → ℝ)
    (n k : ℕ)
    (hkn :
      k ≤ n) :
    (∑ j ∈ Finset.Icc k n,
      (f j - f (j + 1)))
      =
    f k - f (n + 1) := by

  induction n generalizing k with

  | zero =>

      have hk :
          k = 0 := by
        omega

      subst k

      simp

  | succ n ih =>

      by_cases hkTop :
          k = n + 1

      · subst k

        simp

      · have hknPrev :
            k ≤ n := by
          omega

        rw [
          Finset.sum_Icc_succ_top
            hkn
        ]

        rw [
          ih
            k
            hknPrev
        ]

        ring

/--
A finite sum of adjacent binomial boundary differences over `[k,m]`
telescopes to the first boundary term minus the boundary term after `m`.
-/
theorem binomialBoundaryDifferences_telescope
    (m k : ℕ)
    (x : ℝ)
    (hkm :
      k ≤ m) :
    (∑ j ∈ Finset.Icc k m,
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
      ))
      =
    binomialTailBoundaryTerm
        m
        k
        x
      -
    binomialTailBoundaryTerm
        m
        (m + 1)
        x := by

  exact
    sum_Icc_sub_succ
      (fun j =>
        binomialTailBoundaryTerm
          m
          j
          x)
      m
      k
      hkm

/--
The sum of the PMF-term derivatives over the upper-tail interval
`[k,m]` is the first boundary term.
-/
theorem upperBinomialTailDerivativeSum_eq_boundaryTerm
    (m k : ℕ)
    (x : ℝ)
    (hkm :
      k ≤ m) :
    upperBinomialTailDerivativeSum
        m
        k
        x
      =
    binomialTailBoundaryTerm
        m
        k
        x := by

  rw [
    upperBinomialTailDerivativeSum_eq_boundaryDifferences
      m
      k
      x
  ]

  rw [
    binomialBoundaryDifferences_telescope
      m
      k
      x
      hkm
  ]

  rw [
    binomialTailBoundaryTerm_succ_trials_zero
      m
      x
  ]

  simp

/--
The finite sum of the derivatives of the upper-tail PMF terms is exactly
the upper-binomial-tail derivative kernel.
-/
theorem upperBinomialTailDerivativeSum_eq_upperKernel
    (m k : ℕ)
    (x : ℝ)
    (hkm :
      k ≤ m) :
    upperBinomialTailDerivativeSum
        m
        k
        x
      =
    upperBinomialTailDerivativeKernel
        m
        k
        x := by

  rw [
    upperBinomialTailDerivativeSum_eq_boundaryTerm
      m
      k
      x
      hkm
  ]

  exact
    binomialTailBoundaryTerm_eq_upperKernel
      m
      k
      x

/--
Expanded version of the telescoping result.

This exposes the finite sum directly rather than through
`upperBinomialTailDerivativeSum`, which will be convenient when
differentiating the whole upper-tail polynomial.
-/
theorem sum_binomialPMFTermDerivative_eq_upperKernel
    (m k : ℕ)
    (x : ℝ)
    (hkm :
      k ≤ m) :
    (∑ j ∈ Finset.Icc k m,
      binomialPMFTermDerivative
        m
        j
        x)
      =
    upperBinomialTailDerivativeKernel
        m
        k
        x := by

  exact
    upperBinomialTailDerivativeSum_eq_upperKernel
      m
      k
      x
      hkm

end DROSafety.DRO
