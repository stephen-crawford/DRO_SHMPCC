import DROSafety.DRO.BinomialTailPolynomial
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Endpoint values of the binomial tail polynomials

For the later exact Clopper-Pearson endpoint construction we will define
the endpoints as roots of the corresponding binomial-tail equations.

The intermediate-value argument needs the endpoint values:

* for `k ≤ m`,

      upperBinomialTailPolynomial m k 1 = 1;

* for every `k`,

      lowerBinomialTailPolynomial m k 0 = 1.

Together with the already-proved identities

    upperBinomialTailPolynomial m k 0 = 0    when 0 < k,

and

    lowerBinomialTailPolynomial m k 1 = 0    when k < m,

the relevant tail functions span the whole interval `[0,1]`.
-/

namespace DROSafety.DRO

/--
For `k ≤ m`, the upper binomial-tail polynomial equals `1` at `x = 1`.

At `x = 1`, every term with `j < m` vanishes because it contains the
positive power `(1 - 1)^(m-j)`. The only surviving term is `j = m`.
-/
theorem upperBinomialTailPolynomial_one
    (m k : ℕ)
    (hkm :
      k ≤ m) :
    upperBinomialTailPolynomial
        m
        k
        1
      =
    1 := by

  unfold upperBinomialTailPolynomial

  have hmMem :
      m ∈ Finset.Icc k m :=
    Finset.mem_Icc.mpr
      ⟨hkm, le_rfl⟩

  rw [
    Finset.sum_eq_single_of_mem
      m
      hmMem
  ]

  · simp

  · intro j hj hjNe

    have hjBounds :
        k ≤ j ∧ j ≤ m :=
      Finset.mem_Icc.mp
        hj

    have hjm :
        j < m := by

      have hjLe :
          j ≤ m :=
        hjBounds.2

      have hjNotEq :
          j ≠ m :=
        hjNe

      exact
        lt_of_le_of_ne
          hjLe
          hjNotEq

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
The lower binomial-tail polynomial equals `1` at `x = 0`.

At `x = 0`, every term with `j > 0` vanishes because it contains
`0^j`. The only surviving term is `j = 0`.
-/
theorem lowerBinomialTailPolynomial_zero
    (m k : ℕ) :
    lowerBinomialTailPolynomial
        m
        k
        0
      =
    1 := by

  unfold lowerBinomialTailPolynomial

  have hZeroMem :
      0 ∈ Finset.Iic k :=
    Finset.mem_Iic.mpr
      (Nat.zero_le k)

  rw [
    Finset.sum_eq_single_of_mem
      0
      hZeroMem
  ]

  · simp

  · intro j hj hjNe

    simp [hjNe]

end DROSafety.DRO
