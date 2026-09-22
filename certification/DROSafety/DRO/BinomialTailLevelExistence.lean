import DROSafety.DRO.BinomialTailEndpointValues
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Existence of binomial tail level roots

For the upper-tail polynomial, when `0 < k ≤ m`,

    T⁺_{m,k}(0) = 0,
    T⁺_{m,k}(1) = 1.

Hence every level `α ∈ [0,1]` is attained somewhere in `[0,1]`.

For the lower-tail polynomial, when `k < m`,

    T⁻_{m,k}(0) = 1,
    T⁻_{m,k}(1) = 0.

Hence every level `α ∈ [0,1]` is likewise attained somewhere in
`[0,1]`.

These existence results are the basis for constructing exact
Clopper-Pearson endpoints.
-/

namespace DROSafety.DRO

/--
The upper binomial-tail polynomial is continuous.
-/
theorem continuous_upperBinomialTailPolynomial'
    (m k : ℕ) :
    Continuous
      (upperBinomialTailPolynomial m k) := by

  unfold upperBinomialTailPolynomial

  fun_prop

/--
The lower binomial-tail polynomial is continuous.
-/
theorem continuous_lowerBinomialTailPolynomial
    (m k : ℕ) :
    Continuous
      (lowerBinomialTailPolynomial m k) := by

  unfold lowerBinomialTailPolynomial

  fun_prop

/--
For `0 < k ≤ m`, every level `α ∈ [0,1]` is attained by the upper
binomial-tail polynomial at some point of `[0,1]`.
-/
theorem exists_upperBinomialTailPolynomial_eq
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
    ∃ x ∈ Set.Icc (0 : ℝ) 1,
      upperBinomialTailPolynomial
          m
          k
          x
        =
      α := by

  have hCont :
      ContinuousOn
        (upperBinomialTailPolynomial m k)
        (Set.Icc (0 : ℝ) 1) :=
    (
      continuous_upperBinomialTailPolynomial'
        m
        k
    ).continuousOn

  have hImage :
      Set.Icc
          (
            upperBinomialTailPolynomial
              m
              k
              0
          )
          (
            upperBinomialTailPolynomial
              m
              k
              1
          )
        ⊆
      upperBinomialTailPolynomial
          m
          k
        ''
      Set.Icc (0 : ℝ) 1 :=
    intermediate_value_Icc
      (show (0 : ℝ) ≤ 1 by norm_num)
      hCont

  have hαMem :
      α ∈
        Set.Icc
          (
            upperBinomialTailPolynomial
              m
              k
              0
          )
          (
            upperBinomialTailPolynomial
              m
              k
              1
          ) := by

    rw [
      upperBinomialTailPolynomial_zero
        m
        k
        hk,
      upperBinomialTailPolynomial_one
        m
        k
        hkm
    ]

    exact
      ⟨hα0, hα1⟩

  have hαImage :
      α ∈
        upperBinomialTailPolynomial
            m
            k
          ''
        Set.Icc (0 : ℝ) 1 :=
    hImage
      hαMem

  rcases hαImage with
    ⟨x, hx, hEq⟩

  exact
    ⟨x, hx, hEq⟩

/--
For `k < m`, every level `α ∈ [0,1]` is attained by the lower
binomial-tail polynomial at some point of `[0,1]`.

The lower-tail polynomial runs from `1` at zero to `0` at one, so we use
the decreasing-orientation version of the intermediate value theorem.
-/
theorem exists_lowerBinomialTailPolynomial_eq
    (m k : ℕ)
    (α : ℝ)
    (hkm :
      k < m)
    (hα0 :
      0 ≤ α)
    (hα1 :
      α ≤ 1) :
    ∃ x ∈ Set.Icc (0 : ℝ) 1,
      lowerBinomialTailPolynomial
          m
          k
          x
        =
      α := by

  have hCont :
      ContinuousOn
        (lowerBinomialTailPolynomial m k)
        (Set.Icc (0 : ℝ) 1) :=
    (
      continuous_lowerBinomialTailPolynomial
        m
        k
    ).continuousOn

  have hImage :
      Set.Icc
          (
            lowerBinomialTailPolynomial
              m
              k
              1
          )
          (
            lowerBinomialTailPolynomial
              m
              k
              0
          )
        ⊆
      lowerBinomialTailPolynomial
          m
          k
        ''
      Set.Icc (0 : ℝ) 1 :=
    intermediate_value_Icc'
      (show (0 : ℝ) ≤ 1 by norm_num)
      hCont

  have hαMem :
      α ∈
        Set.Icc
          (
            lowerBinomialTailPolynomial
              m
              k
              1
          )
          (
            lowerBinomialTailPolynomial
              m
              k
              0
          ) := by

    rw [
      lowerBinomialTailPolynomial_one
        m
        k
        hkm,
      lowerBinomialTailPolynomial_zero
        m
        k
    ]

    exact
      ⟨hα0, hα1⟩

  have hαImage :
      α ∈
        lowerBinomialTailPolynomial
            m
            k
          ''
        Set.Icc (0 : ℝ) 1 :=
    hImage
      hαMem

  rcases hαImage with
    ⟨x, hx, hEq⟩

  exact
    ⟨x, hx, hEq⟩

end DROSafety.DRO
