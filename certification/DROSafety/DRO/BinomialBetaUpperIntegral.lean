import DROSafety.DRO.BinomialBetaPDFBridge
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Upper binomial tail as an integral of the Beta PDF

The previous files established, for `0 < k ≤ m`,

    d/dx T_{m,k}(x)
      =
    betaPDFReal k (m-k+1) x

for `0 < x < 1`.

Rather than applying the fundamental theorem directly to `betaPDFReal`,
we first integrate the polynomial derivative kernel

    upperBinomialTailDerivativeKernel m k,

which is continuous everywhere.

We then use equality of that kernel with `betaPDFReal` on the open
interval `(0,p)`.

This yields

    T_{m,k}(p)
      =
    ∫ x in 0..p,
      betaPDFReal k (m-k+1) x

for `0 ≤ p ≤ 1`.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory
open intervalIntegral

/--
The upper-tail derivative kernel is continuous as a real-valued
polynomial function.
-/
theorem continuous_upperBinomialTailDerivativeKernel
    (m k : ℕ) :
    Continuous
      (upperBinomialTailDerivativeKernel m k) := by

  unfold upperBinomialTailDerivativeKernel

  fun_prop

/--
The upper-tail polynomial is continuous.
-/
theorem continuous_upperBinomialTailPolynomial
    (m k : ℕ) :
    Continuous
      (upperBinomialTailPolynomial m k) := by

  unfold upperBinomialTailPolynomial

  fun_prop

/--
The polynomial derivative kernel is interval-integrable on every finite
interval.
-/
theorem intervalIntegrable_upperBinomialTailDerivativeKernel
    (m k : ℕ)
    (a b : ℝ) :
    IntervalIntegrable
      (upperBinomialTailDerivativeKernel m k)
      MeasureTheory.volume
      a
      b := by

  exact
    (
      continuous_upperBinomialTailDerivativeKernel
        m
        k
    ).intervalIntegrable
      a
      b

/--
Integrating the upper-tail derivative kernel from `0` to `p` recovers
the upper-tail polynomial.

The assumption `0 < k` gives

    T_{m,k}(0) = 0.
-/
theorem integral_upperKernel_eq_upperBinomialTailPolynomial
    (m k : ℕ)
    (p : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hp0 :
      0 ≤ p) :
    (∫ x in (0 : ℝ)..p,
      upperBinomialTailDerivativeKernel
        m
        k
        x)
      =
    upperBinomialTailPolynomial
      m
      k
      p := by

  have hContinuous :
      ContinuousOn
        (upperBinomialTailPolynomial m k)
        (Set.Icc (0 : ℝ) p) :=
    (
      continuous_upperBinomialTailPolynomial
        m
        k
    ).continuousOn

  have hDerivative :
      ∀ x ∈ Set.Ioo (0 : ℝ) p,
        HasDerivAt
          (upperBinomialTailPolynomial m k)
          (
            upperBinomialTailDerivativeKernel
              m
              k
              x
          )
          x := by

    intro x hx

    exact
      hasDerivAt_upperBinomialTailPolynomial
        m
        k
        x
        hkm

  have hIntegrable :
      IntervalIntegrable
        (upperBinomialTailDerivativeKernel m k)
        MeasureTheory.volume
        0
        p :=
    intervalIntegrable_upperBinomialTailDerivativeKernel
      m
      k
      0
      p

  have hFTC :
      (∫ x in (0 : ℝ)..p,
        upperBinomialTailDerivativeKernel
          m
          k
          x)
        =
      upperBinomialTailPolynomial
          m
          k
          p
        -
      upperBinomialTailPolynomial
          m
          k
          0 :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le
      hp0
      hContinuous
      hDerivative
      hIntegrable

  rw [
    upperBinomialTailPolynomial_zero
      m
      k
      hk
  ] at hFTC

  simpa using hFTC

/--
On an interval contained in `[0,1]`, the polynomial derivative kernel
and the Beta PDF have the same interval integral.

Endpoint values do not matter because interval integration is unchanged
when two functions agree on the open interval.
-/
theorem integral_upperKernel_eq_betaPDFReal
    (m k : ℕ)
    (p : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hp0 :
      0 ≤ p)
    (hp1 :
      p ≤ 1) :
    (∫ x in (0 : ℝ)..p,
      upperBinomialTailDerivativeKernel
        m
        k
        x)
      =
    ∫ x in (0 : ℝ)..p,
      ProbabilityTheory.betaPDFReal
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
        x := by

  apply
    intervalIntegral.integral_congr_Ioo_of_le
      hp0

  intro x hx

  have hx0 :
      0 < x :=
    hx.1

  have hxp :
      x < p :=
    hx.2

  have hx1 :
      x < 1 :=
    lt_of_lt_of_le
      hxp
      hp1

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
The upper-binomial-tail polynomial is exactly the interval integral of
the corresponding Beta PDF from `0` to `p`.
-/
theorem upperBinomialTailPolynomial_eq_integral_betaPDFReal
    (m k : ℕ)
    (p : ℝ)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hp0 :
      0 ≤ p)
    (hp1 :
      p ≤ 1) :
    upperBinomialTailPolynomial
        m
        k
        p
      =
    ∫ x in (0 : ℝ)..p,
      ProbabilityTheory.betaPDFReal
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
        x := by

  calc
    upperBinomialTailPolynomial
        m
        k
        p
        =
      ∫ x in (0 : ℝ)..p,
        upperBinomialTailDerivativeKernel
          m
          k
          x := by

            symm

            exact
              integral_upperKernel_eq_upperBinomialTailPolynomial
                m
                k
                p
                hk
                hkm
                hp0

    _ =
      ∫ x in (0 : ℝ)..p,
        ProbabilityTheory.betaPDFReal
          (k : ℝ)
          (
            ((m - k : ℕ) : ℝ)
              +
            1
          )
          x :=
        integral_upperKernel_eq_betaPDFReal
          m
          k
          p
          hk
          hkm
          hp0
          hp1

/--
Specialization to a Bernoulli/binomial probability parameter
`p : unitInterval`.
-/
theorem upperBinomialTailPolynomial_eq_integral_betaPDFReal_unitInterval
    (m k : ℕ)
    (p : unitInterval)
    (hk :
      0 < k)
    (hkm :
      k ≤ m) :
    upperBinomialTailPolynomial
        m
        k
        (p : ℝ)
      =
    ∫ x in (0 : ℝ)..(p : ℝ),
      ProbabilityTheory.betaPDFReal
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
        x := by

  have hp0 :
      0 ≤ (p : ℝ) :=
    p.property.1

  have hp1 :
      (p : ℝ) ≤ 1 :=
    p.property.2

  exact
    upperBinomialTailPolynomial_eq_integral_betaPDFReal
      m
      k
      (p : ℝ)
      hk
      hkm
      hp0
      hp1

/--
The real-valued upper binomial probability is the corresponding Beta
PDF interval integral.
-/
theorem binomial_real_Ici_eq_integral_betaPDFReal
    (m k : ℕ)
    (p : unitInterval)
    (hk :
      0 < k)
    (hkm :
      k ≤ m) :
    (ProbabilityTheory.binomial m p).real
        (Set.Ici k)
      =
    ∫ x in (0 : ℝ)..(p : ℝ),
      ProbabilityTheory.betaPDFReal
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
        x := by

  rw [
    binomial_real_Ici_eq_upperTailPolynomial
      m
      p
      k
  ]

  exact
    upperBinomialTailPolynomial_eq_integral_betaPDFReal_unitInterval
      m
      k
      p
      hk
      hkm

end DROSafety.DRO
