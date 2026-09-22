import DROSafety.DRO.ClopperPearsonExactQuantileReal
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# ENNReal-valued exact Clopper-Pearson quantile equations

The previous file established the exact Beta-tail equations in
`Measure.real`:

    Beta(...).real (...) = α.toReal.

The abstract `BetaClopperPearsonEndpointSpec`, however, states its tail
budgets in `ENNReal`.

This file lifts the real-valued equalities back to exact `ENNReal`
measure equalities.

The only required assumption is that the tail budget is at most `1`,
which guarantees that it is finite and that its `toReal` representation
is faithful.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory
open Set

/--
An `ENNReal` bounded above by `1` is finite.
-/
theorem ennreal_ne_top_of_le_one
    (α : ENNReal)
    (hα :
      α ≤ 1) :
    α ≠ ⊤ := by

  exact
    ne_top_of_le_ne_top
      ENNReal.one_ne_top
      hα

/--
If `α ≤ 1`, then its real representative also satisfies
`α.toReal ≤ 1`.
-/
theorem ennreal_toReal_le_one
    (α : ENNReal)
    (hα :
      α ≤ 1) :
    α.toReal ≤ 1 := by

  have hαTop :
      α ≠ ⊤ :=
    ennreal_ne_top_of_le_one
      α
      hα

  have h :
      α ≤ ENNReal.ofReal (1 : ℝ) := by
    simpa using hα

  exact
    (
      ENNReal.le_ofReal_iff_toReal_le
        hαTop
        (by norm_num)
    ).mp h

/--
Two finite `ENNReal` values are equal if their real representatives are
equal.
-/
theorem ennreal_eq_of_toReal_eq
    {a b : ENNReal}
    (ha :
      a ≠ ⊤)
    (hb :
      b ≠ ⊤)
    (h :
      a.toReal = b.toReal) :
    a = b := by

  have h' :
      ENNReal.ofReal a.toReal
        =
      ENNReal.ofReal b.toReal :=
    congrArg
      ENNReal.ofReal
      h

  rw [
    ENNReal.ofReal_toReal
      ha,
    ENNReal.ofReal_toReal
      hb
  ] at h'

  exact h'

/--
Exact ENNReal lower-tail quantile equation for the exact lower
Clopper-Pearson endpoint.

For `0 < k ≤ m` and `α ≤ 1`,

    Beta(k,m-k+1) (-∞,L(k)] = α.
-/
theorem betaMeasure_Iic_exactLowerEndpoint
    (m k : ℕ)
    (α : ENNReal)
    (hk :
      0 < k)
    (hkm :
      k ≤ m)
    (hα :
      α ≤ 1) :
    ProbabilityTheory.betaMeasure
        (k : ℝ)
        (
          (m : ℝ)
            -
          (k : ℝ)
            +
          1
        )
        (
          Set.Iic
            (
              exactClopperPearsonLowerEndpoint
                m
                α.toReal
                k
            )
        )
      =
    α := by

  have hαTop :
      α ≠ ⊤ :=
    ennreal_ne_top_of_le_one
      α
      hα

  have hα0 :
      0 ≤ α.toReal :=
    ENNReal.toReal_nonneg

  have hα1 :
      α.toReal ≤ 1 :=
    ennreal_toReal_le_one
      α
      hα

  have hkReal :
      0 < (k : ℝ) := by
    exact_mod_cast hk

  have hkmReal :
      (k : ℝ) ≤ (m : ℝ) := by
    exact_mod_cast hkm

  have hSecond :
      0 <
        (m : ℝ)
          -
        (k : ℝ)
          +
        1 := by
    linarith

  letI :
      MeasureTheory.IsProbabilityMeasure
        (
          ProbabilityTheory.betaMeasure
            (k : ℝ)
            (
              (m : ℝ)
                -
              (k : ℝ)
                +
              1
            )
        ) :=
    ProbabilityTheory.isProbabilityMeasureBeta
      hkReal
      hSecond

  have hReal :
      (
        ProbabilityTheory.betaMeasure
          (k : ℝ)
          (
            (m : ℝ)
              -
            (k : ℝ)
              +
            1
          )
      ).real
        (
          Set.Iic
            (
              exactClopperPearsonLowerEndpoint
                m
                α.toReal
                k
            )
        )
        =
      α.toReal :=
    betaMeasure_real_Iic_exactLowerEndpoint
      m
      k
      α.toReal
      hk
      hkm
      hα0
      hα1

  have hMeasureTop :
      ProbabilityTheory.betaMeasure
          (k : ℝ)
          (
            (m : ℝ)
              -
            (k : ℝ)
              +
            1
          )
          (
            Set.Iic
              (
                exactClopperPearsonLowerEndpoint
                  m
                  α.toReal
                  k
              )
          )
        ≠
      ⊤ := by
    finiteness

  have hToReal :
      (
        ProbabilityTheory.betaMeasure
          (k : ℝ)
          (
            (m : ℝ)
              -
            (k : ℝ)
              +
            1
          )
          (
            Set.Iic
              (
                exactClopperPearsonLowerEndpoint
                  m
                  α.toReal
                  k
              )
          )
      ).toReal
        =
      α.toReal := by

    simpa [
      MeasureTheory.measureReal_def
    ] using hReal

  exact
    ennreal_eq_of_toReal_eq
      hMeasureTop
      hαTop
      hToReal

/--
Exact ENNReal upper-tail quantile equation for the exact upper
Clopper-Pearson endpoint.

For `k < m` and `α ≤ 1`,

    Beta(k+1,m-k) [U(k),∞) = α.
-/
theorem betaMeasure_Ici_exactUpperEndpoint
    (m k : ℕ)
    (α : ENNReal)
    (hkm :
      k < m)
    (hα :
      α ≤ 1) :
    ProbabilityTheory.betaMeasure
        (
          (k : ℝ)
            +
          1
        )
        (
          (m : ℝ)
            -
          (k : ℝ)
        )
        (
          Set.Ici
            (
              exactClopperPearsonUpperEndpoint
                m
                α.toReal
                k
            )
        )
      =
    α := by

  have hαTop :
      α ≠ ⊤ :=
    ennreal_ne_top_of_le_one
      α
      hα

  have hα0 :
      0 ≤ α.toReal :=
    ENNReal.toReal_nonneg

  have hα1 :
      α.toReal ≤ 1 :=
    ennreal_toReal_le_one
      α
      hα

  have hkSuccReal :
      0 <
        (k : ℝ) + 1 := by
    positivity

  have hkmReal :
      (k : ℝ) < (m : ℝ) := by
    exact_mod_cast hkm

  have hSecond :
      0 <
        (m : ℝ)
          -
        (k : ℝ) := by
    linarith

  letI :
      MeasureTheory.IsProbabilityMeasure
        (
          ProbabilityTheory.betaMeasure
            (
              (k : ℝ)
                +
              1
            )
            (
              (m : ℝ)
                -
              (k : ℝ)
            )
        ) :=
    ProbabilityTheory.isProbabilityMeasureBeta
      hkSuccReal
      hSecond

  have hReal :
      (
        ProbabilityTheory.betaMeasure
          (
            (k : ℝ)
              +
            1
          )
          (
            (m : ℝ)
              -
            (k : ℝ)
          )
      ).real
        (
          Set.Ici
            (
              exactClopperPearsonUpperEndpoint
                m
                α.toReal
                k
            )
        )
        =
      α.toReal :=
    betaMeasure_real_Ici_exactUpperEndpoint
      m
      k
      α.toReal
      hkm
      hα0
      hα1

  have hMeasureTop :
      ProbabilityTheory.betaMeasure
          (
            (k : ℝ)
              +
            1
          )
          (
            (m : ℝ)
              -
            (k : ℝ)
          )
          (
            Set.Ici
              (
                exactClopperPearsonUpperEndpoint
                  m
                  α.toReal
                  k
              )
          )
        ≠
      ⊤ := by
    finiteness

  have hToReal :
      (
        ProbabilityTheory.betaMeasure
          (
            (k : ℝ)
              +
            1
          )
          (
            (m : ℝ)
              -
            (k : ℝ)
          )
          (
            Set.Ici
              (
                exactClopperPearsonUpperEndpoint
                  m
                  α.toReal
                  k
              )
          )
      ).toReal
        =
      α.toReal := by

    simpa [
      MeasureTheory.measureReal_def
    ] using hReal

  exact
    ennreal_eq_of_toReal_eq
      hMeasureTop
      hαTop
      hToReal

end DROSafety.DRO
