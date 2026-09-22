import DROSafety.DRO.BinomialBetaUpperMeasure
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Lower binomial tail equals the corresponding Beta upper tail

The upper-tail identity already proved gives, at index `k+1`,

    Bin(m,p) [k+1,∞)
      =
    Beta(k+1,m-k) (-∞,p].

Taking complements gives

    Bin(m,p) (-∞,k]
      =
    Beta(k+1,m-k) (p,∞).

Because the Beta measure is absolutely continuous with respect to
Lebesgue measure, singleton sets have measure zero. Hence

    Beta(k+1,m-k) (p,∞)
      =
    Beta(k+1,m-k) [p,∞).

Therefore, for `k < m`,

    Bin(m,p) [0,k]
      =
    Beta(k+1,m-k) [p,∞).

This is precisely the lower-tail identity required by
`BinomialBetaTailBridge`.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory
open Set

/--
For natural-valued counts,

    complement (-∞,k] = [k+1,∞).
-/
theorem nat_compl_Iic_eq_Ici_succ
    (k : ℕ) :
    (Set.Iic k)ᶜ
      =
    Set.Ici (k + 1) := by

  ext n

  simp only [
    Set.mem_compl_iff,
    Set.mem_Iic,
    Set.mem_Ici
  ]

  omega

/--
For real numbers,

    complement (-∞,p] = (p,∞).
-/
theorem real_compl_Iic_eq_Ioi
    (p : ℝ) :
    (Set.Iic p)ᶜ
      =
    Set.Ioi p := by

  ext x

  simp

/--
For `k < m`, the second Beta parameter obtained by applying the upper
tail identity at index `k+1` simplifies to `m-k`:

    m - (k+1) + 1 = m-k.
-/
theorem shifted_upper_beta_second_parameter
    (m k : ℕ)
    (hkm :
      k < m) :
    (
      ((m - (k + 1) : ℕ) : ℝ)
        +
      1
    )
      =
    ((m - k : ℕ) : ℝ) := by

  have hNat :
      m - (k + 1) + 1
        =
      m - k := by
    omega

  exact_mod_cast hNat

/--
A Beta measure assigns equal mass to `(p,∞)` and `[p,∞)` because it has
zero mass on the singleton `{p}`.

`betaMeasure` is definitionally a `withDensity` measure.  We expose that
definition locally so Mathlib can synthesize the `NullSingletonClass`
instance inherited from Lebesgue measure.
-/
theorem betaMeasure_real_Ioi_eq_Ici
    (α β p : ℝ) :
    (
      ProbabilityTheory.betaMeasure
        α
        β
    ).real
      (Set.Ioi p)
      =
    (
      ProbabilityTheory.betaMeasure
        α
        β
    ).real
      (Set.Ici p) := by

  letI :
      MeasureTheory.NullSingletonClass
        (
          ProbabilityTheory.betaMeasure
            α
            β
        ) := by

    unfold ProbabilityTheory.betaMeasure

    infer_instance

  have hMeasure :
      ProbabilityTheory.betaMeasure
          α
          β
          (Set.Ioi p)
        =
      ProbabilityTheory.betaMeasure
          α
          β
          (Set.Ici p) := by

    apply MeasureTheory.measure_congr

    exact
      MeasureTheory.Ioi_ae_eq_Ici
        (μ :=
          ProbabilityTheory.betaMeasure
            α
            β)

  unfold Measure.real

  rw [hMeasure]

/--
The real-valued lower binomial tail equals the real-valued Beta upper
tail.

For `k < m`,

    Bin(m,p).real [0,k]
      =
    Beta(k+1,m-k).real [p,∞).
-/
theorem binomial_real_Iic_eq_betaMeasure_real_Ici
    (m k : ℕ)
    (p : unitInterval)
    (hkm :
      k < m) :
    (
      ProbabilityTheory.binomial
        m
        p
    ).real
      (Set.Iic k)
      =
    (
      ProbabilityTheory.betaMeasure
        (((k + 1 : ℕ) : ℝ))
        (((m - k : ℕ) : ℝ))
    ).real
      (Set.Ici (p : ℝ)) := by

  have hkSucc :
      0 < k + 1 :=
    Nat.succ_pos k

  have hkSuccLe :
      k + 1 ≤ m := by
    omega

  have hSecondNat :
      0 < m - k :=
    Nat.sub_pos_of_lt
      hkm

  have hAlpha :
      0 < (((k + 1 : ℕ) : ℝ)) := by
    exact_mod_cast hkSucc

  have hBeta :
      0 < (((m - k : ℕ) : ℝ)) := by
    exact_mod_cast hSecondNat

  letI :
      MeasureTheory.IsProbabilityMeasure
        (
          ProbabilityTheory.betaMeasure
            (((k + 1 : ℕ) : ℝ))
            (((m - k : ℕ) : ℝ))
        ) :=
    ProbabilityTheory.isProbabilityMeasureBeta
      hAlpha
      hBeta

  have hUpper :
      (
        ProbabilityTheory.binomial
          m
          p
      ).real
        (Set.Ici (k + 1))
        =
      (
        ProbabilityTheory.betaMeasure
          (((k + 1 : ℕ) : ℝ))
          (((m - k : ℕ) : ℝ))
      ).real
        (Set.Iic (p : ℝ)) := by

    have h :=
      binomial_real_Ici_eq_betaMeasure_real_Iic
        m
        (k + 1)
        p
        hkSucc
        hkSuccLe

    have hParam :
        (
          ((m - (k + 1) : ℕ) : ℝ)
            +
          1
        )
          =
        ((m - k : ℕ) : ℝ) :=
      shifted_upper_beta_second_parameter
        m
        k
        hkm

    rw [hParam] at h

    exact h

  have hBinComplement :
      (
        ProbabilityTheory.binomial
          m
          p
      ).real
          (Set.Iic k)
        +
      (
        ProbabilityTheory.binomial
          m
          p
      ).real
          (Set.Ici (k + 1))
        =
      1 := by

    have h :=
      MeasureTheory.probReal_add_probReal_compl
        (μ :=
          ProbabilityTheory.binomial
            m
            p)
        (s := Set.Iic k)
        measurableSet_Iic

    rw [
      nat_compl_Iic_eq_Ici_succ
        k
    ] at h

    exact h

  have hBetaComplement :
      (
        ProbabilityTheory.betaMeasure
          (((k + 1 : ℕ) : ℝ))
          (((m - k : ℕ) : ℝ))
      ).real
          (Set.Iic (p : ℝ))
        +
      (
        ProbabilityTheory.betaMeasure
          (((k + 1 : ℕ) : ℝ))
          (((m - k : ℕ) : ℝ))
      ).real
          (Set.Ioi (p : ℝ))
        =
      1 := by

    have h :=
      MeasureTheory.probReal_add_probReal_compl
        (μ :=
          ProbabilityTheory.betaMeasure
            (((k + 1 : ℕ) : ℝ))
            (((m - k : ℕ) : ℝ)))
        (s := Set.Iic (p : ℝ))
        measurableSet_Iic

    rw [
      real_compl_Iic_eq_Ioi
        (p : ℝ)
    ] at h

    exact h

  have hLowerOpen :
      (
        ProbabilityTheory.binomial
          m
          p
      ).real
          (Set.Iic k)
        =
      (
        ProbabilityTheory.betaMeasure
          (((k + 1 : ℕ) : ℝ))
          (((m - k : ℕ) : ℝ))
      ).real
          (Set.Ioi (p : ℝ)) := by

    rw [hUpper] at hBinComplement

    linarith

  calc
    (
      ProbabilityTheory.binomial
        m
        p
    ).real
      (Set.Iic k)
        =
      (
        ProbabilityTheory.betaMeasure
          (((k + 1 : ℕ) : ℝ))
          (((m - k : ℕ) : ℝ))
      ).real
        (Set.Ioi (p : ℝ)) :=
      hLowerOpen

    _ =
      (
        ProbabilityTheory.betaMeasure
          (((k + 1 : ℕ) : ℝ))
          (((m - k : ℕ) : ℝ))
      ).real
        (Set.Ici (p : ℝ)) :=
      betaMeasure_real_Ioi_eq_Ici
        (((k + 1 : ℕ) : ℝ))
        (((m - k : ℕ) : ℝ))
        (p : ℝ)

/--
Exact lower binomial/Beta tail identity.

For `k < m`,

    Bin(m,p) [0,k]
      =
    Beta(k+1,m-k) [p,∞).

This is the lower-tail identity required by
`BinomialBetaTailBridge`.
-/
theorem binomial_Iic_eq_betaMeasure_Ici
    (m k : ℕ)
    (p : unitInterval)
    (hkm :
      k < m) :
    ProbabilityTheory.binomial
        m
        p
        (Set.Iic k)
      =
    ProbabilityTheory.betaMeasure
        (((k + 1 : ℕ) : ℝ))
        (((m - k : ℕ) : ℝ))
        (Set.Ici (p : ℝ)) := by

  have hkSucc :
      0 < k + 1 :=
    Nat.succ_pos k

  have hSecondNat :
      0 < m - k :=
    Nat.sub_pos_of_lt
      hkm

  have hAlpha :
      0 < (((k + 1 : ℕ) : ℝ)) := by
    exact_mod_cast hkSucc

  have hBeta :
      0 < (((m - k : ℕ) : ℝ)) := by
    exact_mod_cast hSecondNat

  letI :
      MeasureTheory.IsProbabilityMeasure
        (
          ProbabilityTheory.betaMeasure
            (((k + 1 : ℕ) : ℝ))
            (((m - k : ℕ) : ℝ))
        ) :=
    ProbabilityTheory.isProbabilityMeasureBeta
      hAlpha
      hBeta

  apply
    (
      MeasureTheory.measureReal_eq_measureReal_iff
    ).mp

  exact
    binomial_real_Iic_eq_betaMeasure_real_Ici
      m
      k
      p
      hkm

end DROSafety.DRO
