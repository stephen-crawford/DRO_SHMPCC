import DROSafety.DRO.BinomialBetaUpperIntegral
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Upper binomial tail equals the corresponding Beta measure

The previous file proved

    Bin(m,p).real [k,∞)
      =
    ∫ x in 0..p,
      betaPDFReal k (m-k+1) x.

This file identifies that interval integral with

    betaMeasure k (m-k+1) (-∞,p].

Hence, for `0 < k ≤ m`,

    Bin(m,p) [k,∞)
      =
    betaMeasure k (m-k+1) (-∞,p].

This is exactly the upper-tail field required by
`BinomialBetaTailBridge`.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory
open Set
open intervalIntegral

/--
For positive Beta parameters, the real-valued Beta PDF is nonnegative
everywhere.
-/
theorem betaPDFReal_nonneg
    (α β x : ℝ)
    (hα :
      0 < α)
    (hβ :
      0 < β) :
    0 ≤
      ProbabilityTheory.betaPDFReal
        α
        β
        x := by

  by_cases hx :
      0 < x ∧ x < 1

  · exact
      (
        ProbabilityTheory.betaPDFReal_pos
          hx.1
          hx.2
          hα
          hβ
      ).le

  · simp [
      ProbabilityTheory.betaPDFReal,
      hx
    ]

/--
The Beta measure assigns zero mass to `(-∞,0]`, since its density is zero
there.
-/
theorem betaMeasure_Iic_zero
    (α β : ℝ) :
    ProbabilityTheory.betaMeasure
        α
        β
        (Set.Iic (0 : ℝ))
      =
    0 := by

  rw [
    ProbabilityTheory.betaMeasure,
    MeasureTheory.withDensity_apply
      _
      measurableSet_Iic
  ]

  exact
    MeasureTheory.setLIntegral_eq_zero
      measurableSet_Iic
      (
        fun x hx =>
          ProbabilityTheory.betaPDF_eq_zero_of_nonpos
            hx
      )

/--
For `0 ≤ p`, all Beta mass in `(-∞,p]` is contained in `(0,p]`.
-/
theorem betaMeasure_Iic_eq_Ioc
    (α β p : ℝ)
    (hp0 :
      0 ≤ p) :
    ProbabilityTheory.betaMeasure
        α
        β
        (Set.Iic p)
      =
    ProbabilityTheory.betaMeasure
        α
        β
        (Set.Ioc 0 p) := by

  have hUnion :
      Set.Iic p
        =
      Set.Iic (0 : ℝ)
        ∪
      Set.Ioc 0 p := by

    ext x

    simp only [
      Set.mem_Iic,
      Set.mem_union,
      Set.mem_Ioc
    ]

    constructor

    · intro hx

      by_cases hx0 :
          x ≤ 0

      · exact
          Or.inl
            hx0

      · have hxPos :
            0 < x := by
          linarith

        exact
          Or.inr
            ⟨hxPos, hx⟩

    · intro hx

      rcases hx with hx | hx

      · exact
          hx.trans
            hp0

      · exact
          hx.2

  have hDisjoint :
      Disjoint
        (Set.Iic (0 : ℝ))
        (Set.Ioc 0 p) :=
    Set.Iic_disjoint_Ioc
      le_rfl

  rw [
    hUnion,
    MeasureTheory.measure_union
      hDisjoint
      measurableSet_Ioc,
    betaMeasure_Iic_zero
      α
      β,
    zero_add
  ]

/--
The real Beta measure of `(0,p]` is the ordinary integral of
`betaPDFReal` over that interval.
-/
theorem betaMeasure_real_Ioc_eq_integral_betaPDFReal
    (α β p : ℝ)
    (hα :
      0 < α)
    (hβ :
      0 < β)
    (hp0 :
      0 ≤ p) :
    (
      ProbabilityTheory.betaMeasure
        α
        β
    ).real
      (Set.Ioc 0 p)
      =
    ∫ x in (0 : ℝ)..p,
      ProbabilityTheory.betaPDFReal
        α
        β
        x := by

  have hNonneg :
      0 ≤ᵐ[
        MeasureTheory.volume.restrict
          (Set.Ioc (0 : ℝ) p)
      ]
        ProbabilityTheory.betaPDFReal
          α
          β :=

    Filter.Eventually.of_forall
      (
        fun x =>
          betaPDFReal_nonneg
            α
            β
            x
            hα
            hβ
      )

  have hStronglyMeasurable :
      AEStronglyMeasurable
        (
          ProbabilityTheory.betaPDFReal
            α
            β
        )
        (
          MeasureTheory.volume.restrict
            (Set.Ioc (0 : ℝ) p)
        ) :=
    (
      ProbabilityTheory.stronglyMeasurable_betaPDFReal
        α
        β
    ).aestronglyMeasurable

  have hIntegral :
      (
        ∫ x in Set.Ioc (0 : ℝ) p,
          ProbabilityTheory.betaPDFReal
            α
            β
            x
      )
        =
      (
        ∫⁻ x in Set.Ioc (0 : ℝ) p,
          ENNReal.ofReal
            (
              ProbabilityTheory.betaPDFReal
                α
                β
                x
            )
      ).toReal :=
    MeasureTheory.integral_eq_lintegral_of_nonneg_ae
      hNonneg
      hStronglyMeasurable

  change
    (
      ProbabilityTheory.betaMeasure
        α
        β
        (Set.Ioc 0 p)
    ).toReal
      =
    ∫ x in (0 : ℝ)..p,
      ProbabilityTheory.betaPDFReal
        α
        β
        x

  rw [
    ProbabilityTheory.betaMeasure,
    MeasureTheory.withDensity_apply
      _
      measurableSet_Ioc
  ]

  rw [
    intervalIntegral.integral_of_le
      hp0
  ]

  simpa [
    ProbabilityTheory.betaPDF
  ] using hIntegral.symm

/--
For `0 ≤ p`, the real Beta measure of `(-∞,p]` is the interval integral
of the Beta PDF from `0` to `p`.
-/
theorem betaMeasure_real_Iic_eq_integral_betaPDFReal
    (α β p : ℝ)
    (hα :
      0 < α)
    (hβ :
      0 < β)
    (hp0 :
      0 ≤ p) :
    (
      ProbabilityTheory.betaMeasure
        α
        β
    ).real
      (Set.Iic p)
      =
    ∫ x in (0 : ℝ)..p,
      ProbabilityTheory.betaPDFReal
        α
        β
        x := by

  change
    (
      ProbabilityTheory.betaMeasure
        α
        β
        (Set.Iic p)
    ).toReal
      =
    ∫ x in (0 : ℝ)..p,
      ProbabilityTheory.betaPDFReal
        α
        β
        x

  rw [
    betaMeasure_Iic_eq_Ioc
      α
      β
      p
      hp0
  ]

  exact
    betaMeasure_real_Ioc_eq_integral_betaPDFReal
      α
      β
      p
      hα
      hβ
      hp0

/--
The real-valued upper binomial tail equals the real-valued Beta measure
of `(-∞,p]`.
-/
theorem binomial_real_Ici_eq_betaMeasure_real_Iic
    (m k : ℕ)
    (p : unitInterval)
    (hk :
      0 < k)
    (hkm :
      k ≤ m) :
    (
      ProbabilityTheory.binomial
        m
        p
    ).real
      (Set.Ici k)
      =
    (
      ProbabilityTheory.betaMeasure
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
    ).real
      (Set.Iic (p : ℝ)) := by

  have hkReal :
      0 < (k : ℝ) := by
    exact_mod_cast hk

  have hSecond :
      0 <
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        ) := by
    positivity

  have hp0 :
      0 ≤ (p : ℝ) :=
    p.property.1

  calc
    (
      ProbabilityTheory.binomial
        m
        p
    ).real
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
          x :=
      binomial_real_Ici_eq_integral_betaPDFReal
        m
        k
        p
        hk
        hkm

    _ =
      (
        ProbabilityTheory.betaMeasure
          (k : ℝ)
          (
            ((m - k : ℕ) : ℝ)
              +
            1
          )
      ).real
        (Set.Iic (p : ℝ)) := by

          symm

          exact
            betaMeasure_real_Iic_eq_integral_betaPDFReal
              (k : ℝ)
              (
                ((m - k : ℕ) : ℝ)
                  +
                1
              )
              (p : ℝ)
              hkReal
              hSecond
              hp0

/--
Exact upper binomial/Beta tail identity:

    Bin(m,p) [k,∞)
      =
    Beta(k,m-k+1) (-∞,p].

This is the upper-tail identity required by
`BinomialBetaTailBridge`.
-/
theorem binomial_Ici_eq_betaMeasure_Iic
    (m k : ℕ)
    (p : unitInterval)
    (hk :
      0 < k)
    (hkm :
      k ≤ m) :
    ProbabilityTheory.binomial
        m
        p
        (Set.Ici k)
      =
    ProbabilityTheory.betaMeasure
        (k : ℝ)
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        )
        (Set.Iic (p : ℝ)) := by

  have hkReal :
      0 < (k : ℝ) := by
    exact_mod_cast hk

  have hSecond :
      0 <
        (
          ((m - k : ℕ) : ℝ)
            +
          1
        ) := by
    positivity

  letI :
      MeasureTheory.IsProbabilityMeasure
        (
          ProbabilityTheory.betaMeasure
            (k : ℝ)
            (
              ((m - k : ℕ) : ℝ)
                +
              1
            )
        ) :=
    ProbabilityTheory.isProbabilityMeasureBeta
      hkReal
      hSecond

  apply
    (
      MeasureTheory.measureReal_eq_measureReal_iff
    ).mp

  exact
    binomial_real_Ici_eq_betaMeasure_real_Iic
      m
      k
      p
      hk
      hkm

end DROSafety.DRO
