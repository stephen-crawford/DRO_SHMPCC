import DROSafety.DRO.ClopperPearsonBetaBridge
import Mathlib.Probability.Distributions.Beta
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Beta-measure tail probabilities as density integrals

`ProbabilityTheory.betaMeasure α β` is defined as Lebesgue measure with
density `ProbabilityTheory.betaPDF α β`.

This file records the exact lower-tail and upper-tail integral formulas
needed for the binomial/Beta identity.

The next step will specialize the shape parameters to positive natural
numbers and turn these density integrals into the finite binomial-tail
polynomials.
-/

namespace DROSafety.DRO

open MeasureTheory
open ProbabilityTheory

/--
The Beta lower-tail probability is the integral of the Beta density over
`(-∞, x]`.
-/
theorem betaMeasure_Iic_eq_lintegral
    (a b x : ℝ) :
    ProbabilityTheory.betaMeasure
        a
        b
        (Set.Iic x)
      =
    ∫⁻ y in Set.Iic x,
      ProbabilityTheory.betaPDF a b y := by

  simpa [ProbabilityTheory.betaMeasure] using
    (MeasureTheory.withDensity_apply
      (μ := (MeasureTheory.volume : Measure ℝ))
      (ProbabilityTheory.betaPDF a b)
      (s := Set.Iic x)
      measurableSet_Iic)

/--
The Beta upper-tail probability is the integral of the Beta density over
`[x, ∞)`.
-/
theorem betaMeasure_Ici_eq_lintegral
    (a b x : ℝ) :
    ProbabilityTheory.betaMeasure
        a
        b
        (Set.Ici x)
      =
    ∫⁻ y in Set.Ici x,
      ProbabilityTheory.betaPDF a b y := by

  simpa [ProbabilityTheory.betaMeasure] using
    (MeasureTheory.withDensity_apply
      (μ := (MeasureTheory.volume : Measure ℝ))
      (ProbabilityTheory.betaPDF a b)
      (s := Set.Ici x)
      measurableSet_Ici)

/--
The Beta probability of an open interval is the corresponding density
integral.
-/
theorem betaMeasure_Ioo_eq_lintegral
    (a b x y : ℝ) :
    ProbabilityTheory.betaMeasure
        a
        b
        (Set.Ioo x y)
      =
    ∫⁻ z in Set.Ioo x y,
      ProbabilityTheory.betaPDF a b z := by

  simpa [ProbabilityTheory.betaMeasure] using
    (MeasureTheory.withDensity_apply
      (μ := (MeasureTheory.volume : Measure ℝ))
      (ProbabilityTheory.betaPDF a b)
      (s := Set.Ioo x y)
      measurableSet_Ioo)

/--
Lower Beta tails are monotone in their cutoff.
-/
theorem betaMeasure_Iic_mono
    (a b : ℝ)
    {x y : ℝ}
    (hxy :
      x ≤ y) :
    ProbabilityTheory.betaMeasure
        a
        b
        (Set.Iic x)
      ≤
    ProbabilityTheory.betaMeasure
        a
        b
        (Set.Iic y) := by

  apply MeasureTheory.measure_mono

  exact
    Set.Iic_subset_Iic.mpr
      hxy

/--
Upper Beta tails are antitone in their cutoff.
-/
theorem betaMeasure_Ici_antitone
    (a b : ℝ)
    {x y : ℝ}
    (hxy :
      x ≤ y) :
    ProbabilityTheory.betaMeasure
        a
        b
        (Set.Ici y)
      ≤
    ProbabilityTheory.betaMeasure
        a
        b
        (Set.Ici x) := by

  apply MeasureTheory.measure_mono

  exact
    Set.Ici_subset_Ici.mpr
      hxy

/--
The lower-tail identity from `BinomialBetaTailBridge` can equivalently be
viewed as an equality with a Beta-density integral.
-/
theorem upperBinomialTail_eq_beta_lintegral_of_bridge
    (m : ℕ)
    (p : unitInterval)
    (hBridge :
      BinomialBetaTailBridge
        m
        p)
    (k : ℕ)
    (hkPos :
      0 < k)
    (hkm :
      k ≤ m) :
    ProbabilityTheory.binomial m p
        (Set.Ici k)
      =
    ∫⁻ x in Set.Iic (p : ℝ),
      ProbabilityTheory.betaPDF
        (k : ℝ)
        (m - k + 1 : ℝ)
        x := by

  rw [
    hBridge.upperBinomialTail
      k
      hkPos
      hkm
  ]

  exact
    betaMeasure_Iic_eq_lintegral
      (k : ℝ)
      (m - k + 1 : ℝ)
      (p : ℝ)

/--
Likewise, the lower binomial tail corresponds to the Beta-density upper
tail integral.
-/
theorem lowerBinomialTail_eq_beta_lintegral_of_bridge
    (m : ℕ)
    (p : unitInterval)
    (hBridge :
      BinomialBetaTailBridge
        m
        p)
    (k : ℕ)
    (hkm :
      k < m) :
    ProbabilityTheory.binomial m p
        (Set.Iic k)
      =
    ∫⁻ x in Set.Ici (p : ℝ),
      ProbabilityTheory.betaPDF
        (k + 1 : ℝ)
        (m - k : ℝ)
        x := by

  rw [
    hBridge.lowerBinomialTail
      k
      hkm
  ]

  exact
    betaMeasure_Ici_eq_lintegral
      (k + 1 : ℝ)
      (m - k : ℝ)
      (p : ℝ)

end DROSafety.DRO
