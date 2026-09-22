import DROSafety.DRO.WorstCaseRisk
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Wasserstein ambiguity sets

This file specializes the abstract DRO results in `WorstCaseRisk`
to radius-indexed Wasserstein ambiguity sets.

For now, the Wasserstein transport cost is represented abstractly by
`WassersteinMetric`.  A later file will instantiate this interface using
the finite categorical optimal-transport problem actually implemented by
the controller.

The required properties are:

* transport cost is nonnegative;
* a distribution has zero transport cost to itself;
* zero transport cost identifies equal distributions.

From those properties we prove:

* the nominal distribution belongs to every nonnegative-radius ball;
* ambiguity balls are nested in the radius;
* the zero-radius ball is exactly the nominal singleton;
* worst-case risk dominates nominal risk;
* worst-case risk is monotone in the radius;
* zero radius produces no DRO risk lift.
-/

namespace DROSafety.DRO

/--
Abstract Wasserstein transport-cost interface.

The later finite-transport development will instantiate `cost` with the
optimal transport value induced by the categorical ground-cost matrix.
-/
structure WassersteinMetric (d : ℕ) where
  cost :
    ModeDistribution d →
    ModeDistribution d →
    ℝ

  nonneg :
    ∀ p q,
      0 ≤ cost p q

  self_zero :
    ∀ p,
      cost p p = 0

  eq_of_zero :
    ∀ {p q},
      cost p q = 0 →
      p = q

/--
The Wasserstein ambiguity ball of radius `ρ` around nominal distribution `p`.

Membership requires both:

1. `q` is a probability vector;
2. its Wasserstein transport cost from `p` is at most `ρ`.
-/
def WassersteinAmbiguity
    {d : ℕ}
    (W : WassersteinMetric d)
    (p : ModeDistribution d)
    (ρ : ℝ) :
    Set (ModeDistribution d) :=
  {
    q |
      IsProbabilityVector q ∧
      W.cost p q ≤ ρ
  }

/--
The nominal distribution belongs to every nonnegative-radius Wasserstein
ambiguity ball, provided it is itself a probability vector.
-/
theorem nominal_mem_wassersteinAmbiguity
    {d : ℕ}
    (W : WassersteinMetric d)
    {p : ModeDistribution d}
    (hp :
      IsProbabilityVector p)
    {ρ : ℝ}
    (hρ :
      0 ≤ ρ) :
    p ∈ WassersteinAmbiguity W p ρ := by

  constructor

  · exact hp

  · rw [W.self_zero p]

    exact hρ

/--
Increasing the ambiguity radius can only enlarge the Wasserstein ball.
-/
theorem wassersteinAmbiguity_mono
    {d : ℕ}
    (W : WassersteinMetric d)
    (p : ModeDistribution d)
    {ρ₁ ρ₂ : ℝ}
    (hρ :
      ρ₁ ≤ ρ₂) :
    WassersteinAmbiguity W p ρ₁ ⊆
      WassersteinAmbiguity W p ρ₂ := by

  intro q hq

  constructor

  · exact hq.1

  · exact
      le_trans
        hq.2
        hρ

/--
The radius-indexed Wasserstein ambiguity family is nested.
-/
theorem wassersteinAmbiguity_isNested
    {d : ℕ}
    (W : WassersteinMetric d)
    (p : ModeDistribution d) :
    IsNestedAmbiguityFamily
      (WassersteinAmbiguity W p) := by

  intro ρ₁ ρ₂ hρ

  exact
    wassersteinAmbiguity_mono
      W
      p
      hρ

/--
At radius zero, the Wasserstein ambiguity ball consists only of the nominal
distribution.
-/
theorem wassersteinAmbiguity_zero
    {d : ℕ}
    (W : WassersteinMetric d)
    (p : ModeDistribution d)
    (hp :
      IsProbabilityVector p) :
    WassersteinAmbiguity W p 0 =
      ({p} : Set (ModeDistribution d)) := by

  apply Set.Subset.antisymm

  · intro q hq

    have hCostNonneg :
        0 ≤ W.cost p q :=
      W.nonneg p q

    have hCostZero :
        W.cost p q = 0 :=
      le_antisymm
        hq.2
        hCostNonneg

    have hpq :
        p = q :=
      W.eq_of_zero
        hCostZero

    simpa [hpq]

  · intro q hq

    have hqp :
        q = p := by
      simpa using hq

    subst q

    exact
      nominal_mem_wassersteinAmbiguity
        W
        hp
        (show (0 : ℝ) ≤ 0 from le_rfl)

/--
A worst-case distribution over a Wasserstein ambiguity set is necessarily
a valid probability vector.
-/
theorem wassersteinWorstCase_isProbabilityVector
    {d : ℕ}
    {W : WassersteinMetric d}
    {p risk qStar : ModeDistribution d}
    {ρ : ℝ}
    (hWorst :
      IsWorstCase
        (WassersteinAmbiguity W p ρ)
        risk
        qStar) :
    IsProbabilityVector qStar := by

  exact
    hWorst.1.1

/--
Fundamental Wasserstein-DRO risk inequality.

When `ρ ≥ 0`, the nominal distribution belongs to its ambiguity set.
Therefore any worst-case optimizer has expected risk at least as large
as nominal expected risk.
-/
theorem nominalRisk_le_wassersteinWorstCaseRisk
    {d : ℕ}
    (W : WassersteinMetric d)
    {p risk qStar : ModeDistribution d}
    {ρ : ℝ}
    (hp :
      IsProbabilityVector p)
    (hρ :
      0 ≤ ρ)
    (hWorst :
      IsWorstCase
        (WassersteinAmbiguity W p ρ)
        risk
        qStar) :
    expectedRisk p risk ≤
      expectedRisk qStar risk := by

  apply
    nominalRisk_le_worstCaseRisk
      (Q :=
        WassersteinAmbiguity
          W
          p
          ρ)
      (p := p)
      (qStar := qStar)

  · exact
      nominal_mem_wassersteinAmbiguity
        W
        hp
        hρ

  · exact hWorst

/--
The DRO risk lift relative to nominal risk is nonnegative.
-/
theorem wassersteinRiskLift_nonneg
    {d : ℕ}
    (W : WassersteinMetric d)
    {p risk qStar : ModeDistribution d}
    {ρ : ℝ}
    (hp :
      IsProbabilityVector p)
    (hρ :
      0 ≤ ρ)
    (hWorst :
      IsWorstCase
        (WassersteinAmbiguity W p ρ)
        risk
        qStar) :
    0 ≤
      expectedRisk qStar risk -
        expectedRisk p risk := by

  exact
    sub_nonneg.mpr
      (nominalRisk_le_wassersteinWorstCaseRisk
        W
        hp
        hρ
        hWorst)

/--
Worst-case Wasserstein risk cannot decrease when the ambiguity radius
increases.
-/
theorem wassersteinWorstCaseRisk_mono_radius
    {d : ℕ}
    (W : WassersteinMetric d)
    (p : ModeDistribution d)
    {risk q₁ q₂ : ModeDistribution d}
    {ρ₁ ρ₂ : ℝ}
    (hρ :
      ρ₁ ≤ ρ₂)
    (hWorst₁ :
      IsWorstCase
        (WassersteinAmbiguity W p ρ₁)
        risk
        q₁)
    (hWorst₂ :
      IsWorstCase
        (WassersteinAmbiguity W p ρ₂)
        risk
        q₂) :
    expectedRisk q₁ risk ≤
      expectedRisk q₂ risk := by

  apply
    worstCaseRisk_mono_of_subset
      (Q₁ :=
        WassersteinAmbiguity W p ρ₁)
      (Q₂ :=
        WassersteinAmbiguity W p ρ₂)
      (risk := risk)
      (q₁ := q₁)
      (q₂ := q₂)

  · exact
      wassersteinAmbiguity_mono
        W
        p
        hρ

  · exact hWorst₁

  · exact hWorst₂

/--
Any worst-case optimizer of the zero-radius ambiguity set must equal the
nominal distribution.
-/
theorem zeroRadius_worstCase_eq_nominal
    {d : ℕ}
    (W : WassersteinMetric d)
    (p : ModeDistribution d)
    {risk qStar : ModeDistribution d}
    (hp :
      IsProbabilityVector p)
    (hWorst :
      IsWorstCase
        (WassersteinAmbiguity W p 0)
        risk
        qStar) :
    qStar = p := by

  have hMem :
      qStar ∈
        WassersteinAmbiguity W p 0 :=
    hWorst.1

  rw [
    wassersteinAmbiguity_zero
      W
      p
      hp
  ] at hMem

  simpa using hMem

/--
At zero ambiguity radius, worst-case risk equals nominal risk exactly.
-/
theorem zeroRadius_worstCaseRisk_eq_nominal
    {d : ℕ}
    (W : WassersteinMetric d)
    (p : ModeDistribution d)
    {risk qStar : ModeDistribution d}
    (hp :
      IsProbabilityVector p)
    (hWorst :
      IsWorstCase
        (WassersteinAmbiguity W p 0)
        risk
        qStar) :
    expectedRisk qStar risk =
      expectedRisk p risk := by

  have hEq :
      qStar = p :=
    zeroRadius_worstCase_eq_nominal
      W
      p
      hp
      hWorst

  rw [hEq]

end DROSafety.DRO
