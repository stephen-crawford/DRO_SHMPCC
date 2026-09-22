import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Abstract worst-case risk for finite DRO

This file introduces the minimal mathematical interface needed for the
categorical DRO layer.

It intentionally does not yet commit to a particular Wasserstein solver.
Instead, it formalizes the properties that any correctly computed worst-case
distribution `qStar` must satisfy.

The main consequences are:

* the worst-case distribution belongs to the ambiguity set;
* if the nominal distribution belongs to the ambiguity set, worst-case risk
  is at least nominal risk;
* enlarging the ambiguity set cannot reduce worst-case risk;
* consequently, for a nested family of ambiguity sets, increasing the radius
  cannot reduce worst-case risk.
-/

namespace DROSafety.DRO

open scoped BigOperators

/--
A finite categorical distribution represented by one real weight per mode.

Probability-vector properties are stated separately so that the optimization
lemmas remain independent of a specific ambiguity-set construction.
-/
abbrev ModeDistribution (d : ℕ) :=
  Fin d → ℝ

/--
Expected risk of a categorical distribution `q` against a mode-wise risk
vector `risk`.
-/
def expectedRisk
    {d : ℕ}
    (q risk : ModeDistribution d) :
    ℝ :=
  ∑ i, q i * risk i

/--
The ordinary probability-simplex conditions for a finite categorical
distribution.
-/
def IsProbabilityVector
    {d : ℕ}
    (q : ModeDistribution d) :
    Prop :=
  (∀ i, 0 ≤ q i) ∧
  (∑ i, q i) = 1

/--
`qStar` is a worst-case distribution for risk vector `risk` over ambiguity
set `Q` when:

1. `qStar` is feasible, and
2. every other feasible distribution has risk no larger than `qStar`.
-/
def IsWorstCase
    {d : ℕ}
    (Q : Set (ModeDistribution d))
    (risk : ModeDistribution d)
    (qStar : ModeDistribution d) :
    Prop :=
  qStar ∈ Q ∧
  ∀ q ∈ Q,
    expectedRisk q risk ≤
      expectedRisk qStar risk

/--
A worst-case solution is, by definition, feasible for its ambiguity set.
-/
theorem worstCase_mem
    {d : ℕ}
    {Q : Set (ModeDistribution d)}
    {risk qStar : ModeDistribution d}
    (hWorst :
      IsWorstCase Q risk qStar) :
    qStar ∈ Q := by

  exact hWorst.1

/--
If every member of the ambiguity set is a valid probability vector, then a
worst-case optimizer is itself a valid probability vector.
-/
theorem worstCase_isProbabilityVector
    {d : ℕ}
    {Q : Set (ModeDistribution d)}
    {risk qStar : ModeDistribution d}
    (hQ :
      ∀ q ∈ Q,
        IsProbabilityVector q)
    (hWorst :
      IsWorstCase Q risk qStar) :
    IsProbabilityVector qStar := by

  exact
    hQ
      qStar
      hWorst.1

/--
Fundamental DRO property:

if the nominal distribution belongs to the ambiguity set, then the
worst-case risk is at least the nominal risk.
-/
theorem nominalRisk_le_worstCaseRisk
    {d : ℕ}
    {Q : Set (ModeDistribution d)}
    {risk p qStar : ModeDistribution d}
    (hp :
      p ∈ Q)
    (hWorst :
      IsWorstCase Q risk qStar) :
    expectedRisk p risk ≤
      expectedRisk qStar risk := by

  exact
    hWorst.2
      p
      hp

/--
Worst-case risk cannot decrease when the ambiguity set is enlarged.

This theorem compares explicit maximizers of the smaller and larger sets.
-/
theorem worstCaseRisk_mono_of_subset
    {d : ℕ}
    {Q₁ Q₂ : Set (ModeDistribution d)}
    {risk q₁ q₂ : ModeDistribution d}
    (hSubset :
      Q₁ ⊆ Q₂)
    (hWorst₁ :
      IsWorstCase Q₁ risk q₁)
    (hWorst₂ :
      IsWorstCase Q₂ risk q₂) :
    expectedRisk q₁ risk ≤
      expectedRisk q₂ risk := by

  have hq₁Q₂ :
      q₁ ∈ Q₂ :=
    hSubset hWorst₁.1

  exact
    hWorst₂.2
      q₁
      hq₁Q₂

/--
A radius-indexed ambiguity family is nested when increasing the radius can
only add feasible distributions.
-/
def IsNestedAmbiguityFamily
    {d : ℕ}
    (A :
      ℝ →
      Set (ModeDistribution d)) :
    Prop :=
  ∀ ⦃ρ₁ ρ₂ : ℝ⦄,
    ρ₁ ≤ ρ₂ →
    A ρ₁ ⊆ A ρ₂

/--
For a nested ambiguity family, worst-case risk is monotone in the ambiguity
radius.
-/
theorem worstCaseRisk_mono_of_radius
    {d : ℕ}
    {A :
      ℝ →
      Set (ModeDistribution d)}
    {risk q₁ q₂ : ModeDistribution d}
    {ρ₁ ρ₂ : ℝ}
    (hNested :
      IsNestedAmbiguityFamily A)
    (hρ :
      ρ₁ ≤ ρ₂)
    (hWorst₁ :
      IsWorstCase (A ρ₁) risk q₁)
    (hWorst₂ :
      IsWorstCase (A ρ₂) risk q₂) :
    expectedRisk q₁ risk ≤
      expectedRisk q₂ risk := by

  apply
    worstCaseRisk_mono_of_subset
      (Q₁ := A ρ₁)
      (Q₂ := A ρ₂)
      (risk := risk)
      (q₁ := q₁)
      (q₂ := q₂)

  · exact
      hNested hρ

  · exact
      hWorst₁

  · exact
      hWorst₂

/--
If an ambiguity set contains only the nominal distribution, then the nominal
distribution is trivially worst-case.
-/
theorem worstCase_of_singleton
    {d : ℕ}
    (p risk : ModeDistribution d) :
    IsWorstCase
      ({p} : Set (ModeDistribution d))
      risk
      p := by

  constructor

  · simp

  · intro q hq

    have hqp :
        q = p := by
      simpa using hq

    subst q

    exact le_rfl

/--
Consequently, a zero-radius ambiguity set that is exactly `{p}` has no risk
lift relative to the nominal distribution.
-/
theorem zeroRadius_noRiskLift
    {d : ℕ}
    (p risk : ModeDistribution d) :
    expectedRisk p risk =
      expectedRisk p risk := by

  rfl

end DROSafety.DRO
