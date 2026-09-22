import DROSafety.DRO.WassersteinAmbiguity
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Finite categorical optimal transport

This file introduces the finite transport problem underlying the categorical
Wasserstein DRO implementation.

A transport plan `π i j` moves probability mass from nominal mode `i`
to candidate mode `j`.

The plan must satisfy

    ∑ j, π i j = p i

and

    ∑ i, π i j = q j.

The transport cost is

    ∑ i, ∑ j, D i j * π i j.

For ambiguity-set membership we do not yet define the minimum transport cost.
Instead, `q` belongs to the radius-`ρ` ambiguity set when there exists a
feasible coupling whose cost is at most `ρ`.

For finite optimal transport this is exactly the feasibility statement needed
for the Wasserstein ball and avoids introducing minimizer existence before it
is necessary.
-/

namespace DROSafety.DRO

open scoped BigOperators

/--
Finite ground-cost matrix.

For now we require only nonnegativity and zero diagonal.
Strictly positive off-diagonal costs will be added when proving that the
zero-radius ambiguity set is exactly the nominal singleton.
-/
structure FiniteGroundCost (d : ℕ) where
  cost :
    Fin d →
    Fin d →
    ℝ

  nonneg :
    ∀ i j,
      0 ≤ cost i j

  diagonal_zero :
    ∀ i,
      cost i i = 0

/--
A feasible transport coupling between distributions `p` and `q`.

`mass i j` is the amount of probability mass transported from nominal
mode `i` to candidate mode `j`.
-/
structure TransportPlan
    {d : ℕ}
    (p q : ModeDistribution d) where

  mass :
    Fin d →
    Fin d →
    ℝ

  nonneg :
    ∀ i j,
      0 ≤ mass i j

  row_marginal :
    ∀ i,
      (∑ j, mass i j) = p i

  col_marginal :
    ∀ j,
      (∑ i, mass i j) = q j

/--
Cost of a finite transport plan.
-/
def transportCost
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p q : ModeDistribution d}
    (π : TransportPlan p q) :
    ℝ :=
  ∑ i, ∑ j,
    D.cost i j * π.mass i j

/--
Every feasible transport plan has nonnegative cost.
-/
theorem transportCost_nonneg
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p q : ModeDistribution d}
    (π : TransportPlan p q) :
    0 ≤ transportCost D π := by

  apply Finset.sum_nonneg
  intro i hi

  apply Finset.sum_nonneg
  intro j hj

  exact
    mul_nonneg
      (D.nonneg i j)
      (π.nonneg i j)

/--
Diagonal coupling: leave every unit of probability mass in its original mode.
-/
def diagonalMass
    {d : ℕ}
    (p : ModeDistribution d)
    (i j : Fin d) :
    ℝ :=
  if i = j then
    p i
  else
    0

/--
Every probability vector admits a diagonal self-coupling.
-/
def diagonalTransportPlan
    {d : ℕ}
    (p : ModeDistribution d)
    (hp : IsProbabilityVector p) :
    TransportPlan p p where

  mass :=
    diagonalMass p

  nonneg := by
    intro i j

    by_cases h : i = j

    · subst j
      simp [
        diagonalMass,
        hp.1 i
      ]

    · simp [
        diagonalMass,
        h
      ]

  row_marginal := by
    intro i
    classical

    simp [diagonalMass]

  col_marginal := by
    intro j
    classical

    simp [diagonalMass]

/--
The diagonal self-coupling has zero transport cost.
-/
theorem transportCost_diagonal
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d)
    (hp : IsProbabilityVector p) :
    transportCost
        D
        (diagonalTransportPlan p hp)
      =
    0 := by

  classical

  simp [
    transportCost,
    diagonalTransportPlan,
    diagonalMass,
    D.diagonal_zero
  ]

/--
Every transport plan has the same total row and column mass.

This is a useful consistency property of the coupling constraints.
-/
theorem transportPlan_total_mass_eq
    {d : ℕ}
    {p q : ModeDistribution d}
    (π : TransportPlan p q) :
    (∑ i, p i) =
      ∑ j, q j := by

  calc
    (∑ i, p i)
        =
      ∑ i, ∑ j, π.mass i j := by
        apply Finset.sum_congr rfl
        intro i hi
        symm
        exact π.row_marginal i

    _ =
      ∑ j, ∑ i, π.mass i j := by
        rw [Finset.sum_comm]

    _ =
      ∑ j, q j := by
        apply Finset.sum_congr rfl
        intro j hj
        exact π.col_marginal j

/--
Finite categorical Wasserstein ambiguity set.

A candidate distribution `q` belongs to the radius-`ρ` ball around `p` when:

1. `q` is a probability vector; and
2. there exists a transport plan from `p` to `q` with cost at most `ρ`.
-/
def FiniteWassersteinAmbiguity
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d)
    (ρ : ℝ) :
    Set (ModeDistribution d) :=
  {
    q |
      IsProbabilityVector q ∧
      ∃ π : TransportPlan p q,
        transportCost D π ≤ ρ
  }

/--
The nominal distribution belongs to every nonnegative-radius finite
Wasserstein ambiguity set.
-/
theorem nominal_mem_finiteWassersteinAmbiguity
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p : ModeDistribution d}
    (hp : IsProbabilityVector p)
    {ρ : ℝ}
    (hρ : 0 ≤ ρ) :
    p ∈ FiniteWassersteinAmbiguity D p ρ := by

  constructor

  · exact hp

  · refine
      ⟨diagonalTransportPlan p hp, ?_⟩

    rw [transportCost_diagonal]

    exact hρ

/--
Increasing the radius enlarges the finite Wasserstein ambiguity set.
-/
theorem finiteWassersteinAmbiguity_mono
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d)
    {ρ₁ ρ₂ : ℝ}
    (hρ : ρ₁ ≤ ρ₂) :
    FiniteWassersteinAmbiguity D p ρ₁
      ⊆
    FiniteWassersteinAmbiguity D p ρ₂ := by

  intro q hq

  rcases hq with
    ⟨hqProb, π, hCost⟩

  refine
    ⟨hqProb, π, ?_⟩

  exact
    le_trans
      hCost
      hρ

/--
The finite transport ambiguity family is nested in the radius.
-/
theorem finiteWassersteinAmbiguity_isNested
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d) :
    IsNestedAmbiguityFamily
      (FiniteWassersteinAmbiguity D p) := by

  intro ρ₁ ρ₂ hρ

  exact
    finiteWassersteinAmbiguity_mono
      D
      p
      hρ

/--
The finite categorical DRO worst-case risk is at least nominal risk whenever
the ambiguity radius is nonnegative.
-/
theorem nominalRisk_le_finiteWassersteinWorstCaseRisk
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p risk qStar : ModeDistribution d}
    {ρ : ℝ}
    (hp : IsProbabilityVector p)
    (hρ : 0 ≤ ρ)
    (hWorst :
      IsWorstCase
        (FiniteWassersteinAmbiguity D p ρ)
        risk
        qStar) :
    expectedRisk p risk ≤
      expectedRisk qStar risk := by

  apply
    nominalRisk_le_worstCaseRisk
      (Q :=
        FiniteWassersteinAmbiguity
          D
          p
          ρ)

  · exact
      nominal_mem_finiteWassersteinAmbiguity
        D
        hp
        hρ

  · exact hWorst

/--
The finite-Wasserstein DRO risk lift is nonnegative.
-/
theorem finiteWassersteinRiskLift_nonneg
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p risk qStar : ModeDistribution d}
    {ρ : ℝ}
    (hp : IsProbabilityVector p)
    (hρ : 0 ≤ ρ)
    (hWorst :
      IsWorstCase
        (FiniteWassersteinAmbiguity D p ρ)
        risk
        qStar) :
    0 ≤
      expectedRisk qStar risk -
        expectedRisk p risk := by

  exact
    sub_nonneg.mpr
      (nominalRisk_le_finiteWassersteinWorstCaseRisk
        D
        hp
        hρ
        hWorst)

/--
Worst-case finite-Wasserstein risk is monotone in the radius.
-/
theorem finiteWassersteinWorstCaseRisk_mono_radius
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d)
    {risk q₁ q₂ : ModeDistribution d}
    {ρ₁ ρ₂ : ℝ}
    (hρ : ρ₁ ≤ ρ₂)
    (hWorst₁ :
      IsWorstCase
        (FiniteWassersteinAmbiguity D p ρ₁)
        risk
        q₁)
    (hWorst₂ :
      IsWorstCase
        (FiniteWassersteinAmbiguity D p ρ₂)
        risk
        q₂) :
    expectedRisk q₁ risk ≤
      expectedRisk q₂ risk := by

  apply
    worstCaseRisk_mono_of_subset
      (Q₁ :=
        FiniteWassersteinAmbiguity
          D
          p
          ρ₁)
      (Q₂ :=
        FiniteWassersteinAmbiguity
          D
          p
          ρ₂)

  · exact
      finiteWassersteinAmbiguity_mono
        D
        p
        hρ

  · exact hWorst₁

  · exact hWorst₂

end DROSafety.DRO
