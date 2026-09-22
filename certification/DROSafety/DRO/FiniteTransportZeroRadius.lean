import DROSafety.DRO.FiniteTransport
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Zero-radius finite Wasserstein ambiguity

This file proves the zero-radius property for the finite categorical
transport formulation.

If every off-diagonal ground cost is strictly positive, then a transport
plan of total cost zero cannot move any mass between distinct modes.

Consequently,

    FiniteWassersteinAmbiguity D p 0 = {p},

and any worst-case distribution at radius zero is exactly the nominal
distribution.
-/

namespace DROSafety.DRO

open scoped BigOperators

/--
The ground cost separates distinct modes: transporting mass between
different modes has strictly positive cost.
-/
def PositiveOffDiagonal
    {d : ℕ}
    (D : FiniteGroundCost d) :
    Prop :=
  ∀ i j,
    i ≠ j →
    0 < D.cost i j

/--
If a feasible transport plan has total cost zero and the ground cost is
strictly positive off the diagonal, then every off-diagonal transported
mass is zero.
-/
theorem transportPlan_offdiag_mass_zero_of_cost_zero
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p q : ModeDistribution d}
    (π : TransportPlan p q)
    (hPositive :
      PositiveOffDiagonal D)
    (hCost :
      transportCost D π = 0)
    (i j : Fin d)
    (hij :
      i ≠ j) :
    π.mass i j = 0 := by

  classical

  have hTermNonneg :
      0 ≤ D.cost i j * π.mass i j := by
    exact
      mul_nonneg
        (D.nonneg i j)
        (π.nonneg i j)

  have hTermLeInner :
      D.cost i j * π.mass i j
        ≤
      ∑ j', D.cost i j' * π.mass i j' := by

    exact
      Finset.single_le_sum
        (fun j' _ =>
          mul_nonneg
            (D.nonneg i j')
            (π.nonneg i j'))
        (Finset.mem_univ j)

  have hInnerLeTotal :
      (∑ j', D.cost i j' * π.mass i j')
        ≤
      transportCost D π := by

    unfold transportCost

    exact
      Finset.single_le_sum
        (fun i' _ =>
          Finset.sum_nonneg
            (fun j' _ =>
              mul_nonneg
                (D.nonneg i' j')
                (π.nonneg i' j')))
        (Finset.mem_univ i)

  have hTermLeZero :
      D.cost i j * π.mass i j ≤ 0 := by

    calc
      D.cost i j * π.mass i j
          ≤
        ∑ j', D.cost i j' * π.mass i j' :=
        hTermLeInner

      _ ≤ transportCost D π :=
        hInnerLeTotal

      _ = 0 :=
        hCost

  have hProductZero :
      D.cost i j * π.mass i j = 0 := by

    exact
      le_antisymm
        hTermLeZero
        hTermNonneg

  rcases mul_eq_zero.mp hProductZero with
    hGroundZero | hMassZero

  · have hGroundPositive :
        0 < D.cost i j :=
      hPositive i j hij

    exact
      False.elim
        ((ne_of_gt hGroundPositive)
          hGroundZero)

  · exact hMassZero

/--
A zero-cost transport plan with strictly positive off-diagonal ground
cost can only couple a distribution to itself.
-/
theorem distributions_eq_of_zero_transportCost
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p q : ModeDistribution d}
    (π : TransportPlan p q)
    (hPositive :
      PositiveOffDiagonal D)
    (hCost :
      transportCost D π = 0) :
    p = q := by

  classical

  funext j

  have hRow :
      p j = π.mass j j := by

    calc
      p j =
          ∑ k, π.mass j k := by
            symm
            exact π.row_marginal j

      _ = π.mass j j := by

        apply Finset.sum_eq_single j

        · intro k hk hkj

          exact
            transportPlan_offdiag_mass_zero_of_cost_zero
              D
              π
              hPositive
              hCost
              j
              k
              (Ne.symm hkj)

        · simp

  have hCol :
      q j = π.mass j j := by

    calc
      q j =
          ∑ i, π.mass i j := by
            symm
            exact π.col_marginal j

      _ = π.mass j j := by

        apply Finset.sum_eq_single j

        · intro i hi hij

          exact
            transportPlan_offdiag_mass_zero_of_cost_zero
              D
              π
              hPositive
              hCost
              i
              j
              hij

        · simp

  exact
    hRow.trans
      hCol.symm

/--
With strictly positive off-diagonal ground cost, the radius-zero finite
Wasserstein ambiguity set is exactly the nominal singleton.
-/
theorem finiteWassersteinAmbiguity_zero
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d)
    (hp :
      IsProbabilityVector p)
    (hPositive :
      PositiveOffDiagonal D) :
    FiniteWassersteinAmbiguity D p 0 =
      ({p} : Set (ModeDistribution d)) := by

  apply Set.Subset.antisymm

  · intro q hq

    rcases hq with
      ⟨hqProbability, π, hCostLe⟩

    have hCostNonneg :
        0 ≤ transportCost D π :=
      transportCost_nonneg
        D
        π

    have hCostZero :
        transportCost D π = 0 := by

      exact
        le_antisymm
          hCostLe
          hCostNonneg

    have hpq :
        p = q :=
      distributions_eq_of_zero_transportCost
        D
        π
        hPositive
        hCostZero

    have hqp :
        q = p :=
      hpq.symm

    simpa [hqp]

  · intro q hq

    have hqp :
        q = p := by
      simpa using hq

    subst q

    exact
      nominal_mem_finiteWassersteinAmbiguity
        D
        hp
        (show (0 : ℝ) ≤ 0 from le_rfl)

/--
Any worst-case optimizer at zero finite-Wasserstein radius must be the
nominal distribution.
-/
theorem finiteWasserstein_zeroRadius_worstCase_eq_nominal
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d)
    {risk qStar : ModeDistribution d}
    (hp :
      IsProbabilityVector p)
    (hPositive :
      PositiveOffDiagonal D)
    (hWorst :
      IsWorstCase
        (FiniteWassersteinAmbiguity D p 0)
        risk
        qStar) :
    qStar = p := by

  have hMem :
      qStar ∈
        FiniteWassersteinAmbiguity
          D
          p
          0 :=
    hWorst.1

  rw [
    finiteWassersteinAmbiguity_zero
      D
      p
      hp
      hPositive
  ] at hMem

  simpa using hMem

/--
At zero finite-Wasserstein radius, DRO worst-case risk is exactly nominal
risk.
-/
theorem finiteWasserstein_zeroRadius_risk_eq_nominal
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d)
    {risk qStar : ModeDistribution d}
    (hp :
      IsProbabilityVector p)
    (hPositive :
      PositiveOffDiagonal D)
    (hWorst :
      IsWorstCase
        (FiniteWassersteinAmbiguity D p 0)
        risk
        qStar) :
    expectedRisk qStar risk =
      expectedRisk p risk := by

  have hEq :
      qStar = p :=
    finiteWasserstein_zeroRadius_worstCase_eq_nominal
      D
      p
      hp
      hPositive
      hWorst

  rw [hEq]

/--
Consequently, the DRO risk lift is exactly zero at radius zero.
-/
theorem finiteWasserstein_zeroRadius_riskLift_eq_zero
    {d : ℕ}
    (D : FiniteGroundCost d)
    (p : ModeDistribution d)
    {risk qStar : ModeDistribution d}
    (hp :
      IsProbabilityVector p)
    (hPositive :
      PositiveOffDiagonal D)
    (hWorst :
      IsWorstCase
        (FiniteWassersteinAmbiguity D p 0)
        risk
        qStar) :
    expectedRisk qStar risk -
        expectedRisk p risk
      =
    0 := by

  rw [
    finiteWasserstein_zeroRadius_risk_eq_nominal
      D
      p
      hp
      hPositive
      hWorst
  ]

  exact sub_self _

end DROSafety.DRO
