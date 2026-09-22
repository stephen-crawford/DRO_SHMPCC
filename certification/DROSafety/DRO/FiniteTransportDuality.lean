import DROSafety.DRO.FiniteTransportZeroRadius
import Mathlib

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Primal-dual certification for finite categorical DRO

This file proves weak duality for the finite categorical transport problem.

For nominal distribution `p`, risk vector `risk`, ground cost `D`, and
ambiguity radius `ρ`, the primal problem maximizes

    expectedRisk q risk

over all distributions `q` admitting a transport plan from `p` with cost
at most `ρ`.

A dual certificate consists of

    λ ≥ 0

and one scalar `α i` for each nominal mode, satisfying

    risk j ≤ α i + λ * D i j

for every pair of modes `i,j`.

Weak duality then gives

    expectedRisk q risk
      ≤ ∑ i, p i * α i + λ * ρ.

If a feasible candidate `qStar` attains this dual bound, it is a genuine
worst-case distribution.
-/

namespace DROSafety.DRO

open scoped BigOperators

/--
Risk represented directly through a transport coupling.

Because the column marginals of `π` equal `q`, this is equal to
`expectedRisk q risk`.
-/
def transportRisk
    {d : ℕ}
    {p q : ModeDistribution d}
    (π : TransportPlan p q)
    (risk : ModeDistribution d) :
    ℝ :=
  ∑ i, ∑ j,
    π.mass i j * risk j

/--
The transport representation of risk equals expected risk under the
candidate distribution.
-/
theorem transportRisk_eq_expectedRisk
    {d : ℕ}
    {p q : ModeDistribution d}
    (π : TransportPlan p q)
    (risk : ModeDistribution d) :
    transportRisk π risk =
      expectedRisk q risk := by

  classical

  calc
    transportRisk π risk
        =
      ∑ j, ∑ i,
        π.mass i j * risk j := by
          unfold transportRisk
          rw [Finset.sum_comm]

    _ =
      ∑ j,
        q j * risk j := by
          apply Finset.sum_congr rfl
          intro j hj

          calc
            (∑ i,
                π.mass i j * risk j)
                =
              (∑ i, π.mass i j) *
                risk j := by
                  rw [Finset.sum_mul]

            _ =
              q j * risk j := by
                rw [π.col_marginal j]

    _ =
      expectedRisk q risk := by
        rfl

/--
Feasibility conditions for a finite-transport dual solution.
-/
def TransportDualFeasible
    {d : ℕ}
    (D : FiniteGroundCost d)
    (risk : ModeDistribution d)
    (α : Fin d → ℝ)
    (lam : ℝ) :
    Prop :=
  0 ≤ lam ∧
  ∀ i j,
    risk j ≤
      α i + lam * D.cost i j

/--
Objective value of a finite-transport dual certificate.
-/
def transportDualObjective
    {d : ℕ}
    (p : ModeDistribution d)
    (ρ : ℝ)
    (α : Fin d → ℝ)
    (lam : ℝ) :
    ℝ :=
  (∑ i, p i * α i) +
    lam * ρ

/--
The part of the transport sum involving the dual variables `α` collapses
through the row marginals to `∑ i, p i * α i`.
-/
theorem transportAlpha_eq
    {d : ℕ}
    {p q : ModeDistribution d}
    (π : TransportPlan p q)
    (α : Fin d → ℝ) :
    (∑ i, ∑ j,
      π.mass i j * α i)
      =
    ∑ i, p i * α i := by

  classical

  apply Finset.sum_congr rfl

  intro i hi

  calc
    (∑ j,
        π.mass i j * α i)
        =
      (∑ j, π.mass i j) *
        α i := by
          rw [Finset.sum_mul]

    _ =
      p i * α i := by
        rw [π.row_marginal i]

/--
The cost-dependent portion of the transport sum factors into
`λ * transportCost`.
-/
theorem transportLambdaCost_eq
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p q : ModeDistribution d}
    (π : TransportPlan p q)
    (lam : ℝ) :
    (∑ i, ∑ j,
      π.mass i j *
        (lam * D.cost i j))
      =
    lam * transportCost D π := by

  classical

  calc
    (∑ i, ∑ j,
        π.mass i j *
          (lam * D.cost i j))
        =
      ∑ i, ∑ j,
        lam *
          (D.cost i j *
            π.mass i j) := by

          apply Finset.sum_congr rfl
          intro i hi

          apply Finset.sum_congr rfl
          intro j hj

          ring

    _ =
      ∑ i,
        lam *
          (∑ j,
            D.cost i j *
              π.mass i j) := by

          apply Finset.sum_congr rfl
          intro i hi

          rw [Finset.mul_sum]

    _ =
      lam *
        (∑ i, ∑ j,
          D.cost i j *
            π.mass i j) := by
          rw [Finset.mul_sum]

    _ =
      lam *
        transportCost D π := by
          rfl

/--
Expansion of the summed dual upper bound.
-/
theorem transportDualExpansion
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p q : ModeDistribution d}
    (π : TransportPlan p q)
    (α : Fin d → ℝ)
    (lam : ℝ) :
    (∑ i, ∑ j,
      π.mass i j *
        (α i + lam * D.cost i j))
      =
    (∑ i, p i * α i) +
      lam * transportCost D π := by

  classical

  calc
    (∑ i, ∑ j,
        π.mass i j *
          (α i + lam * D.cost i j))
        =
      (∑ i, ∑ j,
        π.mass i j * α i) +
      (∑ i, ∑ j,
        π.mass i j *
          (lam * D.cost i j)) := by

          simp_rw [
            mul_add,
            Finset.sum_add_distrib
          ]

    _ =
      (∑ i, p i * α i) +
        lam * transportCost D π := by

          rw [
            transportAlpha_eq,
            transportLambdaCost_eq
          ]

/--
Weak duality for finite categorical Wasserstein DRO.

Every primal-feasible distribution has risk no larger than every
dual-feasible objective value.
-/
theorem finiteTransport_weakDuality
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p q risk : ModeDistribution d}
    {ρ lam : ℝ}
    {α : Fin d → ℝ}
    (hq :
      q ∈
        FiniteWassersteinAmbiguity
          D
          p
          ρ)
    (hDual :
      TransportDualFeasible
        D
        risk
        α
        lam) :
    expectedRisk q risk ≤
      transportDualObjective
        p
        ρ
        α
        lam := by

  rcases hq with
    ⟨hqProbability, π, hTransportBudget⟩

  have hTransportRiskBound :
      transportRisk π risk
        ≤
      ∑ i, ∑ j,
        π.mass i j *
          (α i + lam * D.cost i j) := by

    unfold transportRisk

    apply Finset.sum_le_sum

    intro i hi

    apply Finset.sum_le_sum

    intro j hj

    exact
      mul_le_mul_of_nonneg_left
        (hDual.2 i j)
        (π.nonneg i j)

  have hLambdaCost :
      lam * transportCost D π
        ≤
      lam * ρ := by

    exact
      mul_le_mul_of_nonneg_left
        hTransportBudget
        hDual.1

  calc
    expectedRisk q risk
        =
      transportRisk π risk := by
        symm
        exact
          transportRisk_eq_expectedRisk
            π
            risk

    _ ≤
      ∑ i, ∑ j,
        π.mass i j *
          (α i + lam * D.cost i j) :=
      hTransportRiskBound

    _ =
      (∑ i, p i * α i) +
        lam * transportCost D π := by

      exact
        transportDualExpansion
          D
          π
          α
          lam

    _ ≤
      (∑ i, p i * α i) +
        lam * ρ := by

      exact
        add_le_add_right
          hLambdaCost
          (∑ i, p i * α i)

    _ =
      transportDualObjective
        p
        ρ
        α
        lam := by
      rfl

/--
Primal-dual equality certifies that a feasible candidate is genuinely
worst-case.

This is the key theorem for connecting the formal DRO model to an
optimization implementation: it is enough to exhibit

* primal feasibility of `qStar`,
* dual feasibility of `(α, λ)`, and
* equality of primal and dual objective values.
-/
theorem isWorstCase_of_primalDualEquality
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p risk qStar : ModeDistribution d}
    {ρ lam : ℝ}
    {α : Fin d → ℝ}
    (hStarFeasible :
      qStar ∈
        FiniteWassersteinAmbiguity
          D
          p
          ρ)
    (hDual :
      TransportDualFeasible
        D
        risk
        α
        lam)
    (hEquality :
      expectedRisk qStar risk =
        transportDualObjective
          p
          ρ
          α
          lam) :
    IsWorstCase
      (FiniteWassersteinAmbiguity
        D
        p
        ρ)
      risk
      qStar := by

  constructor

  · exact hStarFeasible

  · intro q hq

    calc
      expectedRisk q risk
          ≤
        transportDualObjective
          p
          ρ
          α
          lam :=
        finiteTransport_weakDuality
          D
          hq
          hDual

      _ =
        expectedRisk qStar risk :=
        hEquality.symm

/--
A primal-dual certified candidate has risk at least as large as nominal risk,
provided the nominal distribution is a probability vector and `ρ ≥ 0`.
-/
theorem nominalRisk_le_primalDualCertifiedRisk
    {d : ℕ}
    (D : FiniteGroundCost d)
    {p risk qStar : ModeDistribution d}
    {ρ lam : ℝ}
    {α : Fin d → ℝ}
    (hp :
      IsProbabilityVector p)
    (hρ :
      0 ≤ ρ)
    (hStarFeasible :
      qStar ∈
        FiniteWassersteinAmbiguity
          D
          p
          ρ)
    (hDual :
      TransportDualFeasible
        D
        risk
        α
        lam)
    (hEquality :
      expectedRisk qStar risk =
        transportDualObjective
          p
          ρ
          α
          lam) :
    expectedRisk p risk ≤
      expectedRisk qStar risk := by

  have hWorst :
      IsWorstCase
        (FiniteWassersteinAmbiguity
          D
          p
          ρ)
        risk
        qStar :=
    isWorstCase_of_primalDualEquality
      D
      hStarFeasible
      hDual
      hEquality

  exact
    nominalRisk_le_finiteWassersteinWorstCaseRisk
      D
      hp
      hρ
      hWorst

end DROSafety.DRO
