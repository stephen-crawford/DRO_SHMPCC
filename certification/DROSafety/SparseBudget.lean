import DROSafety.SparseConfidence

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Sparse Risk-Budget Decomposition

This file decomposes the sparse DRO risk certificate into

1. risk contributed by sampled modes, and
2. probability mass assigned to omitted modes.

For omitted modes, conditional violation risk is conservatively
bounded by one. Therefore their contribution to total risk is
bounded by their total probability mass.
-/

open scoped BigOperators

namespace DROSafety

variable {K : ℕ}

/--
Scenario-risk vector containing only sampled modes.

Omitted modes contribute zero here because their contribution is
handled separately through `omittedMass`.
-/
noncomputable def sampledScenarioEpsilon
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K) :
    ModeVec K :=
  fun m =>
    if m ∈ omitted then
      0
    else
      scenarioEpsilon (S m) (nBar m) (β m)

/--
Indicator vector for omitted modes.
-/
def omittedIndicator
    (omitted : Finset (Fin K)) :
    ModeVec K :=
  fun m =>
    if m ∈ omitted then 1 else 0

/--
The certified risk contribution from sampled modes.
-/
noncomputable def sampledRisk
    (q : ModeVec K)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K) :
    ℝ :=
  dot q (sampledScenarioEpsilon omitted S nBar β)

/--
Pointwise decomposition of the sparse risk vector.

For every mode,

    sparse ε = sampled ε + omitted indicator.
-/
theorem sparseScenarioEpsilon_eq_sampled_add_indicator
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (m : Fin K) :
    sparseScenarioEpsilon omitted S nBar β m =
      sampledScenarioEpsilon omitted S nBar β m +
        omittedIndicator omitted m := by
  by_cases hm : m ∈ omitted
  · simp [
      sparseScenarioEpsilon,
      sampledScenarioEpsilon,
      omittedIndicator,
      hm
    ]
  · simp [
      sparseScenarioEpsilon,
      sampledScenarioEpsilon,
      omittedIndicator,
      hm
    ]

/--
Weighting the omitted-mode indicator by `q` gives exactly the
total probability mass of the omitted modes.
-/
theorem dot_omittedIndicator_eq_omittedMass
    (q : ModeVec K)
    (omitted : Finset (Fin K)) :
    dot q (omittedIndicator omitted) =
      omittedMass q omitted := by
  classical
  simp [dot, omittedIndicator, omittedMass]

/--
Exact decomposition of sparse certified risk:

    qᵀ ε_sparse
      =
    sampledRisk(q) + omittedMass(q).
-/
theorem sparseRisk_decomposition
    (q : ModeVec K)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K) :
    dot q
        (sparseScenarioEpsilon
          omitted S nBar β) =
      sampledRisk
        q omitted S nBar β +
      omittedMass q omitted := by
  unfold sampledRisk

  calc
    dot q
        (sparseScenarioEpsilon
          omitted S nBar β)
        =
      dot q
          (sampledScenarioEpsilon
            omitted S nBar β) +
      dot q
          (omittedIndicator omitted) := by
            unfold dot
            simp_rw [
              sparseScenarioEpsilon_eq_sampled_add_indicator
            ]
            simp_rw [mul_add]
            rw [Finset.sum_add_distrib]

    _ =
      dot q
          (sampledScenarioEpsilon
            omitted S nBar β) +
      omittedMass q omitted := by
        rw [dot_omittedIndicator_eq_omittedMass]

/--
Uniform ambiguity-set upper bound on the risk contribution
from sampled modes.
-/
def SampledRiskBound
    (U : AmbiguitySet K)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εSampled : ℝ) :
    Prop :=
  ∀ q, q ∈ U →
    sampledRisk
      q omitted S nBar β ≤ εSampled

/--
If

    sampled-mode risk ≤ εSampled,

    omitted-mode mass ≤ η,

and

    εSampled + η ≤ εTarget,

then the complete sparse DRO allocation satisfies the desired
risk target.
-/
theorem sampled_plus_omitted_implies_sparse_allocation
    (U : AmbiguitySet K)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εSampled η εTarget : ℝ)
    (hSampled :
      SampledRiskBound
        U omitted S nBar β εSampled)
    (hOmitted :
      OmittedMassBound
        U omitted η)
    (hBudget :
      εSampled + η ≤ εTarget) :
    RobustSparseScenarioAllocation
      U omitted S nBar β εTarget := by
  intro q hq

  rw [sparseRisk_decomposition]

  have hSampledQ :
      sampledRisk
        q omitted S nBar β ≤ εSampled :=
    hSampled q hq

  have hOmittedQ :
      omittedMass q omitted ≤ η :=
    hOmitted q hq

  exact
    le_trans
      (add_le_add hSampledQ hOmittedQ)
      hBudget

/--
Omitted modes receive exactly zero scenarios.
-/
def ZeroSamplesOnOmitted
    (omitted : Finset (Fin K))
    (S : Fin K → ℕ) :
    Prop :=
  ∀ m, m ∈ omitted → S m = 0

/--
Only sampled modes need to satisfy the support-count condition.
-/
def ValidSampledScenarioCounts
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ) :
    Prop :=
  ∀ m, m ∉ omitted →
    nBar m < S m

/--
Only sampled modes require a scenario confidence parameter.
-/
def ValidSampledConfidence
    (omitted : Finset (Fin K))
    (β : ModeVec K) :
    Prop :=
  ∀ m, m ∉ omitted →
    IsScenarioConfidence (β m)


/--
A feasible sparse sample allocation.

Omitted modes receive zero samples. Sampled modes satisfy the
scenario-theoretic count and confidence requirements. The sampled
risk plus worst-case omitted mass must fit inside the total risk
budget.
-/
def FeasibleSparseAllocation
    (U : AmbiguitySet K)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εSampled η εTarget : ℝ) :
    Prop :=
  ZeroSamplesOnOmitted omitted S ∧
  ValidSampledScenarioCounts omitted S nBar ∧
  ValidSampledConfidence omitted β ∧
  SampledRiskBound
    U omitted S nBar β εSampled ∧
  OmittedMassBound U omitted η ∧
  εSampled + η ≤ εTarget

theorem feasibleSparseAllocation_is_safe
    (U : AmbiguitySet K)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εSampled η εTarget : ℝ)
    (h :
      FeasibleSparseAllocation
        U
        omitted
        S
        nBar
        β
        εSampled
        η
        εTarget) :
    RobustSparseScenarioAllocation
      U omitted S nBar β εTarget := by
  exact
    sampled_plus_omitted_implies_sparse_allocation
      U
      omitted
      S
      nBar
      β
      εSampled
      η
      εTarget
      h.2.2.2.1
      h.2.2.2.2.1
      h.2.2.2.2.2

end DROSafety
