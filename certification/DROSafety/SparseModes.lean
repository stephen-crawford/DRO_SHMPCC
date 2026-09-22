import DROSafety.SampleComplexity

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Sparse / Omitted Mode Certification

This file formalizes safe treatment of modes that receive no scenario
samples.

An omitted mode is not assumed impossible. Instead, its conditional
violation probability is conservatively bounded by `1`, and the risk
contributed by omitted modes is controlled by their worst-case
probability mass over the DRO ambiguity set.
-/

open scoped BigOperators

namespace DROSafety

variable {K : ℕ}

/--
Scenario-risk vector when some modes are omitted.

For a sampled mode, use the Safe-Horizon scenario bound.

For an omitted mode, use the conservative bound `1`.
-/
noncomputable def sparseScenarioEpsilon
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K) :
    ModeVec K :=
  fun m =>
    if m ∈ omitted then
      1
    else
      scenarioEpsilon (S m) (nBar m) (β m)

/--
A sparse mode-wise certificate requires:

1. every conditional violation probability is at most `1`;
2. every non-omitted mode satisfies its scenario certificate.

No scenario certificate is assumed for omitted modes.
-/
def SparseModewiseCertified
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β V : ModeVec K) :
    Prop :=
  (∀ m, V m ≤ 1) ∧
  (∀ m, m ∉ omitted →
    V m ≤ scenarioEpsilon
      (S m) (nBar m) (β m))

/--
A sparse certificate implies the pointwise upper bound

    Vₘ ≤ ε̃ₘ,

where omitted modes have `ε̃ₘ = 1`.
-/
theorem sparseModewiseCertified_bound
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β V : ModeVec K)
    (hCert :
      SparseModewiseCertified
        omitted S nBar β V) :
    ∀ m,
      V m ≤ sparseScenarioEpsilon
        omitted S nBar β m := by
  intro m
  rcases hCert with ⟨hProb, hSampled⟩
  by_cases hm : m ∈ omitted
  · simpa [sparseScenarioEpsilon, hm] using hProb m
  · simpa [sparseScenarioEpsilon, hm] using
      hSampled m hm

/--
The total probability mass assigned to a set of omitted modes.
-/
def omittedMass
    (q : ModeVec K)
    (omitted : Finset (Fin K)) :
    ℝ :=
  ∑ m ∈ omitted, q m

/--
`η` is an ambiguity-set upper bound on the total probability mass
of the omitted modes.
-/
def OmittedMassBound
    (U : AmbiguitySet K)
    (omitted : Finset (Fin K))
    (η : ℝ) :
    Prop :=
  ∀ q, q ∈ U →
    omittedMass q omitted ≤ η

/--
If each conditional violation probability is at most one, then the
violation risk contributed by omitted modes is no larger than their
total probability mass.
-/
theorem omitted_violation_le_mass
    (q V : ModeVec K)
    (omitted : Finset (Fin K))
    (hq : ∀ m, 0 ≤ q m)
    (hV : ∀ m, V m ≤ 1) :
    (∑ m ∈ omitted, q m * V m) ≤
      omittedMass q omitted := by
  unfold omittedMass
  apply Finset.sum_le_sum
  intro m hm
  have h :=
    mul_le_mul_of_nonneg_left
      (hV m)
      (hq m)
  simpa using h

/--
If the ambiguity set guarantees that omitted modes have total mass
at most `η`, then their total violation contribution is also at most
`η`.
-/
theorem omitted_violation_le_bound
    (U : AmbiguitySet K)
    (q V : ModeVec K)
    (omitted : Finset (Fin K))
    (η : ℝ)
    (hqProb : IsProbabilityVector q)
    (hqU : q ∈ U)
    (hV : ∀ m, V m ≤ 1)
    (hMass :
      OmittedMassBound U omitted η) :
    (∑ m ∈ omitted, q m * V m) ≤ η := by
  have hRiskMass :
      (∑ m ∈ omitted, q m * V m) ≤
        omittedMass q omitted :=
    omitted_violation_le_mass
      q V omitted hqProb.1 hV

  have hMassBound :
      omittedMass q omitted ≤ η :=
    hMass q hqU

  exact le_trans hRiskMass hMassBound

/--
A robust sparse allocation uses the conservative risk vector with
risk `1` assigned to omitted modes.
-/
def RobustSparseScenarioAllocation
    (U : AmbiguitySet K)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ) :
    Prop :=
  ∀ q, q ∈ U →
    dot q
      (sparseScenarioEpsilon
        omitted S nBar β) ≤
      εTarget

/--
If the true mode distribution belongs to the ambiguity set,
the sparse conditional certificates hold, and the sparse robust
risk constraint holds, then the true aggregate violation probability
is bounded by `εTarget`.
-/
theorem sparse_allocation_implies_risk_bound
    (U : AmbiguitySet K)
    (omitted : Finset (Fin K))
    (p V β : ModeVec K)
    (S nBar : Fin K → ℕ)
    (εTarget : ℝ)
    (hpProb : IsProbabilityVector p)
    (hpU : p ∈ U)
    (hCert :
      SparseModewiseCertified
        omitted S nBar β V)
    (hRobust :
      RobustSparseScenarioAllocation
        U omitted S nBar β εTarget) :
    dot p V ≤ εTarget := by
  exact
    dro_risk_bound
      U
      p
      V
      (sparseScenarioEpsilon
        omitted S nBar β)
      εTarget
      hpProb
      hpU
      (sparseModewiseCertified_bound
        omitted S nBar β V hCert)
      hRobust

end DROSafety
