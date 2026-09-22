import DROSafety.MainTheorem
import DROSafety.ScenarioTheorem

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Mode-Stratified Sample Complexity

This file formalizes the sample-allocation condition underlying the
mode-stratified DRO scenario theorem.

Each mode `m` may use its own:

* scenario count `S m`,
* support cap `nBar m`,
* confidence parameter `β m`.

An allocation is considered robustly admissible when every probability
distribution in the ambiguity set assigns weighted scenario risk at most
`εTarget`.
-/

namespace DROSafety

variable {K : ℕ}

/--
The total number of scenarios used by a mode-stratified allocation.
-/
def totalSamples (S : Fin K → ℕ) : ℕ :=
  ∑ m, S m

/--
A mode-stratified scenario allocation satisfies the DRO risk target
when every distribution in the ambiguity set gives weighted scenario
risk at most `εTarget`.
-/
def RobustScenarioAllocation
    (U : AmbiguitySet K)
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ) :
    Prop :=
  ∀ q, q ∈ U →
    dot q (modeScenarioEpsilon S nBar β) ≤ εTarget

/--
Unpack a robust scenario allocation into exactly the robust inequality
required by the deterministic DRO aggregation theorem.
-/
theorem RobustScenarioAllocation.bound
    {U : AmbiguitySet K}
    {S nBar : Fin K → ℕ}
    {β : ModeVec K}
    {εTarget : ℝ}
    (h :
      RobustScenarioAllocation
        U S nBar β εTarget) :
    ∀ q, q ∈ U →
      dot q (modeScenarioEpsilon S nBar β) ≤ εTarget := by
  exact h

/--
If the true mode distribution belongs to the ambiguity set and the
mode-wise violation probabilities satisfy their scenario bounds,
then any robustly admissible sample allocation guarantees the desired
aggregate risk target.
-/
theorem robust_allocation_implies_risk_bound
    (U : AmbiguitySet K)
    (p V β : ModeVec K)
    (S nBar : Fin K → ℕ)
    (εTarget : ℝ)
    (hpProb : IsProbabilityVector p)
    (hpU : p ∈ U)
    (hCert :
      ModewiseScenarioCertified
        S nBar β V)
    (hAllocation :
      RobustScenarioAllocation
        U S nBar β εTarget) :
    dot p V ≤ εTarget := by
  exact
    dro_risk_bound_from_scenario_certificate
      U
      p
      V
      β
      S
      nBar
      εTarget
      hpProb
      hpU
      hCert
      hAllocation

/--
A sample allocation is no larger than another allocation in total
scenario count.
-/
def NoMoreSamples
    (S₁ S₂ : Fin K → ℕ) :
    Prop :=
  totalSamples S₁ ≤ totalSamples S₂

/--
A feasible sample allocation satisfies:

1. valid per-mode scenario counts,
2. valid confidence parameters,
3. the robust aggregate risk constraint.
-/
def FeasibleSampleAllocation
    (U : AmbiguitySet K)
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ) :
    Prop :=
  ValidScenarioCounts S nBar ∧
  IsScenarioConfidenceVector β ∧
  RobustScenarioAllocation
    U S nBar β εTarget

/--
`Sstar` is sample-minimal among all feasible allocations with the same
support caps, confidence allocation, ambiguity set, and risk target.
-/
def IsSampleMinimal
    (U : AmbiguitySet K)
    (Sstar nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ) :
    Prop :=
  FeasibleSampleAllocation
      U Sstar nBar β εTarget ∧
  ∀ S,
    FeasibleSampleAllocation
        U S nBar β εTarget →
    totalSamples Sstar ≤ totalSamples S

theorem sample_minimal_valid_counts
    (U : AmbiguitySet K)
    (Sstar nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (hMinimal :
      IsSampleMinimal
        U Sstar nBar β εTarget) :
    ValidScenarioCounts Sstar nBar := by
  exact hMinimal.1.1

theorem sample_minimal_valid_confidence
    (U : AmbiguitySet K)
    (Sstar nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (hMinimal :
      IsSampleMinimal
        U Sstar nBar β εTarget) :
    IsScenarioConfidenceVector β := by
  exact hMinimal.1.2.1

theorem sample_minimal_is_safe
    (U : AmbiguitySet K)
    (Sstar nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (hMinimal :
      IsSampleMinimal
        U Sstar nBar β εTarget) :
    RobustScenarioAllocation
      U Sstar nBar β εTarget := by
  exact hMinimal.1.2.2

/--
A sample-minimal feasible allocation guarantees the deterministic
aggregate risk bound whenever the true distribution lies in the
ambiguity set and the mode-wise scenario certificates hold.
-/
theorem sample_minimal_implies_risk_bound
    (U : AmbiguitySet K)
    (p V β : ModeVec K)
    (Sstar nBar : Fin K → ℕ)
    (εTarget : ℝ)
    (hpProb : IsProbabilityVector p)
    (hpU : p ∈ U)
    (hCert :
      ModewiseScenarioCertified
        Sstar nBar β V)
    (hMinimal :
      IsSampleMinimal
        U Sstar nBar β εTarget) :
    dot p V ≤ εTarget := by
  exact
    robust_allocation_implies_risk_bound
      U
      p
      V
      β
      Sstar
      nBar
      εTarget
      hpProb
      hpU
      hCert
      (sample_minimal_is_safe
        U
        Sstar
        nBar
        β
        εTarget
        hMinimal)


/--
End-to-end safety guarantee for a sample-minimal mode-stratified
allocation.

The same sample allocation `Sstar` is assumed to satisfy the robust
allocation constraint for every ambiguity set that can arise from an
outcome `ω`.
-/
theorem sample_minimal_mode_stratified_dro_safety
    {Ω : Type*}
    [MeasurableSpace Ω]
    {K : ℕ}
    (μ : MeasureTheory.Measure Ω)
    (U : Ω → AmbiguitySet K)
    (p : ModeVec K)
    (V : Ω → ModeVec K)
    (Sstar nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (βDRO βTotal : ENNReal)
    (hpProb :
      IsProbabilityVector p)
    (hMinimal :
      ∀ ω,
        IsSampleMinimal
          (U ω)
          Sstar
          nBar
          β
          εTarget)
    (hScenario :
      ModewiseScenarioGuarantee
        μ Sstar nBar β V)
    (hAmbiguity :
      μ (AmbiguityFailure U p) ≤ βDRO)
    (hBudget :
      βDRO + ∑ m, ENNReal.ofReal (β m) ≤
        βTotal) :
    μ (RiskFailure p V εTarget) ≤
      βTotal := by

  have hRobust :
      ∀ ω q, q ∈ U ω →
        dot q
          (modeScenarioEpsilon
            Sstar nBar β) ≤
          εTarget := by
    intro ω q hq

    have hAllocation :
        RobustScenarioAllocation
          (U ω)
          Sstar
          nBar
          β
          εTarget :=
      sample_minimal_is_safe
        (U ω)
        Sstar
        nBar
        β
        εTarget
        (hMinimal ω)

    exact hAllocation q hq

  exact
    mode_stratified_dro_safety
      μ
      U
      p
      V
      Sstar
      nBar
      β
      εTarget
      βDRO
      βTotal
      hpProb
      hScenario
      hAmbiguity
      hRobust
      hBudget


end DROSafety
