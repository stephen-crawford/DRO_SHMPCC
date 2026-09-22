import DROSafety.MainTheorem
import DROSafety.SampleComplexity
import DROSafety.SparseModes
import DROSafety.SparseConfidence
import DROSafety.SparseBudget
import DROSafety.SparseOptimization

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Current Verification Chain

This file exposes the current end-to-end deterministic verification chain
for the mode-stratified DRO scenario-safety certificate.

## Current verified chain

1. Scenario risk function

   `εₘ := scenarioEpsilon Sₘ nₘ βₘ`

2. Conditional scenario certificate

   `hCert : ∀ m, Vₘ ≤ εₘ`

   At present, this is an assumption. The probabilistic scenario theorem
   establishing this statement has not yet been formalized.

3. Weighted conditional aggregation

   From

   `pₘ ≥ 0`

   and

   `Vₘ ≤ εₘ`

   we obtain

   `pᵀV ≤ pᵀε`.

4. DRO ambiguity certificate

   From

   `p ∈ U`

   and

   `∀ q ∈ U, qᵀε ≤ εTarget`

   we obtain

   `pᵀε ≤ εTarget`.

5. Transitivity

   Combining

   `pᵀV ≤ pᵀε`

   and

   `pᵀε ≤ εTarget`

   gives

   `pᵀV ≤ εTarget`.
-/

namespace DROSafety

#check scenarioEpsilon
#check modeScenarioEpsilon
#check ModewiseScenarioCertified
#check modewiseScenarioCertified_bound
#check dot_le_dot_of_nonneg_left
#check dro_risk_bound
#check dro_risk_bound_from_scenario_certificate
#check IsScenarioConfidence
#check IsScenarioConfidenceVector
#check ValidScenarioCounts

#check ModewiseScenarioGuarantee
#check ModewiseScenarioGuarantee.failure_le

#check riskFailure_measure_le
#check riskFailure_measure_le_total_budget
#check mode_stratified_dro_safety

#check totalSamples
#check RobustScenarioAllocation
#check FeasibleSampleAllocation
#check IsSampleMinimal
#check robust_allocation_implies_risk_bound
#check sample_minimal_is_safe

#check sample_minimal_implies_risk_bound
#check sample_minimal_mode_stratified_dro_safety

#check sparseScenarioEpsilon
#check SparseModewiseCertified
#check sparseModewiseCertified_bound

#check omittedMass
#check OmittedMassBound
#check omitted_violation_le_mass
#check omitted_violation_le_bound

#check RobustSparseScenarioAllocation
#check sparse_allocation_implies_risk_bound

#check SampledScenarioFailure
#check sampledConfidenceBudget
#check sampledScenarioFailure_measure_le
#check goodOutcome_has_sparse_certificates
#check sparseRiskFailure_subset_totalFailure
#check sparseRiskFailure_measure_le

#check sampledScenarioEpsilon
#check sampledRisk
#check sparseRisk_decomposition

#check SampledRiskBound
#check sampled_plus_omitted_implies_sparse_allocation

#check ZeroSamplesOnOmitted
#check ValidSampledScenarioCounts
#check ValidSampledConfidence
#check FeasibleSparseAllocation
#check feasibleSparseAllocation_is_safe

#check SparseAllocationFeasible
#check SparseAllocationFeasible.is_safe
#check SparseAllocationFeasible.zero_on_omitted
#check SparseAllocationFeasible.valid_sampled_counts

#check IsSparseSampleMinimal
#check sparse_sample_minimal_is_safe
#check sparse_sample_minimal_zero_on_omitted

#print IsSparseSampleMinimal
#print sparse_sample_minimal_is_safe
#print axioms sparse_sample_minimal_is_safe

#print sparseRisk_decomposition
#print sampled_plus_omitted_implies_sparse_allocation
#print axioms feasibleSparseAllocation_is_safe

#print sparseRiskFailure_measure_le
#print axioms sparseRiskFailure_measure_le

#print axioms mode_stratified_dro_safety
#print axioms sparseRiskFailure_measure_le
#print axioms sparse_sample_minimal_is_safe

#print sparse_allocation_implies_risk_bound
#print axioms sparse_allocation_implies_risk_bound

#print sample_minimal_mode_stratified_dro_safety
#print axioms sample_minimal_mode_stratified_dro_safety

#print axioms mode_stratified_dro_safety

#print mode_stratified_dro_safety
#print axioms mode_stratified_dro_safety

#print dro_risk_bound_from_scenario_certificate
#print axioms dro_risk_bound_from_scenario_certificate

/--
An explicit walkthrough of the current verification chain.

This example proves the same result as
`dro_risk_bound_from_scenario_certificate`, but exposes each intermediate
step so that the proof state can be inspected interactively in VS Code.
-/
example
    {K : ℕ}
    (U : AmbiguitySet K)
    (p V β : ModeVec K)
    (S n : Fin K → ℕ)
    (εTarget : ℝ)
    (hpProb : IsProbabilityVector p)
    (hpU : p ∈ U)
    (hCert : ModewiseScenarioCertified S n β V)
    (hRobust :
      ∀ q, q ∈ U →
        dot q (modeScenarioEpsilon S n β) ≤ εTarget) :
    dot p V ≤ εTarget := by

  -- Step 1: obtain the pointwise mode-wise scenario bounds.
  have hMode :
      ∀ m, V m ≤ modeScenarioEpsilon S n β m :=
    modewiseScenarioCertified_bound S n β V hCert

  -- Step 2: weight each bound by the true nonnegative mode probability.
  have hWeighted :
      dot p V ≤ dot p (modeScenarioEpsilon S n β) :=
    dot_le_dot_of_nonneg_left
      p
      V
      (modeScenarioEpsilon S n β)
      hpProb.1
      hMode

  -- Step 3: because the true distribution p belongs to the ambiguity set,
  -- the robust ambiguity-set bound also applies to p.
  have hDRO :
      dot p (modeScenarioEpsilon S n β) ≤ εTarget :=
    hRobust p hpU

  -- Step 4: combine the two inequalities.
  exact le_trans hWeighted hDRO

end DROSafety
