import DROSafety.SparseModes
import DROSafety.MainTheorem

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Sparse-Mode Confidence Composition

This file extends the sparse-mode deterministic certificate to a
high-probability guarantee.

Scenario-theoretic confidence is required only for sampled modes.
Omitted modes receive no scenario certificate and therefore consume
no scenario-confidence budget. Their risk is handled conservatively
through the sparse risk vector, which assigns them conditional risk 1.
-/

namespace DROSafety

variable {K : ℕ}
variable {Ω : Type*}
variable [MeasurableSpace Ω]

/--
The scenario failure event associated with a mode under sparse sampling.

For an omitted mode, the failure event is empty because no scenario
certificate is claimed.

For a sampled mode, this is the usual scenario failure event.
-/
def SampledScenarioFailure
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (V : Ω → ModeVec K)
    (m : Fin K) :
    Set Ω :=
  if m ∈ omitted then
    ∅
  else
    ScenarioFailure S nBar β V m

/--
Confidence budget associated with a mode under sparse sampling.

Omitted modes consume zero scenario-confidence budget.
-/
noncomputable def sampledConfidenceBudget
    (omitted : Finset (Fin K))
    (β : ModeVec K)
    (m : Fin K) :
    ENNReal :=
  if m ∈ omitted then
    0
  else
    ENNReal.ofReal (β m)

/--
If the usual scenario bound holds for every sampled mode, then the
sparse failure event for every mode is bounded by its sparse confidence
budget.
-/
theorem sampledScenarioFailure_measure_le
    (μ : MeasureTheory.Measure Ω)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (V : Ω → ModeVec K)
    (hScenario :
      ∀ m, m ∉ omitted →
        μ (ScenarioFailure S nBar β V m) ≤
          ENNReal.ofReal (β m)) :
    ∀ m,
      μ (SampledScenarioFailure
        omitted S nBar β V m) ≤
        sampledConfidenceBudget omitted β m := by
  intro m
  by_cases hm : m ∈ omitted
  · simp [SampledScenarioFailure,
          sampledConfidenceBudget,
          hm]
  · simpa [SampledScenarioFailure,
           sampledConfidenceBudget,
           hm] using
      hScenario m hm

/--
Outside the ambiguity-set failure event and all sampled-mode scenario
failure events,

* the true distribution lies in the ambiguity set;
* every mode has conditional violation probability at most 1;
* every sampled mode satisfies its scenario bound.

Therefore the sparse mode-wise certificate holds.
-/
theorem goodOutcome_has_sparse_certificates
    (U : Ω → AmbiguitySet K)
    (p : ModeVec K)
    (V : Ω → ModeVec K)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (ω : Ω)
    (hVBound :
      ∀ ω m, V ω m ≤ 1)
    (hGood :
      ω ∉ TotalFailure
        (AmbiguityFailure U p)
        (SampledScenarioFailure
          omitted S nBar β V)) :
    p ∈ U ω ∧
      SparseModewiseCertified
        omitted S nBar β (V ω) := by
  constructor

  · have hNotAmb :
        ω ∉ AmbiguityFailure U p := by
      intro hAmb
      apply hGood
      exact Or.inl hAmb

    simpa [AmbiguityFailure] using hNotAmb

  · constructor

    · intro m
      exact hVBound ω m

    · intro m hm

      have hNotModes :
          ω ∉ AnyModeFailure
            (SampledScenarioFailure
              omitted S nBar β V) := by
        intro hModes
        apply hGood
        exact Or.inr hModes

      have hNotM :
          ω ∉ SampledScenarioFailure
            omitted S nBar β V m := by
        intro hFail
        apply hNotModes
        unfold AnyModeFailure
        exact Set.mem_iUnion.2 ⟨m, hFail⟩

      have hNotScenario :
          ω ∉ ScenarioFailure
            S nBar β V m := by
        simpa [SampledScenarioFailure, hm] using hNotM

      have hNotLt :
          ¬ scenarioEpsilon
              (S m)
              (nBar m)
              (β m) <
            V ω m := by
        simpa [ScenarioFailure] using hNotScenario

      exact not_lt.mp hNotLt

/--
If the final aggregate risk bound fails, then either

* ambiguity-set coverage failed, or
* at least one sampled-mode scenario certificate failed.

Omitted modes do not appear as scenario failures.
-/
theorem sparseRiskFailure_subset_totalFailure
    (U : Ω → AmbiguitySet K)
    (p : ModeVec K)
    (V : Ω → ModeVec K)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (hpProb :
      IsProbabilityVector p)
    (hVBound :
      ∀ ω m, V ω m ≤ 1)
    (hRobust :
      ∀ ω,
        RobustSparseScenarioAllocation
          (U ω)
          omitted
          S
          nBar
          β
          εTarget) :
    RiskFailure p V εTarget ⊆
      TotalFailure
        (AmbiguityFailure U p)
        (SampledScenarioFailure
          omitted S nBar β V) := by
  intro ω hRisk
  by_contra hGood

  obtain ⟨hpU, hCert⟩ :=
    goodOutcome_has_sparse_certificates
      U
      p
      V
      omitted
      S
      nBar
      β
      ω
      hVBound
      hGood

  have hSafe :
      dot p (V ω) ≤ εTarget :=
    sparse_allocation_implies_risk_bound
      (U ω)
      omitted
      p
      (V ω)
      β
      S
      nBar
      εTarget
      hpProb
      hpU
      hCert
      (hRobust ω)

  exact (not_lt_of_ge hSafe) hRisk

/--
High-probability sparse-mode DRO safety theorem.

Only sampled modes contribute scenario-confidence failure probability.
Omitted modes contribute zero scenario-confidence budget.
-/
theorem sparseRiskFailure_measure_le
    (μ : MeasureTheory.Measure Ω)
    (U : Ω → AmbiguitySet K)
    (p : ModeVec K)
    (V : Ω → ModeVec K)
    (omitted : Finset (Fin K))
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (βDRO : ENNReal)
    (hpProb :
      IsProbabilityVector p)
    (hVBound :
      ∀ ω m, V ω m ≤ 1)
    (hRobust :
      ∀ ω,
        RobustSparseScenarioAllocation
          (U ω)
          omitted
          S
          nBar
          β
          εTarget)
    (hAmbiguity :
      μ (AmbiguityFailure U p) ≤ βDRO)
    (hScenario :
      ∀ m, m ∉ omitted →
        μ (ScenarioFailure
          S nBar β V m) ≤
          ENNReal.ofReal (β m)) :
    μ (RiskFailure p V εTarget) ≤
      βDRO +
        ∑ m,
          sampledConfidenceBudget
            omitted β m := by

  have hSparseScenario :
      ∀ m,
        μ (SampledScenarioFailure
          omitted S nBar β V m) ≤
          sampledConfidenceBudget
            omitted β m :=
    sampledScenarioFailure_measure_le
      μ
      omitted
      S
      nBar
      β
      V
      hScenario

  calc
    μ (RiskFailure p V εTarget)
        ≤ μ
          (TotalFailure
            (AmbiguityFailure U p)
            (SampledScenarioFailure
              omitted S nBar β V)) :=
      MeasureTheory.measure_mono
        (sparseRiskFailure_subset_totalFailure
          U
          p
          V
          omitted
          S
          nBar
          β
          εTarget
          hpProb
          hVBound
          hRobust)

    _ ≤ βDRO +
          ∑ m,
            sampledConfidenceBudget
              omitted β m :=
      totalFailure_measure_le_budget
        μ
        (AmbiguityFailure U p)
        (SampledScenarioFailure
          omitted S nBar β V)
        βDRO
        (sampledConfidenceBudget
          omitted β)
        hAmbiguity
        hSparseScenario

end DROSafety
