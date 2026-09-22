import DROSafety.Aggregation
import DROSafety.ScenarioBound
import DROSafety.Confidence
import DROSafety.ScenarioTheorem

set_option linter.style.header false
set_option linter.style.emptyLine false

/-!
# Main DRO Scenario Safety Theorem

This file connects the per-mode scenario bounds, confidence composition,
and distributionally robust aggregation certificate.
-/

namespace DROSafety

variable {K : ℕ}

/--
If

1. `p` is the true mode probability distribution,
2. `p` lies in the ambiguity set `U`,
3. each mode satisfies its scenario certificate,
4. the worst-case ambiguity-weighted scenario risk is at most `εTarget`,

then the true total violation probability is at most `εTarget`.
-/
theorem dro_risk_bound_from_scenario_certificate
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
  exact dro_risk_bound
    U
    p
    V
    (modeScenarioEpsilon S n β)
    εTarget
    hpProb
    hpU
    (modewiseScenarioCertified_bound S n β V hCert)
    hRobust

/--
On every outcome outside the total certification failure event,
the true aggregate violation probability satisfies the desired
DRO risk bound.
-/
theorem dro_risk_bound_on_good_outcome
    {Ω : Type*}
    [MeasurableSpace Ω]
    {K : ℕ}
    (U : Ω → AmbiguitySet K)
    (p : ModeVec K)
    (V : Ω → ModeVec K)
    (S n : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (ω : Ω)
    (hpProb : IsProbabilityVector p)
    (hGood :
      ω ∉ TotalFailure
        (AmbiguityFailure U p)
        (ScenarioFailure S n β V))
    (hRobust :
      ∀ q, q ∈ U ω →
        dot q (modeScenarioEpsilon S n β) ≤ εTarget) :
    dot p (V ω) ≤ εTarget := by
  obtain ⟨hpU, hCert⟩ :=
    goodOutcome_has_certificates
      U p V S n β ω hGood
  exact
    dro_risk_bound_from_scenario_certificate
      (U ω)
      p
      (V ω)
      β
      S
      n
      εTarget
      hpProb
      hpU
      hCert
      hRobust

/--
The event that the final aggregate violation probability exceeds
the desired target.
-/
def RiskFailure
    {Ω : Type*}
    {K : ℕ}
    (p : ModeVec K)
    (V : Ω → ModeVec K)
    (εTarget : ℝ) :
    Set Ω :=
  {ω | εTarget < dot p (V ω)}

/--
Any violation of the final DRO risk bound must occur inside the
total certification failure event.
-/
theorem riskFailure_subset_totalFailure
    {Ω : Type*}
    [MeasurableSpace Ω]
    {K : ℕ}
    (U : Ω → AmbiguitySet K)
    (p : ModeVec K)
    (V : Ω → ModeVec K)
    (S n : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (hpProb : IsProbabilityVector p)
    (hRobust :
      ∀ ω q, q ∈ U ω →
        dot q (modeScenarioEpsilon S n β) ≤ εTarget) :
    RiskFailure p V εTarget ⊆
      TotalFailure
        (AmbiguityFailure U p)
        (ScenarioFailure S n β V) := by
  intro ω hRisk
  by_contra hGood

  have hSafe :
      dot p (V ω) ≤ εTarget :=
    dro_risk_bound_on_good_outcome
      U
      p
      V
      S
      n
      β
      εTarget
      ω
      hpProb
      hGood
      (hRobust ω)

  exact (not_lt_of_ge hSafe) hRisk

/--
High-probability mode-stratified DRO safety certificate.

The probability that the final aggregate risk exceeds `εTarget`
is bounded by the ambiguity-set failure budget plus the sum of
the mode-wise scenario failure budgets.
-/
theorem riskFailure_measure_le
    {Ω : Type*}
    [MeasurableSpace Ω]
    {K : ℕ}
    (μ : MeasureTheory.Measure Ω)
    (U : Ω → AmbiguitySet K)
    (p : ModeVec K)
    (V : Ω → ModeVec K)
    (S n : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (βDRO : ENNReal)
    (hpProb : IsProbabilityVector p)
    (hRobust :
      ∀ ω q, q ∈ U ω →
        dot q (modeScenarioEpsilon S n β) ≤ εTarget)
    (hAmbiguity :
      μ (AmbiguityFailure U p) ≤ βDRO)
    (hScenario :
      ∀ m,
        μ (ScenarioFailure S n β V m) ≤
          ENNReal.ofReal (β m)) :
    μ (RiskFailure p V εTarget) ≤
      βDRO + ∑ m, ENNReal.ofReal (β m) := by
  calc
    μ (RiskFailure p V εTarget)
        ≤ μ
          (TotalFailure
            (AmbiguityFailure U p)
            (ScenarioFailure S n β V)) :=
      MeasureTheory.measure_mono
        (riskFailure_subset_totalFailure
          U
          p
          V
          S
          n
          β
          εTarget
          hpProb
          hRobust)

    _ ≤ βDRO + ∑ m, ENNReal.ofReal (β m) :=
      totalFailure_measure_le_budget
        μ
        (AmbiguityFailure U p)
        (ScenarioFailure S n β V)
        βDRO
        (fun m => ENNReal.ofReal (β m))
        hAmbiguity
        hScenario

/--
If the ambiguity-set confidence budget plus all mode-wise
scenario confidence budgets is at most `βTotal`, then the
probability that the final aggregate risk exceeds `εTarget`
is at most `βTotal`.
-/
theorem riskFailure_measure_le_total_budget
    {Ω : Type*}
    [MeasurableSpace Ω]
    {K : ℕ}
    (μ : MeasureTheory.Measure Ω)
    (U : Ω → AmbiguitySet K)
    (p : ModeVec K)
    (V : Ω → ModeVec K)
    (S n : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (βDRO βTotal : ENNReal)
    (hpProb : IsProbabilityVector p)
    (hRobust :
      ∀ ω q, q ∈ U ω →
        dot q (modeScenarioEpsilon S n β) ≤ εTarget)
    (hAmbiguity :
      μ (AmbiguityFailure U p) ≤ βDRO)
    (hScenario :
      ∀ m,
        μ (ScenarioFailure S n β V m) ≤
          ENNReal.ofReal (β m))
    (hBudget :
      βDRO + ∑ m, ENNReal.ofReal (β m) ≤ βTotal) :
    μ (RiskFailure p V εTarget) ≤ βTotal := by
  have hFailure :
      μ (RiskFailure p V εTarget) ≤
        βDRO + ∑ m, ENNReal.ofReal (β m) :=
    riskFailure_measure_le
      μ
      U
      p
      V
      S
      n
      β
      εTarget
      βDRO
      hpProb
      hRobust
      hAmbiguity
      hScenario

  exact le_trans hFailure hBudget

/--
Main mode-stratified distributionally robust scenario safety theorem.

Suppose:

1. `p` is a valid true mode distribution;
2. the standard scenario theorem holds independently for each
   mode with sample count `Sₘ`, support cap `nBarₘ`, and confidence
   budget `βₘ`;
3. the true mode distribution lies outside the ambiguity set with
   probability at most `βDRO`;
4. for every outcome and every distribution in the corresponding
   ambiguity set, the ambiguity-weighted scenario risk is at most
   `εTarget`;
5. the total confidence budget is at most `βTotal`.

Then the probability that the true aggregate violation probability
exceeds `εTarget` is at most `βTotal`.
-/
theorem mode_stratified_dro_safety
    {Ω : Type*}
    [MeasurableSpace Ω]
    {K : ℕ}
    (μ : MeasureTheory.Measure Ω)
    (U : Ω → AmbiguitySet K)
    (p : ModeVec K)
    (V : Ω → ModeVec K)
    (S nBar : Fin K → ℕ)
    (β : ModeVec K)
    (εTarget : ℝ)
    (βDRO βTotal : ENNReal)
    (hpProb :
      IsProbabilityVector p)
    (hScenario :
      ModewiseScenarioGuarantee
        μ S nBar β V)
    (hAmbiguity :
      μ (AmbiguityFailure U p) ≤ βDRO)
    (hRobust :
      ∀ ω q, q ∈ U ω →
        dot q
          (modeScenarioEpsilon S nBar β) ≤
            εTarget)
    (hBudget :
      βDRO + ∑ m, ENNReal.ofReal (β m) ≤
        βTotal) :
    μ (RiskFailure p V εTarget) ≤
      βTotal := by
  exact
    riskFailure_measure_le_total_budget
      μ
      U
      p
      V
      S
      nBar
      β
      εTarget
      βDRO
      βTotal
      hpProb
      hRobust
      hAmbiguity
      hScenario.failureBound
      hBudget

end DROSafety
