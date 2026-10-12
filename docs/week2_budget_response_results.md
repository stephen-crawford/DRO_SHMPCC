# Week 2 budget-response results

The reduced-budget screen supports further empirical investigation. It does
not establish the complete numerical controller's support theorem.

- 130 frozen pre-sampling contexts; 2,080 actual budget solves.
- Production S(4)=666 and S(10)=1351: 50.70% fewer scenarios.
- h=4 implementation certificate acceptance: 253/260 = 97.31%.
- Hindsight oracle accounted sample reduction: 73.82%, including two unsolved
  blocks charged baseline cost. This excludes oracle search cost.
- Eight longer baseline runs: five path completions, two S-curve rejections,
  one step limit; zero recorded plant collisions. Six straight encounters pass
  their obstacles over 8.75–12 seconds.
- All 130 live-RNG and baseline-replay checks passed. No learner was trained.

[Full report, inputs, commands and limitations](../results/week2-budget-response-v2/README.md)

[Scenario-count curve](../results/week2-budget-response-v2/analysis/01_scenario_budget_curve.png)

[Actual acceptance curve](../results/week2-budget-response-v2/analysis/02_acceptance_curve.png)

[Oracle distribution](../results/week2-budget-response-v2/analysis/03_oracle_distribution.png)

[Oracle savings](../results/week2-budget-response-v2/analysis/04_oracle_savings.png)

[Runtime versus scenario count](../results/week2-budget-response-v2/analysis/05_runtime_by_budget.png)

[Support/certificate audit](week2_support_certificate_audit.md): the earlier-iterate
return matches the paper's intended prefix logging, but sample-dependent anchor
preparation and numerical feasibility/activity assumptions still need a proof.
The three existing regression discrepancies remain failures; no tests were weakened.

No mathematical guarantee or formulation was modified. Candidate experiment
implemented and tests executed; behavior requires user verification.
