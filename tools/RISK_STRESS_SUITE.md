# Paired dangerous-event evidence and adverse-switch suite

This extension is analysis plus a separate experimental fixture. It does not
change production control, risk allocation, sampling, or certification. The
suite uses the original five arms, without the stratified arm.

## Build and test

Run from the repository root:

```bash
source /opt/ros/jazzy/setup.bash
cmake -S . -B build-base
cmake --build build-base --target risk_stress_experiment -j 2
python3 -m unittest discover -s tests -p test_dangerous_events.py
python3 -m unittest discover -s tests -p test_risk_sampling.py
```

The CMake commands reuse the existing acados configuration in `build-base`.
The executable-dependent Python test skips if the executable is not built.
Matplotlib is required for `--plots` and the figure regression test.

## Existing experiments: new measurements without rerunning controllers

```bash
python3 tools/scrub_artifacts.py results/causal-matrix-visuals --out results/causal-csv-updated
python3 tools/analyze_risk_sampling.py results/causal-csv-updated --out results/dangerous-event-report --plots
```

The analyzer adds:

- `mode_mechanism_enriched.csv`: original mode risk, realized mode, nominal and
  sampling probabilities and counts, plus `map_mode`, `map_modes`, `map_tie`,
  and `switch_event`. Nominal risk stays blank when it was not measured; it is
  not imputed from WDRO. A tied MAP has an empty singular MAP and a JSON list of
  maximizers. A missing preceding step (including t=0) gives a blank switch flag.
- `dangerous_event_pairs.csv`: **one row per matched seed, repeat, decision and
  obstacle**, containing true mode, its risk/p/q*, nominal and WDRO counts,
  both decision outcomes, q−p, nD−nN, and successD−successN. All matched rows are
  retained with a `dangerous_event` flag so the event denominator is inspectable.
- `dangerous_event_pairs.log`: the same measurements as JSON lines, including
  provenance. The analyzer prints the file location and event counts.
- `figures/dangerous_chain_*.png` and `.pdf`: three panels showing realized-mode
  risk → mass shift → sample-count change → decision-outcome change. Each
  experimental condition/repeat has its own figure; `figures.json` maps names.
  Blank panels explicitly say there are no eligible observations.

Rows require runner-checked plant/belief pairing and equal nominal/WDRO budgets.
Missing decision outcomes remain `missing`, not failures or successes. Outcome
here means **decision admissibility**, not a collision-free action certificate.
Rollout collision/completion stay separate in `event_rollout_outcomes.csv`.
Repeated observations are descriptive, not independent experimental samples.
Existing logs require `report_all_comparisons.csv` from a completed matrix run;
without it, matched tables remain empty. Existing logs do not gain unmeasured
risk vectors, physical reference scores, or future-mode labels through analysis.

By default, dangerous means the realized mode is the unique positive maximum
of the logged WDRO pre-reweighting risk vector. These scores depend on the WDRO
plan. `--risk-threshold VALUE` substitutes a preregistered cutoff. Figures show
associations; arrows do not establish causality or imply a favorable result.

## New stress-suite commands

Start with a named condition before spending compute on the full suite:

```bash
python3 tests/run_risk_stress.py --output results/risk-stress-v1 --generate-only
python3 tests/run_risk_stress.py --output results/risk-stress-v1 --case rare_turn_p50_s20_switch15 --resume --plots
python3 tests/run_risk_stress.py --output results/risk-stress-v1 --case rare_braking_p50_s20_switch15 --resume --plots
```

Then run all scheduled conditions:

```bash
python3 tests/run_risk_stress.py --output results/risk-stress-v1 --resume
```

The default schedule is **396 conditions × 10 seeds × 5 arms = 19,800 executions**.
Runs are serial. The manifest hashes the executable, fixture, runner and base
configuration. Changed identities require a new output directory. Completed
ERROR records are retained on resume, not retried until favorable. Timeout and
execution errors produce explicit ERROR rows and a nonzero suite exit status;
reports are still written. `schedule.csv` retains the complete predetermined
inventory, even when `--case` selects a pilot. Reports include prior completed
conditions when further cases are run into the same root.

The wrapper produces normalized CSVs directly in `csv/` and analysis in
`report/`; no separate scrub is required. Every arm also has `run.log`,
`fixture.csv`, `plant.csv`, `mode_mechanism.csv`, `decisions.csv`, `plans.csv`, and
`summary.csv` under `condition/seed_N/arm/`.

## Prespecified experiment design

`configs/risk_stress/settings.json` defines:

| Family | Purpose |
|---|---|
| rare_turn | candidate rare across-path interaction, second obstacle below ego's route |
| rare_braking | nearer crossing obstacle and narrower lane to test braking |
| late_switch | same turn fixture, observations delayed five steps |
| two_sided_trap | second obstacle closer to the lower escape direction |
| rare_benign | rare mode turns away from ego's route |
| common_dangerous | crossing mode has about 30% nominal probability |
| equal_probability | equal prior counts, distinct trajectory geometry |
| equal_geometry | all modes have identical motion geometry |

The first five families sweep dangerous history counts 20/50/80/100/150/200
out of 1000, with 100 away-mode observations and the remainder continue-mode.
Production Dirichlet smoothing remains active: these are approximate
probabilities .02/.05/.08/.10/.15/.20; use logged p, not the requested count ratio.
Equal probability uses 333 observations per mode. The common control uses 300
across observations; equal geometry uses 50.

Budgets are 20, 40, and 10; extra nominal uses 2S. Seeds are 77–86. Nominal retry
and WDRO fallback retain their original recovery behavior. Safe Horizon and
automatic sample sizing are disabled for every arm; no certificate is requested.

The fixture is a **stylized intersection interaction**, with a straight ego
route crossing prescribed piecewise-linear obstacle paths. It is not a new
production intersection map or a vehicle turning-dynamics model. Obstacle A
starts at (5,2); continue advances (+.12,0), away (+.08,+.14), and across
(−.04,−.18) per .1-second step. B is stationary at (5,−1.4), or (4,−1.1) in the
trap. Braking starts A at (4,1.8) and sets road width to 2 m. Equal-probability's
middle mode uses (+.04,−.08), an intermediate crossing path despite retaining the
shared `away` identifier. These geometries are hypotheses to diagnose, not proof
that only braking/opposite homotopy is feasible.

The plant follows continue before the prescribed switch and across afterward,
identically for every arm. Switch steps 5/10/15/20 correspond to 0.5/1/1.5/2 s,
or offsets −1/−.5/0/+.5 s about a **reference** commitment time of 1.5 s.
Actual ego commitment is separately logged when x first reaches 3 m; it need
not occur at the reference time. Observations arrive after the physical step,
with the configured additional lag. There is no future-mode leakage.

Initial ego state is (0,0,0,2), dt=.1 s, horizon=20, rollout limit=80 steps,
reference route from (0,0) to (16,0), and completion at x≥15.2. The existing
three-disc, 2 m ego model is used. State integration updates path progress.
Physical clearance subtracts ego and obstacle radii; collision is checked at
sampled physical states, not asserted continuously between them. A rejected
control ends that arm; the fixture never executes an inadmissible plan.

## Independent geometry labels and anticipatory action

The stress fixture evaluates every mode against the same prescribed straight
reference ego trajectory x_ref(t)=2t, independently of either controller's plan.
It records minimum predicted safety clearance over the horizon and
`reference_risk=max(0,-reference_clearance)` in metres. The stress runner uses
`event_risk_reference=fixed`: a dangerous realized mode has positive reference
risk. Both arms must agree on the recorded reference risk. This reference is
not a physical guarantee and does not estimate collision probability.

To regenerate stress analysis manually:

```bash
python3 tools/analyze_risk_sampling.py results/risk-stress-v1/csv --out results/risk-stress-v1/reanalysis --event-risk-reference fixed --plots
```

`dangerous_event_pairs.csv` keeps both the WDRO pre-reweighting score (`risk_score`)
and the independent physical reference score (`reference_risk`); the former is
plotted against mass shift, while the latter defines the stress-event subset.
For fixed-reference labeling, an optional `--risk-threshold` is in metres.

`report/dangerous_mode_timeseries.csv` tracks the candidate across mode even
BEFORE it becomes the realized mode: p, q, r, n, speed, acceleration, angular
rate, clearance, and decision success. This is the table for anticipatory braking.
With `--plots`, `report/timelines/` shows mode allocation, risk, ego speed and
clearance with a dashed switch-time marker. Acceleration remains in the CSV.
`stress_outcomes.csv` retains first braking (a<−.1), actual commitment, termination,
switch reached, completion, sampled-state collision and solve time. Negative
braking/commitment indices mean the event was never observed.
`stress_summary.csv` retains OK/error/missing denominators by condition and arm.

These are **prescribed adverse-switch stress tests**, not collision-rate estimates
under the nominal population. Do not select only winning seeds or omit cases
that stop before the switch. Neither q≈.4 nor a WDRO outcome advantage is forced.

## Validation observed

The paired-report and stress-design tests passed, including MAP ties, unknown
switch flags, positive/negative/missing paired outcomes, equal-budget checks,
fixed versus WDRO event labels, deterministic delayed plant histories, all eight
families and negative controls, and PDF/PNG generation.

A three-step, switch-at-step-1 smoke run exercised all eight families and five
arms: 40 executions completed, 48 paired obstacle-decisions, 12 fixed-reference
dangerous events, eight chain figures and eight timelines. This validates data
plumbing, not performance.

Longer 40-step pilots at switch steps 5 and 15 exposed four turn-across execution
errors (empty free-space polygons); many other runs stopped before their switch.
Those records remain visible and are not converted to collision-free successes.
Updating the fixture's path-progress bookkeeping did not eliminate those errors.
These pilots do NOT demonstrate the desired outcome advantage; the geometry and
controller failure diagnostics need scientific review before paper conclusions.

Matplotlib reported an unavailable Axes3D extension in this environment; the
requested two-dimensional PDF/PNG figures were generated successfully.


## Longer-pilot rejection handling (approved follow-up)

The empty-polygon failure now uses `EmptyFreeSpacePolygon`, a subtype of the
existing runtime exception. `MPCController::solve` catches only this type, only
when Safe Horizon is disabled, and records an unsuccessful attempt with no
executable control. Existing nominal-retry arms then follow their normal retry
policy. Certified controllers rethrow; unrelated exceptions still propagate.
No half-space, radius, risk score, sample law, tolerance, or scene was changed.
This solver failure/retry change was made with explicit user approval.

Reproduction: rare_turn, dangerous count 50, S=20, seed 77, switch step 5,
40-step limit. Before: four ERROR exits at decision 16. After: all five arms
return normal outcome records; the four affected arms stop with
`no_admissible_control` after 17 recorded decisions. Both retry arms record the
nominal attempt; none invents an executable control. Extra nominal retains its
19-decision outcome. Pre-rejection decision and mode records match exactly,
excluding decision wall-clock timing. This does not establish feasibility or
completion: these runs remain stopped controller outcomes.

All 20 longer pilots (turn/braking × switch 5/15 × five arms, seed 77) now
complete the experiment process with no execution errors. Reports include
82 paired rows and 12 fixed-reference dangerous events.

```bash
cmake --build build-base --target risk_stress_experiment experiment_runner -j 2
python3 -m unittest discover -s tests -p test_long_stress_pilots.py
python3 tests/run_risk_stress.py --output results/risk-stress-v2 --case rare_turn_p50_s20_switch5 --plots
```

Use a new output directory because the executable identity changed. The broader
constraint test still fails its existing assertion that exceeding the facet
limit must throw. That failure was reproduced with the pre-change reducer;
neither the assertion nor the facet-limit behavior was modified here.
