# WDRO reweighting rationale: interim evidence audit

Date: 2026-09-24. Branch: `refactor`.

**Current conclusion:** the frozen experiment supports improved representation
of one geometrically dangerous rare mode at equal sampling budget. The audited
closed-loop evidence does not yet support a clear WDRO advantage beyond nominal
sampling or nominal retry. The full five-arm study has been resumed at the
user's explicit request and is still running. These are interim findings.

This task follows the newly attached rare-mode/five-controller plan. It does
not resume the separate covariance-bound/transfer plan from the IDE handoff.
No covariance treatment or protected controller change was made.

## A. Does WDRO change finite-sample representation?

Re-read `build-rare-mode/rare-pilot/{weights,coverage_trials}.csv`. Synthetic
history counts are straight=900, away=90, cut-in=10. The actual production
posterior and allocated probabilities for cut-in are:

| Quantity | Value |
| --- | ---: |
| Nominal probability p | 0.010484273589615577 |
| WDRO probability q | 0.029883644073199624 |
| Logged cut-in risk score | 0.95000314398028707 |
| Fixed-reference safety margin | -0.94999999999999973 m |
| Logged radius rho | 0.04073867801552649 |

The negative margin establishes why this mode is dangerous to this reference.
Its name alone would not establish that. The allocated mass increases from
about 1.05% to 2.99%; both other modes have zero logged risk in this fixture.
This demonstrates the direction of the risk-weighted allocation. These logs
alone are not an independent proof of transport feasibility or LP optimality.

| Total S | Nominal included / 500 | WDRO included / 500 |
| --- | ---: | ---: |
| 16 | 68 (13.6%) | 186 (37.2%) |
| 40 | 164 (32.8%) | 350 (70.0%) |
| 80 | 286 (57.2%) | 458 (91.6%) |
| 160 | 408 (81.6%) | 494 (98.8%) |

![Frozen inclusion curves](results/wdro-rationale-audit-20260924/frozen_inclusion.png)

For independent fixed-law draws, inclusion is `1-(1-p)^S`. WDRO at S=40 has
expected inclusion 0.7028655070899879. Nominal at S=115 gives
0.702415300176063, just below that target; S=116 gives 0.7055352595851008.
Thus **116 nominal draws are needed to match the expectation of 40 WDRO draws
in this fixture**. This is a calculated inclusion threshold, not an observed
116-draw experiment, a Safe-Horizon sample count, or a closed-loop safety result.
The attachment's S=80 nominal example was hypothetical; these data do not
support that particular equivalence.

Nominal single and unconditional split draws have identical counts in the
existing artifact test. Do not apply the unconditional inclusion formula to
conditional controller retries. Frozen controller records are single-decision
experiments and do not substitute for closed-loop evidence.

## B. Robustness under controlled plant shift

The large study is `results/mismatch-concentration-v3`. Its frozen design has
3,600 controller/seed trials, each executed twice (7,200 executions), across
five arms, S=20/40/80, ten seeds, two environments, three obstacle/class
combinations, and override probabilities 0/.05/.15/.30. Repeats are numerical
checks and are not additional independent observations.

At initial inspection: 1,466 trials recorded OK, nine recorded ERROR, and
2,125 had no result. No matrix process was running. Frozen executable and
configuration hashes passed the existing resume preflight.

The separate audit snapshot, taken shortly after resume began, contains
1,468 OK and eight ERROR records. Every saved successful repeat was re-read
and its numerical signature recomputed: **1,468 trials passed; zero integrity
errors**. The audit admits only seed/setup groups with all five arms successful
and shared-prefix plant/nominal-belief pairing checks passing. There are
**284 such groups**, or 1,420 matched controller trials. Unmatched successful
trials and errors remain in the saved input snapshot.

| Arm | Policy | Collision-free completions / 284 | Recorded collisions |
| --- | --- | ---: | ---: |
| A | Nominal S | 208 | 0 |
| B | Nominal 2S | 222 | 0 |
| C | WDRO S | 198 | 0 |
| D | Nominal S, then nominal S on rejection | 254 | 0 |
| E | WDRO S, then nominal S on rejection | 255 | 0 |

These totals are descriptive across dependent setups: 176 matched groups at
override 0 and 108 at override .05. They omit the unexecuted .15/.30 profiles
and exclude incomplete groups. They are not a balanced full-matrix estimate
or an inferential claim. The per-setup/budget denominators and metrics are in
`five_arm_matched.csv`; individual seed rows are in `matched_rollouts.csv`.

There is no observed collision-rate separation here. A/C completion discordance
is 36 groups where only nominal completes versus 26 where only WDRO completes.
The selected population therefore does not show a standalone WDRO completion
advantage. Zero collisions does not establish safety, especially with early
termination and exclusion of execution errors.

The configured override is not measured distributional divergence: the
controller observes plant modes and adapts its history. The focus mode
`turn_right` is not necessarily dangerous in every geometry. In these interim
groups, its inclusion is already about 99.9% under nominal sampling and is
lower under WDRO (about 94.7% at override 0; 96.8% at .05). Do not describe
that as improved rare-mode coverage in this matrix. A final mechanism claim
must inspect actual risk scores and underrepresented modes at each decision.

The separate completed `results/fixed-budget-uncertified-large` dataset has
240 nominal/WDRO pairs, with collisions 6 versus 5 and safe completions 192
versus 191. It uses horizon 8 and disables Safe Horizon, selecting a different
existing constraint/recovery path. It has no retry or extra-sampling arms.
It neither establishes a broad benefit nor answers five-arm attribution.

## C. Is the benefit WDRO or retrying?

Within the 284 matched groups, D/E completion discordance is 22 groups where
only nominal retry completes versus 23 where only WDRO/nominal completes;
232 complete under both and seven under neither. The one-completion difference
does not establish an advantage from WDRO inside the hybrid.

The present evidence is consistent with much of the observed hybrid completion
benefit coming from retrying. This is an interim interpretation, not a final
causal conclusion. Complete all predetermined shift levels and report D/E
alongside A/B and A/C before replacing the paper's empirical headline.

## D. Computational and sample efficiency

`five_arm_matched.csv` reports each fixed setup/budget's actual draws per cycle,
attempts per cycle, retry fraction, first-attempt admissibility, complete-cycle
latency, and mean rollout minimum safety margin. `matched_rollouts.csv` retains
total draws, attempts, decisions, and cycle time per rollout.

Equal S does not mean equal CPU cost. B uses 2S on its sole attempt; D/E spend
their second batch conditionally. Latencies include diagnostic overhead and
historical machine load. Current resumption also changes execution date/load;
the audit itself overlaps a small part of the resumed run. Do not infer a
controlled timing advantage from these historical timings. Numerical repeat
checks intentionally exclude wall-clock durations. Do not pool S=20/40/80
draw averages and call that an equal-budget comparison.

The defensible sample-efficiency claim currently concerns frozen inclusion
(116 versus 40 expected-equivalent draws). Similar closed-loop safety or
completion at lower S has not been established.

## Run state and next analysis

The user explicitly requested resuming the entire interrupted study. Executed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/resume_matrix.py results/mismatch-concentration-v3 \
  > results/wdro-rationale-audit-20260924/resume.log 2>&1
```

The resumer reuses compatible successful trials, archives failed/partial
trials in `resume_history`, and retains new failures. Eight recovery-construction
errors reproduced on rerun. The missing-artifact trial subsequently completed.
No solver or recovery changes were made to turn these errors into successes.
The remaining predetermined cases continue serially. Do not start a duplicate
process or rebuild the frozen executable. Watch `resume.log` for progress.

When the process finishes, its standard reports include all comparisons,
per-setup paired outcomes, mechanism metrics and concentration diagnostics.
Check errors and denominators before interpretation. In particular, inspect
the .15/.30 shift levels, A/B, A/C, B/C and D/E, along with mass-decreased modes.
Do not overwrite this interim snapshot when preparing the final audit.

Suggested present-tense paper statement:

> In a frozen rare-mode experiment, WDRO increased dangerous-mode inclusion
> from 32.8% to 70.0% at 40 scenarios. Under the logged fixed sampling laws,
> nominal sampling requires 116 draws to match WDRO's expected inclusion at
> 40 draws. A five-controller study separating reweighting, sampling budget,
> and retry effects remains in progress; current partial results do not
> establish an additional closed-loop advantage from reweighting.

## Change made

- File: this report and `results/wdro-rationale-audit-20260924/`.
- Section: evidence audit, interim CSV/JSON tables and frozen inclusion figure.
- Exact behavior: read existing artifacts, recheck saved repeat signatures and
  pairing, and write separate evidence outputs; resume the existing frozen study.
- Minimality: reused existing validation/pairing and resume implementations.
  No production code, settings, existing tests or controller formulas edited.

## Mathematical impact

No mathematical guarantee or formulation was modified.

The fixed-law inclusion calculation is the one requested in the attachment.
It is kept separate from scenario certification and conditional retry logic.
No covariance symmetrization, sample-count reduction or acceptance change.

## Validation performed

- Build: not run; no C++ changed and rebuilding would invalidate the frozen
  experiment executable. Resume preflight matched its saved hash and configs.
- Test command: `PYTHONDONTWRITEBYTECODE=1 python3 tests/test_sample_efficiency.py
  --rare-artifacts build-rare-mode/rare-pilot
  --uncertified-artifacts results/fixed-budget-uncertified-large`.
- Output: five unit tests passed; analytic inclusion/split-budget checks and
  96 frozen controller records passed; 480 repeatable two-arm trials,
  240 comparisons, and 101,540 decisions/attempts passed exact 40-scenario
  and zero-certification-request checks.
- Audit command: `PYTHONDONTWRITEBYTECODE=1 python3
  results/wdro-rationale-audit-20260924/audit.py`.
- Output: 1,468 successful trials passed numerical artifact signatures;
  284 complete five-arm groups. Raw status and pair checks retained separately.
- Warnings/failures: eight reproduced recovery-construction failures remain;
  the original missing-artifact failure is preserved in resume history.
  Matplotlib warned that its optional 3D projection was unavailable; the
  requested 2D figure was generated.

## Current status

Candidate fix implemented and test executed; behavior requires user verification.

Here this required repository status refers to the new audit artifacts, not
to a controller fix. The overall research task remains in progress while the
full study runs. Neither research success nor user verification is claimed.
