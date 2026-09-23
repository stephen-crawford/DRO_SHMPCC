# Resuming saved experiment matrices

The standalone resumer uses the saved `matrix.json` and generated YAML files,
not today's settings defaults or configuration generator. It supports analysis,
reviewer and comparison matrices, including five-arm and uncertified suites.
No controller or mathematical formulation is changed.

Check compatibility without writing or launching anything:

```bash
python3 tests/resume_matrix.py results/mismatch-concentration-v3 --dry-run
python3 tests/resume_matrix.py results/fixed-budget-uncertified-large --dry-run
```

After the existing runner has stopped, resume either suite:

```bash
python3 tests/resume_matrix.py results/mismatch-concentration-v3
python3 tests/resume_matrix.py results/fixed-budget-uncertified-large
```

Do not run an original matrix command and this command simultaneously on one
directory. The resumer locks against other resumers, but older runners do not
honor that lock. `--limit 1` executes at most one incomplete seed trial; repeat
the command without the limit to continue the entire remaining set.

Successful trials retain their original result files and are checked against
their saved repeat signatures before reuse. Missing or ERROR trials rerun all
configured repeats. Previous failed/partial directories move to
`resume_history/<UTC timestamp>/...`; they are never silently deleted. The log
scrubber excludes this archive from active-run discovery. Reports are rebuilt
from the complete current trial population, including errors and pending trials.
If execution is interrupted, rerun this command; each completed trial already
has its own result record.

The original manifest, configuration hashes, seeds, repeats, timeout and
executable hash remain fixed. Changed Python orchestration/reporting code is
recorded in `resume_history/<timestamp>/session.json` under the original
experiment identity. Current script hash differences alone do not discard
completed simulations. Only serial saved matrices are supported. Timing is
excluded from numeric equality, as in the original runner.

## Older incompatible runs

The audit found:

- `build-base/analysis-matrix` and `build-base/comparison-matrix`: the recorded
  executable hash no longer matches the available executable.
- `build-base/reviewer-full-design`: the old executable rejects the newer
  `artifact_capture_attempt_diagnostics` default YAML key. Generated configs
  do not freeze that newly introduced key.

Those cannot safely combine old successes with simulations from a different
binary/default environment. Restore the original compatible environment to
continue in place, or start a distinct cohort using the saved settings:

```bash
python3 tests/resume_matrix.py build-base/analysis-matrix \
  --restart-to results/analysis-restarted --runner build-concentration/experiment_runner
python3 tests/resume_matrix.py build-base/comparison-matrix \
  --restart-to results/comparison-restarted --runner build-concentration/experiment_runner
python3 tests/resume_matrix.py build-base/reviewer-full-design \
  --restart-to results/reviewer-restarted --runner build-concentration/experiment_runner
```

A restart reruns **all** trials in a new directory, preserves the old directory,
and writes `restart_origin.json` stating that no old successes were reused.
It uses the current suite generator and new executable, so it is explicitly a
new experiment identity. To resume that new cohort later:

```bash
python3 tests/resume_matrix.py results/reviewer-restarted
```

Resuming retries execution; it does not repair homotopy/constraint failures.
Repeated solver errors remain ERROR and cause a nonzero exit. `--dry-run` checks
identity/configuration compatibility and counts records; actual execution also
validates saved successful trial evidence.

## Validation

Executed `python3 tests/test_resume_matrix.py` against the actual
`build-concentration/experiment_runner`. A small two-arm, two-repeat matrix was
run; one result was marked ERROR and resumed, then its result was removed to
simulate interruption and resumed again. The successful sibling and original
manifest stayed byte-identical; prior files were archived. Changed YAML and a
different binary were rejected. A separate restart executed both arms again
without reusing old successes. The test passed.

Live suites were audited with `--dry-run` only; no second runner was launched.
No C++ build was required. Candidate changes implemented and tested; behavior
requires user verification.
