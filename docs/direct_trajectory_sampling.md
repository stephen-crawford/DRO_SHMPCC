# Direct trajectory sampling candidate

Implemented on `sh_mpc_base` at the user's explicit request to match the local
`scenario_module` and `mpc_planner` sampling implementation. Runtime behavior
requires user verification. Existing uncommitted controller work was preserved.

## Selection and inputs

The default is `multimodal_bilevel: false`. The controller samples supplied
Gaussian trajectory predictions, without inferring categorical weights from
mode histories or computing DRO weights. Direct predictions can themselves be
multimodal: each obstacle supplies an ordered vector of `GaussianPredictionMode`
objects with a unique mode ID, probability, and exactly N prediction steps.
Probabilities must be finite, nonnegative, and sum to one (checked within 1e-12).
Each step supplies position, angle, major_radius and minor_radius, matching the
fields consumed by the reference sampler. Both radii must be positive; both
zero are additionally supported for deterministic fixtures.

Call `set_obstacle_predictions(predictions)` before each solve. Predictions are
consumed once. If none are supplied, the controller constructs a single
constant-velocity path from the current measured state. Its mean at input index k
is position + (k+1)*dt*velocity and its isotropic noise standard deviation is
`trajectory_noise_stddev` (default 1.0). The harness supplies this prediction each
decision, setting noise to zero when `obstacle_prediction_noise: false`.
This fallback is a local predictor adapter, not a reimplementation of
`mpc_planner::getConstantVelocityPrediction`, whose input means start at index 0
and whose separate uncertainty propagation is outside the sampler.

`multimodal_bilevel: true` selects the original `sample_scenarios` path, including
nominal history-based sampling, optional DRO allocation from p_hat to q*, and
optional Markov switching. `dro_enabled` still independently enables the DRO
allocator. DRO, Markov, hybrid fallback, and modal deterministic-anchor settings
require the explicit legacy sampler selection; incompatible combinations fail
validation rather than silently ignoring settings. Existing modal profiles and
experiment drivers opt in. No original sampler arithmetic was changed.

## Source mapping

The target is
[`Sampler::IntegrateAndTranslateToMeanAndVariance`](../../scenario_module/src/sampler.cpp),
called by
[`ScenarioConstraints::onDataReceived`](../../mpc_planner/mpc_planner_modules/src/scenario_constraints.cpp)
when Safe-Horizon is enabled. The separate `TranslateToMeanAndVariance` method
and partition/database-file loading paths are not the target of this change.

For each obstacle and joint scenario:

1. Select one supplied mode using the supplied cumulative probabilities and
   hold it for the whole path, as in the reference.
2. For each input stage k, form
   Sigma[k] = R(-angle[k]) diag(major_radius[k]^2, minor_radius[k]^2) R(-angle[k])^T,
   using the **passive** RosTools rotation convention, then L[k] = chol(Sigma[k]).
3. Accumulate noise e[k] = e[k-1] + L[k] z[batch_pick[k]][shuffle[k][s]].
4. Emit supplied_mean[k] + dt * e[k]. Supplied means are used literally; noise
   radii are not pre-propagated or scaled again by the sampler.

The Gaussian bank contains `trajectory_sample_batches` batches, each with S
bivariate standard normals. It persists across calls and is rebuilt only when
S or the batch count changes, or after reset. Each stage selects a batch and a
permutation of its indices. The same stage selections/permutations are shared
across obstacles, matching the reference, rather than inventing independently
drawn obstacle noise. The Safe-Horizon bank is neither pruned nor truncated.

Reference buffer index k maps to our output trajectory index k+1. Our index 0 is
the measured obstacle position with zero covariance. This accommodates the
existing N-transition/N+1-state controller without changing its constraint
indexing. The nominal covariance metadata is dt^2 * sum_{j<=k} Sigma[j]; it is not
the full joint covariance induced by finite-bank reuse. Collision constraints
use the sampled positions.

The bank uses `std::normal_distribution` rather than RosTools' uniform-to-Gaussian
transform. The reference shuffles using a fresh `random_device` seed; this
implementation seeds that separate generator from the controller RNG so runs
can be replayed. Consequently the same numeric seed alone does **not** promise
identical full-reference traces. The source comparison supplies identical banks
and pairs shuffle seeds to compare the actual translation/integration operation.

## Mathematical scope

This change affects mathematical behavior and was made only after explicit
approval: the user requested this sampling-law change and specifically required
the reference implementation. Existing bounds, sample-size formulas, risk
targets, constraints, support accounting, and solver tolerances were not changed.

The direct sampler intentionally preserves finite-bank reuse and its dependence,
including shared noise across obstacles. It must not be described as the original
fresh-IID sampler or as establishing a new scenario-probability guarantee.
Existing certificate fields evaluate the existing numerical conditions under
their model/data assumptions; numerical/source agreement here does not prove
those probabilistic assumptions. Pilot batches use a separate sampler/bank and
seed stream; augmentation uses the selected sampler and fresh joint IDs.

## Reproduction

```bash
cmake -S . -B /tmp/dro-direct-sampling \
  -DACADOS_ROOT=/home/stephen/acados -DCMAKE_BUILD_TYPE=Release \
  -DDRO_MPC_ENABLE_ROS2=OFF \
  -DDRO_MPC_REFERENCE_SAMPLER_SOURCE=/home/stephen/ros2_ws/src/scenario_module/src/sampler.cpp
cmake --build /tmp/dro-direct-sampling -j 2
ctest --test-dir /tmp/dro-direct-sampling \
  -R '^(test_direct_trajectory_sampler|test_scenario_sampler|test_reference_trajectory_sampling)$' \
  --output-on-failure
python3 tests/compare_reference_sampling.py
```

The optional reference comparison reads and executes the actual local reference
method, with stubs for ROS/config access. Its only sampling-body substitution
replaces the entropy seed with the paired replay seed. It compares two calls,
N=5, S=127, two obstacles, two modes, three batches, dt=0.17, varying means and
rotated anisotropic noise. All 2,540 future positions matched with maximum error
0; batch/mode RNG states also matched after both calls.

The integration test checks measured-stage indexing, exact zero-noise means,
Gaussian-bank permutations/persistence, accumulated perturbations, nominal
covariance metadata, shared obstacle noise, supplied multimodal probabilities,
input rejection, controller dispatch, one-shot predictions, direct diagnostics,
strict YAML, and opt-in legacy held-mode/DRO/Markov paths. With seed 77, the
0.75-probability alternate mode appeared 7,556 times out of 10,000.

Broader regression results and any outstanding limitations are recorded in
`results/direct-trajectory-sampling/README.md` after validation.
