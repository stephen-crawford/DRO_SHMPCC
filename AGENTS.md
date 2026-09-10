# AGENTS.md

## Purpose

This file defines mandatory instructions for any coding agent working in this repository.

These rules are especially strict around:

* mathematical guarantees,
* safety or correctness guarantees,
* probabilistic guarantees,
* optimization formulations,
* MPC constraints,
* proofs or proof-derived logic,
* solver assumptions,
* confidence bounds,
* robustness guarantees,
* support estimation,
* chance constraints,
* ambiguity sets,
* sample complexity,
* and any code whose correctness depends on a mathematical derivation.

The primary goal is to make changes that are:

1. minimal,
2. reversible,
3. mathematically faithful,
4. easy to review,
5. explicitly testable,
6. and never presented as correct before the user has verified the resulting behavior.

---

# 1. Highest-Priority Rule: Do Not Modify Mathematical Guarantees Without Explicit Approval

## 1.1 Protected mathematical code

Any code that implements, enforces, approximates, derives from, or materially affects a mathematical guarantee is considered **protected code**.

Protected code includes, but is not limited to:

* theorem-derived calculations,
* probability bounds,
* confidence bounds,
* sample complexity formulas,
* robustness bounds,
* ambiguity-set construction,
* distributionally robust optimization logic,
* support-size calculations,
* scenario removal logic,
* chance constraints,
* feasibility guarantees,
* stability guarantees,
* safety constraints,
* collision-avoidance guarantees,
* MPC constraint formulations,
* terminal constraints,
* invariant-set calculations,
* worst-case optimization,
* `worst_case_*` functions,
* KL / TV / Wasserstein / divergence-based bounds,
* support estimation,
* risk allocation,
* epsilon / beta calculations,
* finite-sample guarantees,
* proof-derived constants,
* mathematical assumptions encoded in software,
* solver formulations whose modification could change a formal guarantee.

## 1.2 Explicit approval is required

**DO NOT modify protected mathematical code without explicit approval from the user.**

This prohibition includes changes that may appear to be:

* obvious bug fixes,
* numerical improvements,
* simplifications,
* performance optimizations,
* refactors,
* cleanup,
* defensive programming,
* edge-case handling,
* API modernization,
* or equivalent reformulations.

If a modification could alter the meaning, assumptions, domain, constants, inequalities, optimization problem, feasible set, probability statement, or formal guarantee, stop before making the modification.

Instead:

1. identify the exact mathematical issue,
2. identify the exact code involved,
3. explain why the current behavior may be incorrect,
4. describe the proposed modification,
5. explain how it affects the mathematical formulation,
6. and wait for explicit approval before changing it.

Do not interpret a general request such as "fix the bug" as permission to alter mathematical guarantees unless modifying that guarantee is clearly and explicitly approved.

---

# 2. Preserve Mathematical Meaning Exactly

When working around protected mathematical code, preserve the original formulation exactly unless explicitly authorized otherwise.

Do not silently change:

* `<` to `<=`,
* `>` to `>=`,
* strict versus non-strict constraints,
* norm definitions,
* probability conventions,
* logarithm bases,
* epsilon values,
* beta values,
* confidence levels,
* support-size definitions,
* sample counts,
* horizon indexing,
* scenario indexing,
* removal counts,
* dimensions,
* constants,
* units,
* scaling,
* optimization directions,
* max/min ordering,
* clipping behavior,
* asymptotic approximations,
* finite-sample formulas,
* or assumptions encoded by surrounding code.

Do not substitute an approximation for an exact expression unless explicitly approved.

Do not replace a theorem-derived expression with a numerically convenient alternative unless explicitly approved.

Do not assume two mathematically similar formulations are interchangeable.

---

# 3. Smallest Possible Change

Every individual change must be the **smallest change that directly addresses the core request or identified bug**.

Prefer:

* one-line fixes over function rewrites,
* local changes over architectural changes,
* changes in one file over changes across several files,
* existing abstractions over new abstractions,
* existing dependencies over new dependencies,
* existing interfaces over API changes.

Avoid:

* opportunistic refactoring,
* unrelated cleanup,
* renaming unrelated variables,
* formatting unrelated code,
* reorganizing files,
* changing public interfaces unnecessarily,
* modifying nearby code "while already there",
* adding new abstractions unless required,
* changing behavior outside the requested scope.

A bug fix should not become a refactor.

A refactor should not be performed unless explicitly requested.

---

# 4. Every Change Must Be Independently Reversible

Each logical modification must be easy to identify and revert.

Changes should be structured so that one individual change can be removed without requiring unrelated changes to also be undone.

Whenever practical:

* keep one logical fix localized,
* avoid mixing multiple fixes together,
* avoid combining formatting with behavioral changes,
* avoid changing unrelated code in the same edit,
* preserve the original structure,
* minimize dependency between separate edits.

If multiple changes are necessary, treat them as separate logical changes and clearly explain why each is required.

Do not create a chain of unnecessary modifications where reverting one fix requires reverting several unrelated edits.

---

# 5. Diagnose Before Editing

Do not modify code simply because something looks suspicious.

Before changing code:

1. locate the relevant implementation,
2. trace how it is called,
3. identify the inputs,
4. identify the outputs,
5. identify the assumptions,
6. determine the expected behavior,
7. reproduce or understand the observed failure,
8. identify the smallest plausible root cause.

When mathematical code is involved, also determine:

1. the mathematical expression the code intends to implement,
2. the corresponding variables and dimensions,
3. any relevant assumptions,
4. whether the implementation actually differs from the intended formulation.

Distinguish clearly between:

* an observed bug,
* a suspected bug,
* a numerical issue,
* a modeling choice,
* an implementation choice,
* and an actual violation of the mathematical formulation.

Do not present a suspicion as a confirmed root cause.

---

# 6. Do Not Claim an Issue Is Fixed

The words:

* "fixed",
* "resolved",
* "corrected",
* "working",
* "verified",
* "addresses the issue",
* "solves the bug",
* or equivalent language

must not be used as a final conclusion merely because code was modified.

A code change is only a **candidate fix** until validation is complete.

Use language such as:

* "I changed X to address the suspected cause."
* "This is the minimal candidate fix."
* "The code builds successfully, but behavior still needs verification."
* "The test produced the following result; please verify whether this matches the intended behavior."

---

# 7. Required Validation Process

An issue may only be considered successfully addressed after all of the following stages occur.

## Stage 1: Make the minimal change

Apply only the smallest change necessary to address the identified cause.

## Stage 2: Run an appropriate test

The agent must actually run a test that exercises the changed behavior.

Depending on the change, this may include:

* a unit test,
* an integration test,
* a regression test,
* a targeted executable,
* a reproducible simulation,
* a numerical comparison,
* a solver run,
* a build plus runtime test,
* or a minimal reproduction.

Merely compiling the code is not sufficient to establish behavioral correctness.

## Stage 3: Report the raw or relevant test result

Report enough information for the user to independently inspect the behavior.

Include, when relevant:

* command executed,
* important inputs,
* observed outputs,
* expected outputs,
* numerical values,
* errors or warnings,
* test pass/fail status,
* solver status,
* constraint values,
* residuals,
* trajectories,
* or comparison against prior behavior.

Do not hide unexpected results.

## Stage 4: User verification

The user must inspect the reported result and determine whether it demonstrates the intended behavior.

Until the user verifies the result, do not state that the issue is fully resolved.

The correct status before user verification is:

> Candidate fix implemented and tested; awaiting user verification of behavior.

---

# 8. Tests Must Exercise the Actual Bug

Do not use an unrelated passing test as evidence that a bug is addressed.

The validation test should exercise the actual code path associated with the request.

For a numerical or mathematical bug, the test should inspect relevant numerical behavior.

For example, depending on the issue, inspect:

* exact bound values,
* worst-case probabilities,
* objective values,
* feasibility,
* constraint satisfaction,
* support counts,
* confidence radii,
* selected scenarios,
* sample sizes,
* solver output,
* trajectory behavior,
* collision margins,
* or other relevant quantities.

A successful build proves only that the code builds.

A passing unrelated unit test proves only that the unrelated unit test passes.

Neither is sufficient evidence of behavioral correctness.

---

# 9. Never Fabricate Validation

Never claim:

* a test was run when it was not,
* a build succeeded when it was not executed,
* a simulation succeeded when it was not executed,
* output values that were not actually observed,
* user verification that the user did not provide.

If a test cannot be run, state exactly why.

For example:

> The code change has been made, but I could not validate runtime behavior because the required dependency/environment/test setup is unavailable.

That status must not be converted into "fixed."

---

# 10. Protect Existing Tests

Do not modify an existing test merely to make a code change pass unless:

1. the existing test is demonstrably incorrect,
2. the reason is explained,
3. and the user explicitly approves changing the expected behavior.

Never weaken:

* assertions,
* tolerances,
* safety checks,
* expected constraints,
* expected probabilities,
* expected guarantees,
* or regression conditions

just to obtain a passing test.

Do not delete a failing test unless explicitly instructed.

---

# 11. Mathematical Regression Tests

When changing non-protected code adjacent to mathematical logic, preserve or add targeted checks whenever practical.

Examples:

```text
computed_bound >= required_bound
```

```text
empirical_probability <= configured_epsilon
```

```text
num_removed <= configured_removal_limit
```

```text
sample_size == expected_sample_size_for_known_parameters
```

```text
constraint_margin >= 0
```

These examples are illustrative only.

Do not invent a mathematical invariant unless it follows from the intended formulation.

---

# 12. Build and Runtime Behavior

If the repository uses ROS 2:

```bash
source /opt/ros/jazzy/setup.bash
```

If the workspace has already been built:

```bash
source install/setup.bash
```

Typical build:

```bash
colcon build --symlink-install
```

Targeted package build:

```bash
colcon build --packages-select <package_name> --symlink-install
```

Prefer targeted builds and targeted tests when they adequately validate the change.

Avoid rebuilding or modifying unrelated packages unnecessarily.

---

# 13. C++ Guidelines

Follow the style already used in nearby code.

Prefer:

* const correctness,
* explicit dimensions,
* descriptive names,
* existing types and abstractions,
* local changes,
* clear comments where mathematical intent is non-obvious.

Avoid:

* unnecessary dependencies,
* broad header changes,
* global state,
* unexplained magic constants,
* silent type conversions,
* changing floating-point precision without justification,
* changing integer/floating-point semantics without justification.

Do not perform large style or modernization passes during bug fixes.

---

# 14. Numerical Code

Treat seemingly minor numerical changes as potentially behavioral.

Do not change without justification:

* epsilon thresholds,
* clipping,
* saturation,
* minimum denominators,
* regularization,
* tolerances,
* convergence thresholds,
* solver tolerances,
* iteration limits,
* floating-point types,
* initialization,
* fallback behavior,
* random seeds.

If one of these affects a mathematical guarantee, explicit user approval is required before changing it.

---

# 15. Solver and Optimization Code

Be especially conservative when modifying solver-facing code.

Do not change without careful analysis:

* variable ordering,
* parameter ordering,
* dimensions,
* horizon indexing,
* state indexing,
* input indexing,
* constraint indexing,
* objective sign,
* minimization/maximization direction,
* constraint bounds,
* solver parameter mappings.

Generated solver interfaces should be considered stable unless explicitly requested otherwise.

Any change that alters the mathematical optimization problem requires explicit user approval.

---

# 16. Comments and Documentation

Comments describing mathematical behavior must remain consistent with the actual implementation.

If implementation and documentation disagree:

* identify the disagreement,
* determine which one corresponds to the intended formulation,
* do not silently change protected mathematical behavior,
* request approval where required.

Do not alter an equation in documentation merely to make it match questionable code.

Do not alter code merely to make it match questionable documentation.

---

# 17. Required Change Summary

After making changes, report them in this form:

## Change made

* File:
* Function / section:
* Exact behavior changed:
* Why this was the smallest appropriate change:

## Mathematical impact

State one of:

* `No mathematical guarantee or formulation was modified.`

or:

* `This change affects mathematical behavior and was made only after explicit approval.`

Explain the effect precisely.

## Validation performed

* Build command:
* Test command:
* Relevant inputs:
* Relevant output:
* Any warnings or failures:

## Current status

Use one of:

* `Change implemented; test not yet successfully executed.`
* `Candidate fix implemented and test executed; behavior requires user verification.`
* `User verified the test result and confirmed the intended behavior.`

Do not use the final status unless the user actually confirms it.

---

# 18. Before Editing Checklist

Before editing any code, verify:

* [ ] I understand the requested behavior.
* [ ] I identified the relevant code path.
* [ ] I identified the smallest likely root cause.
* [ ] I checked whether the change touches a mathematical guarantee.
* [ ] If it does, I have explicit user approval.
* [ ] The proposed change is minimal.
* [ ] The proposed change is independently reversible.
* [ ] I am not including unrelated cleanup.

---

# 19. After Editing Checklist

Before presenting the result, verify:

* [ ] Only necessary files were modified.
* [ ] No unrelated formatting or refactoring was introduced.
* [ ] Mathematical formulations were preserved unless explicitly approved.
* [ ] The change can be reverted independently.
* [ ] The affected code was built where possible.
* [ ] A targeted behavioral test was actually run.
* [ ] The test exercises the reported issue.
* [ ] I reported the relevant test output.
* [ ] I have not claimed the issue is resolved before user verification.

---

# 20. Default Decision Rule

When uncertain whether a modification might alter a mathematical guarantee:

**Assume that it does.**

Do not make the modification.

Explain the concern and request explicit approval.

When uncertain whether a larger change would be useful:

**Do not make the larger change.**

Make the smallest change that directly addresses the request.

When uncertain whether a test proves the issue is resolved:

**Assume that it does not.**

Report the result and ask the user to verify the observed behavior.

---

# 21. Core Operating Principle

The default workflow for this repository is:

```text
Understand
    ↓
Locate
    ↓
Diagnose
    ↓
Determine whether mathematical guarantees are involved
    ↓
If yes → obtain explicit approval before modifying
    ↓
Make the smallest reversible change
    ↓
Build
    ↓
Run a targeted behavioral test
    ↓
Show the user the result
    ↓
Wait for user verification
    ↓
Only then consider the issue confirmed as resolved
```

The goal is not to make the most changes.

The goal is to make the **smallest defensible change, preserve mathematical guarantees, produce observable evidence, and let the user make the final correctness determination.**
