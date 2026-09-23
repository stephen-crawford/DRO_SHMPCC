#!/usr/bin/env python3

import argparse
import csv
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from scrub_artifacts import (LABELS, BUNDLE_TABLES, REPORT_TABLES, RARE_TABLES,
                             enrich_identity, artifact_tables, collect_reports)


KV_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)"
)

SEED_DIR_RE = re.compile(r"seed_(\d+)$")


def parse_kv(line):
    """Extract key=value tokens from one log line."""
    return {
        key: value.rstrip(",;")
        for key, value in KV_RE.findall(line)
    }


def to_float(value):
    if value is None:
        return None

    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def to_int(value):
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def maximum(values):
    values = [x for x in values if x is not None]
    return max(values) if values else None


def minimum(values):
    values = [x for x in values if x is not None]
    return min(values) if values else None


def fraction(num, den):
    if den == 0:
        return None
    return num / den

def rms(values):
    values = [
        x for x in values
        if x is not None
    ]

    if not values:
        return None

    return math.sqrt(
        sum(x * x for x in values)
        / len(values)
    )


def sum_sq(values):
    values = [
        x for x in values
        if x is not None
    ]

    return sum(
        x * x
        for x in values
    )


def mean_abs_delta(values):
    values = [
        x for x in values
        if x is not None
    ]

    if len(values) < 2:
        return None

    return statistics.mean(
        abs(values[i] - values[i - 1])
        for i in range(1, len(values))
    )


def max_abs_delta(values):
    values = [
        x for x in values
        if x is not None
    ]

    if len(values) < 2:
        return None

    return max(
        abs(values[i] - values[i - 1])
        for i in range(1, len(values))
    )


def locate_run(log_path, root):
    """
    Expected form:

        root/case/seed_77/run.log

    The case portion may itself contain directories.
    """

    rel = log_path.relative_to(root)
    parts = rel.parts

    seed_index = None
    seed = None

    for i, part in enumerate(parts):
        match = SEED_DIR_RE.fullmatch(part)
        if match:
            seed_index = i
            seed = int(match.group(1))
            break

    if seed_index is None:
        return None

    case_parts = list(parts[:seed_index])
    case = "/".join(case_parts)

    # Construct a canonical case for pairing DRO/non-DRO.
    #
    # Example:
    #
    # sh_mpcc_roundabout_o2_c2_m4
    # sh_mpcc_dro_roundabout_o2_c2_m4
    #
    # both become:
    #
    # roundabout_o2_c2_m4
    if case_parts:
        leaf = case_parts[-1]

        if leaf.startswith("sh_mpcc_resample_"):
            canonical_leaf = leaf[len("sh_mpcc_resample_"):]
        elif leaf.startswith("sh_mpcc_extra_"):
            canonical_leaf = leaf[len("sh_mpcc_extra_"):]
        elif leaf.startswith("sh_mpcc_dro_fallback_"):
            canonical_leaf = leaf[len("sh_mpcc_dro_fallback_"):]
        elif leaf.startswith("sh_mpcc_dro_"):
            canonical_leaf = leaf[len("sh_mpcc_dro_"):]
        elif leaf.startswith("sh_mpcc_"):
            canonical_leaf = leaf[len("sh_mpcc_"):]
        else:
            canonical_leaf = leaf

        canonical_parts = case_parts[:-1] + [canonical_leaf]
        pair_case = "/".join(canonical_parts)
    else:
        pair_case = case

    return enrich_identity(log_path, root, {
        "case": case,
        "pair_case": pair_case,
        "seed": seed,
        "log_file": str(rel),
        "run_id": str(rel.with_suffix("")),
    })


def parse_log(log_path, root):
    identity = locate_run(log_path, root)

    if identity is None:
        return None, [], [], [], []

    text = log_path.read_text(
        errors="replace"
    )

    lines = text.splitlines()

    # ---------------------------------------------------------
    # Run outcome / performance
    # ---------------------------------------------------------

    final_result = {}

    harness_steps = []
    harness_failures = []

    path_fractions = []
    path_progress = []
    path_lengths = []
    completion_steps = []

    # ---------------------------------------------------------
    # Safe Horizon / certification
    # ---------------------------------------------------------

    support_final_union = []
    support_final_active = []
    support_limits = []

    support_final_certified = 0
    support_final_exceeded = 0
    support_not_evaluated = 0

    sqp_support_sizes = []
    sqp_support_exceeded_iterations = 0

    # ---------------------------------------------------------
    # Recursive feasibility / recovery
    # ---------------------------------------------------------

    rf_accepted = 0
    rf_rejected = 0
    rf_removed = []
    rf_required_rejected = []

    homotopy_tries = 0
    homotopy_sides = Counter()

    brake_candidates = 0
    brake_feasible = 0

    # ---------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------

    free_poly_facets = []
    free_poly_raw = []
    free_poly_vertices = []
    anchor_shifts = []

    facet_warnings = 0
    facets_over_20 = 0

    anchor_clearances = []


    actual_clearances = []
    actual_center_distances = []
    actual_clearance_steps = []

    collision_radius_values = []

    collision_step_count = 0
    near_miss_0p10_count = 0
    near_miss_0p25_count = 0
    near_miss_0p50_count = 0


    # ---------------------------------------------------------
    # Executed control / controller cost
    # ---------------------------------------------------------

    control_steps = []

    applied_a = []
    applied_omega = []

    unweighted_control_costs = []
    weighted_control_costs = []

    mpc_costs = []

    step_solve_times = []
    step_qp_solve_times = []
    step_constraint_times = []
    step_dro_risk_times = []

    step_dt = []
    acceleration_weights = []
    steering_weights = []

    fallback_steps = 0
    nominal_fallback_attempts = 0
    nominal_fallback_successes = 0
    pending_nominal_fallback = None

    # ---------------------------------------------------------
    # SQP / QP
    # ---------------------------------------------------------

    qp_constraints = []

    qp_iterations = []
    qp_primal_residuals = []
    qp_dual_residuals = []

    qp_converged = 0
    qp_failed = 0

    sqp_final_violations = []
    sqp_final_violated_constraints = []

    # ---------------------------------------------------------
    # DRO
    # ---------------------------------------------------------

    dro_steps = []
    dro_modes = []
    dro_samples = []

    dro_rho = []
    dro_risk_lift = []
    dro_nominal_risk = []
    dro_qstar_risk = []
    dro_tv = []
    dro_transport_cost = []
    dro_lambda = []

    dro_budget_used = 0
    dro_summary_count = 0

    # ---------------------------------------------------------
    # Parse
    # ---------------------------------------------------------

    for line in lines:
        kv = parse_kv(line)

        # Final experiment result.
        if (
            "seed" in kv
            and "collision" in kv
            and "termination" in kv
        ):
            final_result = kv.copy()

        # --------------------
        # HARNESS
        # --------------------

        if "[HARNESS RESULT]" in line:
            step = to_int(kv.get("step"))
            if step is not None:
                harness_steps.append(step)

            success = to_int(kv.get("success"))

            if success == 0:
                harness_failures.append(step)
        
        # --------------------
        # EXECUTED CONTROL / COST
        # --------------------

        if "[NOMINAL FALLBACK]" in line:
            nominal_fallback_attempts += 1
            pending_nominal_fallback = to_int(kv.get("success"))
            if pending_nominal_fallback == 1:
                nominal_fallback_successes += 1

        if "[MPC STEP METRICS]" in line:
            row = dict(identity)
            row.update(kv)
            # Controller iteration numbers and harness steps use different
            # indexing. The fallback event precedes its control metrics.
            row["nominal_fallback_attempted"] = int(
                pending_nominal_fallback is not None
            )
            row["used_nominal_fallback"] = pending_nominal_fallback or 0
            pending_nominal_fallback = None

            control_steps.append(row)

            applied = to_int(
                kv.get("applied")
            )

            fallback = to_int(
                kv.get("fallback")
            )

            if fallback == 1:
                fallback_steps += 1

            if applied == 1:
                a = to_float(
                    kv.get("a")
                )

                omega = to_float(
                    kv.get("omega")
                )

                unweighted_cost = to_float(
                    kv.get("control_cost_unweighted")
                )

                weighted_cost = to_float(
                    kv.get("control_cost_weighted")
                )

                if a is not None:
                    applied_a.append(a)

                if omega is not None:
                    applied_omega.append(omega)

                if unweighted_cost is not None:
                    unweighted_control_costs.append(
                        unweighted_cost
                    )

                if weighted_cost is not None:
                    weighted_control_costs.append(
                        weighted_cost
                    )

            mpc_cost = to_float(
                kv.get("mpc_cost")
            )

            solve_time = to_float(
                kv.get("solve_time")
            )

            qp_time = to_float(
                kv.get("qp_solve_time")
            )

            constraint_time = to_float(
                kv.get("constraint_time")
            )

            dro_risk_time = to_float(
                kv.get("dro_risk_time")
            )

            dt_value = to_float(
                kv.get("dt")
            )

            acceleration_weight = to_float(
                kv.get("acceleration_weight")
            )

            steering_weight = to_float(
                kv.get("steering_weight")
            )

            if mpc_cost is not None:
                mpc_costs.append(
                    mpc_cost
                )

            if solve_time is not None:
                step_solve_times.append(
                    solve_time
                )

            if qp_time is not None:
                step_qp_solve_times.append(
                    qp_time
                )

            if constraint_time is not None:
                step_constraint_times.append(
                    constraint_time
                )

            if dro_risk_time is not None:
                step_dro_risk_times.append(
                    dro_risk_time
                )

            if dt_value is not None:
                step_dt.append(
                    dt_value
                )

            if acceleration_weight is not None:
                acceleration_weights.append(
                    acceleration_weight
                )

            if steering_weight is not None:
                steering_weights.append(
                    steering_weight
                )

        # --------------------
        # PATH COMPLETION
        # --------------------

        if "[PATH COMPLETION]" in line:
            frac = to_float(kv.get("fraction"))
            prog = to_float(kv.get("completion_progress"))
            length = to_float(kv.get("path_length"))
            completed = to_int(kv.get("completed"))
            step = to_int(kv.get("step"))

            if frac is not None:
                path_fractions.append(frac)

            if prog is not None:
                path_progress.append(prog)

            if length is not None:
                path_lengths.append(length)

            if completed == 1 and step is not None:
                completion_steps.append(step)

        # --------------------
        # DRO SUMMARY
        # --------------------

        if "[DRO SUMMARY]" in line:
            row = dict(identity)
            row.update(kv)
            dro_steps.append(row)

            dro_summary_count += 1

            for key, dest in [
                ("rho", dro_rho),
                ("risk_lift", dro_risk_lift),
                ("nominal_risk", dro_nominal_risk),
                ("qstar_risk", dro_qstar_risk),
                ("tv", dro_tv),
                ("transport_cost", dro_transport_cost),
                ("lambda", dro_lambda),
            ]:
                value = to_float(kv.get(key))
                if value is not None:
                    dest.append(value)

            if to_int(kv.get("budget_usage")) == 1:
                dro_budget_used += 1

        # --------------------
        # DRO MODE
        # --------------------

        if "[DRO MODE]" in line:
            row = dict(identity)
            row.update(kv)
            dro_modes.append(row)

        # --------------------
        # DRO SAMPLE
        # --------------------

        if "[DRO SAMPLE]" in line:
            row = dict(identity)
            row.update(kv)
            dro_samples.append(row)

        # --------------------
        # SUPPORT
        # --------------------

        if "[SQP SUPPORT]" in line:
            size = to_int(kv.get("support_size"))
            limit = to_int(kv.get("support_limit"))

            if size is not None:
                sqp_support_sizes.append(size)

            if (
                size is not None
                and limit is not None
                and size > limit
            ):
                sqp_support_exceeded_iterations += 1

        if "[SUPPORT FINAL]" in line:
            union_size = to_int(kv.get("sqp_union"))
            active = to_int(kv.get("final_active"))
            limit = to_int(kv.get("support_limit"))

            if union_size is not None:
                support_final_union.append(union_size)

            if active is not None:
                support_final_active.append(active)

            if limit is not None:
                support_limits.append(limit)

            if (
                union_size is not None
                and limit is not None
            ):
                if union_size <= limit:
                    support_final_certified += 1
                else:
                    support_final_exceeded += 1

        if "SUPPORT_NOT_EVALUATED" in line:
            support_not_evaluated += 1

        # --------------------
        # RECURSIVE FEASIBILITY
        # --------------------

        if "[RF REMOVAL ACCEPTED]" in line:
            rf_accepted += 1

            removed = to_int(kv.get("removed"))
            if removed is not None:
                rf_removed.append(removed)

        if "[RF REMOVAL REJECTED]" in line:
            rf_rejected += 1

            required = to_int(kv.get("required"))
            if required is not None:
                rf_required_rejected.append(required)

        # --------------------
        # HOMOTOPY / BRAKE
        # --------------------

        if "[HOMOTOPY TRY]" in line:
            homotopy_tries += 1

            side = kv.get("side", "unknown")
            homotopy_sides[side] += 1

        if "[BRAKE CANDIDATE]" in line:
            brake_candidates += 1

            if to_int(kv.get("feasible")) == 1:
                brake_feasible += 1

        # --------------------
        # FREE POLY
        # --------------------

        if "[FREE POLY]" in line:
            facets = to_int(kv.get("facets"))
            raw = to_int(kv.get("raw"))
            vertices = to_int(kv.get("vertices"))
            shift = to_float(kv.get("anchor_shift"))

            if facets is not None:
                free_poly_facets.append(facets)

                if facets > 20:
                    facets_over_20 += 1

            if raw is not None:
                free_poly_raw.append(raw)

            if vertices is not None:
                free_poly_vertices.append(vertices)

            if shift is not None:
                anchor_shifts.append(shift)

        if "[FREE POLY FACET WARNING]" in line:
            facet_warnings += 1

        # Important:
        # this is anchor/reference clearance, NOT actual realized
        # vehicle-obstacle clearance.
        if "[SH ANCHOR]" in line and \
           "[SH ANCHOR REPAIR]" not in line:
            clearance = to_float(
                kv.get("min_clearance")
            )

            if clearance is not None:
                anchor_clearances.append(clearance)

        # --------------------
        # REALIZED SAFETY
        # --------------------

        if "[EVAL SAFETY]" in line:
            clearance = to_float(
                kv.get("min_clearance")
            )

            center_distance = to_float(
                kv.get("min_center_distance")
            )

            collision_radius = to_float(
                kv.get("collision_radius")
            )

            step = to_int(
                kv.get("step")
            )

            collision_this_step = to_int(
                kv.get("collision_this_step")
            )

            if clearance is not None:
                actual_clearances.append(
                    clearance
                )

                if step is not None:
                    actual_clearance_steps.append(
                        (step, clearance)
                    )

                if 0.0 <= clearance < 0.10:
                    near_miss_0p10_count += 1

                if 0.0 <= clearance < 0.25:
                    near_miss_0p25_count += 1

                if 0.0 <= clearance < 0.50:
                    near_miss_0p50_count += 1

            if center_distance is not None:
                actual_center_distances.append(
                    center_distance
                )

            if collision_radius is not None:
                collision_radius_values.append(
                    collision_radius
                )

            if collision_this_step == 1:
                collision_step_count += 1

        # --------------------
        # QP
        # --------------------

        if "[SQP] before_solve" in line:
            constraints = to_int(
                kv.get("constraints")
            )

            if constraints is not None:
                qp_constraints.append(constraints)

        if "[SQP] after_solve" in line:
            converged = to_int(kv.get("converged"))
            iterations = to_int(kv.get("iterations"))
            primal = to_float(
                kv.get("primal_residual")
            )
            dual = to_float(
                kv.get("dual_residual")
            )

            if converged == 1:
                qp_converged += 1
            elif converged == 0:
                qp_failed += 1

            if iterations is not None:
                qp_iterations.append(iterations)

            if primal is not None:
                qp_primal_residuals.append(primal)

            if dual is not None:
                qp_dual_residuals.append(dual)

        if "[SQP FINAL]" in line:
            violation = to_float(
                kv.get("final_violation")
            )

            violated = to_int(
                kv.get("violated_constraints")
            )

            if violation is not None:
                sqp_final_violations.append(
                    violation
                )

            if violated is not None:
                sqp_final_violated_constraints.append(
                    violated
                )

    # ---------------------------------------------------------
    # Aggregate one run
    # ---------------------------------------------------------

    variant = (
        "dro_fallback"
        if (Path(identity["case"]).name.startswith("sh_mpcc_dro_fallback_")
            or Path(identity["case"]).name == "sh_mpcc_dro_fallback"
            or nominal_fallback_attempts > 0)
        else "dro"
        if dro_summary_count > 0
        else "non_dro"
    )
    variants = {'sh_mpcc': 'non_dro', 'sh_mpcc_dro': 'dro',
                'sh_mpcc_extra': 'extra_nominal', 'sh_mpcc_resample': 'nominal_resample',
                'sh_mpcc_dro_fallback': 'dro_fallback'}
    if identity.get('solver_style') in variants:
        variant = variants[identity['solver_style']]
    if identity.get('nominal_resampling_baseline') == 'true':
        variant = 'nominal_resample'

    collision_text = final_result.get("collision")

    if collision_text is None:
        collision = None
    else:
        collision = int(
            collision_text.lower()
            in {"yes", "true", "1"}
        )

    termination = final_result.get(
        "termination",
        ""
    )

    completed = int(
        termination == "path_complete"
        or len(completion_steps) > 0
    )

    decisions_evaluated = (
        support_final_certified
        + support_final_exceeded
    )

    certification_denominator = (
        decisions_evaluated
        + support_not_evaluated
    )

    if actual_clearance_steps:
        min_clearance_step, actual_min_clearance = min(
            actual_clearance_steps,
            key=lambda x: x[1]
        )
    else:
        min_clearance_step = None
        actual_min_clearance = None

    run = {
        **identity,

        # -----------------------
        # Variant / outcome
        # -----------------------

        "variant": variant,
        "collision": collision,
        "termination": termination,
        "path_completed": completed,

        "steps": (
            max(harness_steps)
            if harness_steps
            else None
        ),

        "final_path_fraction": (
            path_fractions[-1]
            if path_fractions
            else None
        ),

        "max_path_fraction": maximum(
            path_fractions
        ),

        "path_length": (
            path_lengths[-1]
            if path_lengths
            else None
        ),

        "harness_failure_count":
            len(harness_failures),

        # -----------------------
        # Safety / feasibility
        # -----------------------

        "max_sqp_final_violation":
            maximum(sqp_final_violations),

        "max_sqp_violated_constraints":
            maximum(
                sqp_final_violated_constraints
            ),

        # This is NOT actual realized clearance.
        "min_sh_anchor_clearance":
            minimum(anchor_clearances),

        # -----------------------
        # Realized physical safety
        # -----------------------

        "min_actual_clearance":
            actual_min_clearance,

        "min_actual_clearance_step":
            min_clearance_step,

        "mean_actual_clearance":
            mean(actual_clearances),

        "min_actual_center_distance":
            minimum(actual_center_distances),

        "collision_radius":
            (
                collision_radius_values[0]
                if collision_radius_values
                else None
            ),

        "collision_timestep_count":
            collision_step_count,

        "near_miss_0p10_count":
            near_miss_0p10_count,

        "near_miss_0p25_count":
            near_miss_0p25_count,

        "near_miss_0p50_count":
            near_miss_0p50_count,

        "near_miss_0p10_fraction":
            fraction(
                near_miss_0p10_count,
                len(actual_clearances)
            ),

        "near_miss_0p25_fraction":
            fraction(
                near_miss_0p25_count,
                len(actual_clearances)
            ),

        "near_miss_0p50_fraction":
            fraction(
                near_miss_0p50_count,
                len(actual_clearances)
            ),

        # -----------------------
        # Certification
        # -----------------------

        "support_eval_count":
            decisions_evaluated,

        "support_certified_count":
            support_final_certified,

        "support_exceeded_count":
            support_final_exceeded,

        "support_not_evaluated_count":
            support_not_evaluated,

        "support_certified_fraction_evaluated":
            fraction(
                support_final_certified,
                decisions_evaluated
            ),

        "certified_fraction_all_decisions":
            fraction(
                support_final_certified,
                certification_denominator
            ),

        "max_support_union":
            maximum(support_final_union),

        "mean_support_union":
            mean(support_final_union),

        "max_support_active":
            maximum(support_final_active),

        "support_limit":
            maximum(support_limits),

        "sqp_support_exceeded_iteration_count":
            sqp_support_exceeded_iterations,

        # -----------------------
        # Recursive feasibility
        # -----------------------

        "rf_removal_accepted_count":
            rf_accepted,

        "rf_removal_rejected_count":
            rf_rejected,

        "max_rf_removed":
            maximum(rf_removed),

        "max_rf_rejected_required":
            maximum(rf_required_rejected),

        # -----------------------
        # Recovery
        # -----------------------

        "homotopy_try_count":
            homotopy_tries,

        "homotopy_auto_count":
            homotopy_sides["auto"],

        "homotopy_path_count":
            homotopy_sides["path"],

        "homotopy_left_count":
            homotopy_sides["left"],

        "homotopy_right_count":
            homotopy_sides["right"],

        "brake_candidate_count":
            brake_candidates,

        "brake_feasible_count":
            brake_feasible,

        # -----------------------
        # Geometry / complexity
        # -----------------------

        "mean_free_poly_facets":
            mean(free_poly_facets),

        "max_free_poly_facets":
            maximum(free_poly_facets),

        "free_poly_over_20_count":
            facets_over_20,

        "facet_warning_count":
            facet_warnings,

        "max_raw_collision_constraints":
            maximum(free_poly_raw),

        "max_free_poly_vertices":
            maximum(free_poly_vertices),

        "max_anchor_shift":
            maximum(anchor_shifts),

        # -----------------------
        # Executed control
        # -----------------------

        "executed_control_steps":
            len(applied_a),

        "mean_abs_acceleration":
            mean([
                abs(x)
                for x in applied_a
            ]),

        "max_abs_acceleration":
            maximum([
                abs(x)
                for x in applied_a
            ]),

        "rms_acceleration":
            rms(applied_a),

        "mean_abs_omega":
            mean([
                abs(x)
                for x in applied_omega
            ]),

        "max_abs_omega":
            maximum([
                abs(x)
                for x in applied_omega
            ]),

        "rms_omega":
            rms(applied_omega),

        # Exactly matches the quantity currently accumulated
        # as control_effort in experiment_harness.cpp.
        "total_control_effort_unweighted":
            sum(unweighted_control_costs),

        "mean_control_effort_unweighted":
            mean(unweighted_control_costs),

        # Same controls, but using the actual MPC objective
        # acceleration/steering weights.
        "total_control_effort_weighted":
            sum(weighted_control_costs),

        "mean_control_effort_weighted":
            mean(weighted_control_costs),

        "mean_abs_delta_acceleration":
            mean_abs_delta(applied_a),

        "max_abs_delta_acceleration":
            max_abs_delta(applied_a),

        "mean_abs_delta_omega":
            mean_abs_delta(applied_omega),

        "max_abs_delta_omega":
            max_abs_delta(applied_omega),

        "control_smoothness_cost":
            (
                sum(
                    (applied_a[i] - applied_a[i - 1]) ** 2
                    +
                    (applied_omega[i] - applied_omega[i - 1]) ** 2
                    for i in range(
                        1,
                        min(
                            len(applied_a),
                            len(applied_omega)
                        )
                    )
                )
                if len(applied_a) >= 2
                and len(applied_omega) >= 2
                else None
            ),

        "dt":
            (
                step_dt[0]
                if step_dt
                else None
            ),

        "acceleration_weight":
            (
                acceleration_weights[0]
                if acceleration_weights
                else None
            ),

        "steering_weight":
            (
                steering_weights[0]
                if steering_weights
                else None
            ),

        # Integral-like version with dt included.
        "integrated_control_effort_unweighted":
            (
                step_dt[0]
                * sum(unweighted_control_costs)
                if step_dt
                else None
            ),

        "integrated_control_effort_weighted":
            (
                step_dt[0]
                * sum(weighted_control_costs)
                if step_dt
                else None
            ),

        "fallback_step_count":
            fallback_steps,

        "nominal_fallback_attempt_count":
            nominal_fallback_attempts,

        "nominal_fallback_success_count":
            nominal_fallback_successes,

        # -----------------------
        # Predicted MPC objective
        # -----------------------

        "mean_mpc_cost":
            mean(mpc_costs),

        "min_mpc_cost":
            minimum(mpc_costs),

        "max_mpc_cost":
            maximum(mpc_costs),

        "final_mpc_cost":
            (
                mpc_costs[-1]
                if mpc_costs
                else None
            ),

        # Do NOT interpret this as rollout cost.
        "mpc_cost_sample_count":
            len(mpc_costs),

        # -----------------------
        # Step timing
        # -----------------------

        "mean_controller_solve_ms":
            (
                1000.0 * mean(step_solve_times)
                if step_solve_times
                else None
            ),

        "max_controller_solve_ms":
            (
                1000.0 * maximum(step_solve_times)
                if step_solve_times
                else None
            ),

        "mean_qp_solve_ms":
            (
                1000.0 * mean(step_qp_solve_times)
                if step_qp_solve_times
                else None
            ),

        "max_qp_solve_ms":
            (
                1000.0 * maximum(step_qp_solve_times)
                if step_qp_solve_times
                else None
            ),

        "mean_constraint_build_ms":
            (
                1000.0 * mean(step_constraint_times)
                if step_constraint_times
                else None
            ),

        "max_constraint_build_ms":
            (
                1000.0 * maximum(step_constraint_times)
                if step_constraint_times
                else None
            ),

        "mean_dro_risk_eval_ms":
            (
                1000.0 * mean(step_dro_risk_times)
                if step_dro_risk_times
                else None
            ),

        "max_dro_risk_eval_ms":
            (
                1000.0 * maximum(step_dro_risk_times)
                if step_dro_risk_times
                else None
            ),

        # -----------------------
        # Solver performance
        # -----------------------

        "qp_solve_count":
            len(qp_iterations),

        "qp_converged_count":
            qp_converged,

        "qp_failed_count":
            qp_failed,

        "mean_qp_iterations":
            mean(qp_iterations),

        "max_qp_iterations":
            maximum(qp_iterations),

        "mean_qp_constraints":
            mean(qp_constraints),

        "max_qp_constraints":
            maximum(qp_constraints),

        "max_primal_residual":
            maximum(qp_primal_residuals),

        "max_dual_residual":
            maximum(qp_dual_residuals),

        # -----------------------
        # DRO statistics
        # -----------------------

        "dro_summary_count":
            dro_summary_count,

        "mean_dro_rho":
            mean(dro_rho),

        "max_dro_rho":
            maximum(dro_rho),

        "mean_dro_nominal_risk":
            mean(dro_nominal_risk),

        "mean_dro_qstar_risk":
            mean(dro_qstar_risk),

        "mean_dro_risk_lift":
            mean(dro_risk_lift),

        "max_dro_risk_lift":
            maximum(dro_risk_lift),

        "mean_dro_tv":
            mean(dro_tv),

        "max_dro_tv":
            maximum(dro_tv),

        "mean_transport_cost":
            mean(dro_transport_cost),

        "max_transport_cost":
            maximum(dro_transport_cost),

        "max_dro_lambda":
            maximum(dro_lambda),

        "dro_budget_active_count":
            dro_budget_used,

        "dro_budget_active_fraction":
            fraction(
                dro_budget_used,
                dro_summary_count
            ),
    }

    return (
        run,
        dro_steps,
        dro_modes,
        dro_samples,
        control_steps,
    )


def write_csv(path, rows, preferred_columns=None):
    rows = list(rows)

    keys = set()
    for row in rows:
        keys.update(row.keys())

    preferred_columns = list(dict.fromkeys(LABELS + (preferred_columns or [])))

    columns = [
        c for c in preferred_columns
        if c in keys or not rows
    ]

    columns.extend(
        sorted(keys - set(columns))
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Read simulation logs and structured artifacts into test-labelled CSVs (does not alter inputs).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Compatible tests (pass an output directory, not the settings JSON):
  run_analysis_matrix.py       Analysis matrix and legacy case/seed_N/*.log
  run_reviewer_matrix.py       Reviewer pilot/large suites and ablations
  run_comparison_matrix.py     Original paired comparison matrix
                              sample_efficiency*.json (five arms/concentration)
                              fixed_budget_uncertified*.json (Safe Horizon off)
  run_rare_mode_experiment.py  Frozen rare-mode artifacts or wrapper output root

A parent directory containing several suites is supported. Every exported row
has test_suite, test_name (output-root name), test_root and source provenance.
Repeats stay separate; these exports do not count repeats as independent seeds.
artifact_*.csv contains structured per-decision/per-attempt evidence;
report_*.csv preserves existing matrix reports, including paired statistics;
rare_*.csv contains frozen-scene tables. Log-derived CSV names remain unchanged.
Missing artifacts are not fabricated. Empty CSVs contain headers only.
support_certified_* legacy log fields describe support counts, not certificates;
use artifact_decisions.csv and requested/issued certificate counts instead.

Example:
  python3 tools/scrub_analysis_logs.py build-concentration/fixed-budget-uncertified-short-horizon --out scrubbed/fixed-budget
''')

    parser.add_argument(
        "root",
        type=Path,
        help="test output root or parent directory containing compatible test outputs"
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis_csv"),
        help="output directory"
    )

    args = parser.parse_args()

    root = args.root.resolve()
    args.out = args.out.resolve()
    if not root.is_dir(): parser.error('root must be an existing directory')
    if root.is_relative_to(args.out): parser.error('--out must not be the source root or an ancestor of it')

    logs = sorted(
        p for p in root.rglob("*.log") if not p.is_relative_to(args.out)
        and 'resume_history' not in p.relative_to(root).parts
    )

    run_rows = []
    dro_step_rows = []
    dro_mode_rows = []
    dro_sample_rows = []
    control_step_rows = []
    artifacts = {name: [] for name in BUNDLE_TABLES}

    for log_path in logs:
        parsed = parse_log(
            log_path,
            root
        )

        if parsed[0] is None:
            continue

        (
            run,
            steps,
            modes,
            samples,
            controls,
        ) = parsed

        for name, rows in artifact_tables(log_path, locate_run(log_path, root)).items():
            artifacts[name].extend(rows)
            if name == 'decisions':
                run['certificate_requested_decisions'] = sum(int(r['certificate_requested']) for r in rows)
                run['certified_decisions'] = sum(int(r['certified']) for r in rows)
        run['log_complete'] = int(bool(run['termination']))

        run_rows.append(run)
        dro_step_rows.extend(steps)
        dro_mode_rows.extend(modes)
        dro_sample_rows.extend(samples)
        control_step_rows.extend(controls)

    preferred = LABELS + [
        "pair_case",
        "case",
        "variant",
        "seed",
        "log_file",
        "collision",
        "termination",
        "path_completed",
        "steps",
    ]

    write_csv(
        args.out / "run_summary.csv",
        run_rows,
        preferred
    )

    write_csv(
        args.out / "dro_steps.csv",
        dro_step_rows,
        [
            "pair_case",
            "case",
            "seed",
            "step",
            "obstacle",
        ],
    )

    write_csv(
        args.out / "dro_modes.csv",
        dro_mode_rows,
        [
            "pair_case",
            "case",
            "seed",
            "step",
            "obstacle",
            "mode",
        ],
    )

    write_csv(
        args.out / "dro_samples.csv",
        dro_sample_rows,
        [
            "pair_case",
            "case",
            "seed",
            "step",
            "obstacle",
            "mode",
        ],
    )

    write_csv(
        args.out / "control_steps.csv",
        control_step_rows,
        [
            "pair_case",
            "case",
            "seed",
            "step",
            "applied",
            "success",
            "fallback",
            "nominal_fallback_attempted",
            "used_nominal_fallback",
            "a",
            "omega",
            "control_cost_unweighted",
            "control_cost_weighted",
            "mpc_cost",
            "solve_time",
            "qp_solve_time",
            "constraint_time",
            "dro_risk_time",
            "ambiguity_radius",
            "certificate",
        ],
    )

    print(
        f"Parsed {len(run_rows)} logs"
    )

    print(
        f"Wrote {args.out / 'run_summary.csv'}"
    )

    print(
        f"Wrote {args.out / 'dro_steps.csv'}"
    )

    print(
        f"Wrote {args.out / 'dro_modes.csv'}"
    )

    print(
        f"Wrote {args.out / 'dro_samples.csv'}"
    )

    print(
            f"Wrote {args.out / 'control_steps.csv'}"
    )
    for name, rows in artifacts.items():
        write_csv(args.out/f'artifact_{name}.csv', rows, LABELS + ['case', 'solver_style', 'seed', 'repeat', 'step', 'attempt', 'source_artifact'])
    reports = collect_reports(root, args.out)
    for name in ['report_'+n for n in REPORT_TABLES] + ['rare_'+n for n in RARE_TABLES]:
        write_csv(args.out/(name+'.csv'), reports.get(name, []), LABELS + ['source_artifact'])
    print(f'Exported {sum(map(len, artifacts.values()))} structured artifact rows and '
          f'{sum(map(len, reports.values()))} report/frozen-scene rows; all labelled by test.')


if __name__ == "__main__":
    main()
