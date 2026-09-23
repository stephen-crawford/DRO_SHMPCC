/**
 * @file config.hpp
 * @brief Runtime configuration for scenario MPC — SOURCE OF TRUTH.
 *
 * Owns controller settings only:
 *   - Ego vehicle specification (geometry + kinematic limits)
 *   - MPC type, horizon, objective, constraints, sampling / belief
 *   - DRO on/off, risk measure, ground cost, ambiguity radius, OT
 *   - QP / SQP solver knobs
 *
 * Numeric defaults MUST match configs/default.yaml (the runtime source of
 * truth). In-class initializers exist only as a fallback when that file
 * cannot be loaded.
 *
 * World setup (obstacles, environment, rollout protocol) lives in
 * experiment_harness.hpp.
 *
 * Split:
 *   config.hpp             — controller runtime (RuntimeConfig = mpc + dro + solver)
 *   experiment_harness.hpp — world / trial protocol (ExperimentConfig wraps the above)
 *
 */

#ifndef DRO_MPC_CONFIG_HPP
#define DRO_MPC_CONFIG_HPP

#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dro_mpc {

// ============================================================================
// MPC type
// ============================================================================

enum class MPCType {
    MPC,     // Point-to-point / goal-tracking
    MPCC,    // Contouring control (path following)
    SH_MPC,  // Safe-horizon MPC
    SH_MPCC, // Safe-horizon MPCC
    SH_MPCC_DRO_FALLBACK // DRO first, nominal SH-MPCC after recovery exhaustion
};

inline std::string mpc_type_name(MPCType t) {
    switch (t) {
        case MPCType::MPC:     return "mpc";
        case MPCType::MPCC:    return "mpcc";
        case MPCType::SH_MPC:  return "sh_mpc";
        case MPCType::SH_MPCC: return "sh_mpcc";
        case MPCType::SH_MPCC_DRO_FALLBACK: return "sh_mpcc_dro_fallback";
        default: return "unknown";
    }
}

// ============================================================================
// MPC objective / constraints / ego
// ============================================================================

struct MPCObjectiveWeights {
    double goal_weight = 10.0;
    double progress_weight = 10.0;  // MPCC: reward average projected arc-length rate [m/s]
    double velocity_weight = 1.0;
    double acceleration_weight = 0.1;
    double steering_weight = 0.1;
    double terminal_heading_weight = 1.0;
    double contour_weight = 1.0;            //  Active for MPCC / SH_MPCC
    double lag_weight = 0.1;                // Active for MPCC / SH_MPCC
};

struct MPCConstraintSettings {
    double safety_margin = 0.1;
    double road_width = 7.0;                //  Active when contouring is on

    /// Hard box on v_{k+1} in [ego.dynamics.min_velocity, ego.dynamics.max_velocity].
    bool enable_velocity_bounds = true;

    /// Drop collision half-spaces whose linearization point is farther than this
    /// from the reference (metres). Does not affect scenario dominance pruning.
    double clearance_filter_distance = 20.0;

    /// de Groot support cap n̄: an upper limit on the number of DISTINCT SUPPORT
    /// SCENARIOS (counted by unique scenario_id), NOT individual collision constraints.
    /// Support estimate is the UNION of active scenario IDs across all feasible convex
    /// iterations, n̂ = |∪_ℓ ω_active^ℓ| (with C ⊆ Ĉ, n ≤ n̂), not just the constraints
    /// active at the final optimum. This is the NON-REMOVED support cap: with a removal
    /// budget R the TOTAL support limit is n̄ + R (removed scenarios join the support,
    /// de Groot Thm. 5).
    // scenario_module/config/params.yaml uses n_bar: 6 by default.
    int support_cap_n_bar = 6;

    /// Conservative offline budget R for removed scenarios.  The controller
    /// does not remove scenarios online; nevertheless, a stated budget raises
    /// the total certificate cap exactly as in de Groot Thm. 5.
    int scenario_removal_budget = 0;

    /// Total support cap used by the certificate: n̄ + R.
    int total_support_cap() const noexcept {
        return std::max(0, support_cap_n_bar) +
               std::max(0, scenario_removal_budget);
    }
};

// ---------------------------------------------------------------------------
// Ego dynamics model
// ---------------------------------------------------------------------------


enum class DynamicsModel {
    SECOND_ORDER_UNICYCLE  //State [x,y,theta,v], input [a, w]. (Others deferred.)
};

inline std::string dynamics_model_name(DynamicsModel m) {
    switch (m) {
        case DynamicsModel::SECOND_ORDER_UNICYCLE: return "second_order_unicycle";
        default: return "unknown";
    }
}

/**
 * @brief Ego dynamics-model specification: which motion model, plus the
 *        kinematic limits enforced ON that model.
 */
struct EgoDynamicsConfig {
    DynamicsModel model = DynamicsModel::SECOND_ORDER_UNICYCLE;

    double max_velocity = 4.0;       // Hard upper velocity bound [m/s]
    double min_velocity = 0.0;       // Hard lower velocity bound [m/s]
    double max_acceleration = 3.0;   // Maximum acceleration [m/s^2]
    double min_acceleration = -5.0;  // Minimum acceleration (braking) [m/s^2]
    double max_omega = 0.8;  // Maximum angular velocity [rad/s]
};

/**
 * @brief Ego vehicle geometry + dynamics model.
 *
 * Geometry (collision radius, body length, disc count) is separate from motion:
 * the dynamics model and its kinematic limits live in `dynamics`.
 */
struct EgoVehicleSpecification {
    double radius = 0.5;             // Collision radius [m]
    double length = 4.0;             // Length for multi-disc placement [m]
    int num_discs = 3;               // Number of discs along the body

    EgoDynamicsConfig dynamics;      // Motion model + kinematic limits
};

// ============================================================================
// Nominal mode belief
// ============================================================================

enum class NominalBeliefKind {
    DIRICHLET,  //Symmetric Dirichlet prior
    STICKY      //Dirichlet + self-persistence (sticky) prior
};

// Build a ModeBeliefConfig (types.hpp) from the belief kind. The sticky
// self-persistence prior theta is only wired in for STICKY.
inline ModeBeliefConfig make_mode_belief_config(
    NominalBeliefKind kind,
    double self_persistence_prior = 0.8
) {
    ModeBeliefConfig cfg;
    cfg.prior = DirichletPrior::KRICHEVSKY_TROFIMOV;
    cfg.self_persistence_prior =
        (kind == NominalBeliefKind::STICKY) ? self_persistence_prior : 0.0;
    return cfg;
}

/**
 * @brief Scenario sampling and nominal mode belief.
 *
 * Active for scenario-based MPC. Markov sampling / sticky belief only take
 * effect when the corresponding flags / kinds are selected.
 */
struct ScenarioSamplingSettings {
    /// Effective number of sampled scenarios. When automatic sizing is on it
    /// is derived during RuntimeConfig::normalize(); otherwise it is the
    /// explicit static sample count from YAML or C++.
    int num_scenarios = 40;

    double one_minus_chance_constraint_violation_probability = 0.95; // = 1 - eps  (safety prob)
    double chance_of_certificate_violation = 0.01; // = beta  (certificate confidence = 1 - beta)

    /// Derive S from the certified total support cap n̄ + R.
    bool automatically_compute_sample_size = true;

    /// Select an explicit static sample count. Prefer this over assigning
    /// num_scenarios directly when constructing an experiment in C++ so a
    /// manual sweep cannot be overwritten during normalization.
    void set_manual_sample_count(int count) {
        if (count <= 0) {
            throw std::invalid_argument("num_scenarios must be positive");
        }
        num_scenarios = count;
        automatically_compute_sample_size = false;
    }

    /// Select reference-style automatic Safe-Horizon sample sizing.
    void use_automatic_sample_sizing() {
        automatically_compute_sample_size = true;
    }

    /// Positive values retain a rolling belief window; -1 retains all mode
    /// observations.  Zero is rejected so it cannot silently acquire a
    /// horizon-dependent meaning at runtime.
    int max_history_length = -1;

    bool markov_jump_system = false;

    NominalBeliefKind belief_kind = NominalBeliefKind::DIRICHLET;
    ModeBeliefConfig mode_belief{};

    void sync_belief() {
        const double sticky =
            mode_belief.self_persistence_prior > 0.0
                ? mode_belief.self_persistence_prior : 0.8;
        mode_belief = make_mode_belief_config(belief_kind, sticky);
    }

    double epsilon() const { return 1.0 - one_minus_chance_constraint_violation_probability; }
};

// ============================================================================
// MPC config
// ============================================================================

struct MPCConfig {
    MPCType type = MPCType::SH_MPCC;
    int horizon = 20;
    double dt = 0.1;

    EgoVehicleSpecification ego;
    MPCObjectiveWeights objective;
    MPCConstraintSettings constraints;
    ScenarioSamplingSettings sampling;

    bool safe_horizon_enabled = true;
    bool enable_contouring_constraints = true;  //Road-boundary + contouring cost
    // Opt-in ablation: same hybrid recovery/retry path, both attempts nominal.
    bool nominal_resampling_baseline = false;

    /// Derive safe-horizon / contouring enablement from the MPC type.
    /// Call after setting `type` and before overriding those two flags by hand.
    void sync_from_type() {
        switch (type) {
            case MPCType::MPC:
                safe_horizon_enabled = false;
                enable_contouring_constraints = false;
                break;
            case MPCType::MPCC:
                safe_horizon_enabled = false;
                enable_contouring_constraints = true;
                break;
            case MPCType::SH_MPC:
                safe_horizon_enabled = true;
                enable_contouring_constraints = false;
                break;
            case MPCType::SH_MPCC:
            case MPCType::SH_MPCC_DRO_FALLBACK:
                safe_horizon_enabled = true;
                enable_contouring_constraints = true;
                break;
        }
    }

    bool uses_safe_horizon() const { return safe_horizon_enabled; }
};

// ============================================================================
// DRO — risk / ground cost
// ============================================================================
//
// DROGroundCostType, DRORiskMeasure, and DRORiskScoringModel are defined in
// types.hpp.

inline std::string ground_cost_name(DROGroundCostType g) {
    switch (g) {
        case DROGroundCostType::W2_BURES:      return "w2_bures";
        case DROGroundCostType::W1_METRIC:     return "w1_metric";
        case DROGroundCostType::ZERO_ONE:      return "zero_one";
        case DROGroundCostType::EUCLIDEAN_MEAN: return "euclidean_mean";
        default: return "unknown";
    }
}

inline std::string risk_measure_name(DRORiskMeasure r) {
    switch (r) {
        case DRORiskMeasure::SURROGATE_VAR:            return "surrogate_var";
        case DRORiskMeasure::SURROGATE_CVAR:           return "surrogate_cvar";
        case DRORiskMeasure::SURROGATE_VAR_BONFERRONI: return "surrogate_var_bonferroni";
        case DRORiskMeasure::BONFERRONI_VAR:           return "bonferroni_var";
        case DRORiskMeasure::MIXTURE_VAR:              return "mixture_var";
        case DRORiskMeasure::MIXTURE_CVAR:             return "mixture_cvar";
        case DRORiskMeasure::JOINT_VAR:                return "joint_var";
        case DRORiskMeasure::JOINT_CVAR:               return "joint_cvar";
        default: return "unknown";
    }
}

inline std::string risk_scoring_model_name(DRORiskScoringModel model) {
    switch (model) {
        case DRORiskScoringModel::INHERIT_RISK_MEASURE:
            return "inherit";
        case DRORiskScoringModel::CERTIFIED_SURROGATE:
            return "certified_surrogate";
        case DRORiskScoringModel::EUCLIDEAN_BONFERRONI_VAR:
            return "euclidean_bonferroni_var";
        case DRORiskScoringModel::EUCLIDEAN_JOINT_VAR:
            return "euclidean_joint_var";
        case DRORiskScoringModel::EUCLIDEAN_JOINT_CVAR:
            return "euclidean_joint_cvar";
        default:
            return "unknown";
    }
}

// ============================================================================
// DRO — ambiguity-radius calibration
// ============================================================================

/**
 * @brief Ambiguity-radius calibration and DRO risk/reweighting settings.
 *
 * For Wasserstein ambiguity sets
 *
 *     U(p_hat, rho)
 *       = { q in Delta_M : W_D(p_hat, q) <= rho },
 *
 * D is the configured mode-space ground metric. With W2_BURES,
 *
 *     D_ij = average_k W2(
 *         N(mu_i,k, Sigma_i,k),
 *         N(mu_j,k, Sigma_j,k)).
 *
 * The outer ambiguity distance remains discrete optimal transport W_D over
 * mode probabilities. W2-Bures defines the ground geometry between modes.
 *
 * Three radius calibrations are available:
 *
 *   BHC_DMAX:
 *     Finite-sample categorical TV concentration lifted through D_max.
 *
 *   CLOPPER_PEARSON:
 *     Simultaneous exact-binomial confidence bounds intersected with the
 *     probability simplex, then outer-approximated by the smallest
 *     center-fixed Wasserstein ball containing that confidence polytope.
 *
 *   EXACT_MULTINOMIAL_GRID:
 *     Exact multinomial Wasserstein-test inversion at every tested simplex
 *     lattice point. Exact at each candidate p, but lattice-approximate over
 *     the continuous simplex.
 */
 struct RadiusCalibrationSettings {
    bool use_calibrated_radius = true;

    /// Desired miscoverage probability for the ambiguity set:
    ///
    ///     P(p_true in U) >= 1 - confidence_beta.
    double confidence_beta = 0.05;

    /// Risk level used by VaR/CVaR scoring.
    /// This is independent of ambiguity-set confidence_beta.
    double alpha_one_sided = 0.95;

    /**
     * @brief Radius calibration used when divergence == WASSERSTEIN.
     *
     * BHC_DMAX:
     *   Existing Schuurmans/BHC construction
     *
     *       rho = D_max * sqrt(r_TV(M,m,beta)).
     *
     * CLOPPER_PEARSON:
     *   Construct simultaneous finite-sample Clopper-Pearson intervals for
     *   the categorical mode probabilities, intersect with the simplex, and
     *   choose
     *
     *       rho = max_{p in C_beta} W_D(p_hat, p).
     *
     *   This provides finite-sample coverage over the continuous simplex
     *   under the IID categorical model.
     *
     * EXACT_MULTINOMIAL_GRID:
     *   Invert the exact multinomial Wasserstein test at every candidate p
     *   on a finite simplex lattice. The multinomial p-value at each p is
     *   exact, but the continuum inversion is approximated by the lattice.
     */
    WassersteinRadiusCalibrationMethod wasserstein_radius_method =
        WassersteinRadiusCalibrationMethod::CLOPPER_PEARSON;

    /**
     * Simplex lattice denominator used only by EXACT_MULTINOMIAL_GRID:
     *
     *     p_i = k_i / L,
     *     sum_i k_i = L.
     *
     * Larger L gives a finer inversion but rapidly increases runtime.
     */
    int exact_multinomial_grid_denominator = 10;

    /**
     * Legacy scale factor used by BHC_DMAX.
     *
     * Keep this at 1.0 for a direct theoretical comparison.
     * CLOPPER_PEARSON and EXACT_MULTINOMIAL_GRID should not multiply their
     * statistically calibrated radius by this value.
     */
    double calibration_scale = 1.0;

    /// Exact primal discrete-Wasserstein OT reweighting.
    bool use_primal_ot = true;

    DRORiskMeasure risk_measure =
        DRORiskMeasure::SURROGATE_VAR_BONFERRONI;

    DRORiskScoringModel risk_scoring_model =
        DRORiskScoringModel::INHERIT_RISK_MEASURE;

    /// -1 follows the controller-provided full risk horizon.
    int risk_horizon = -1;

    AmbiguityDivergence divergence =
        AmbiguityDivergence::WASSERSTEIN;

    int joint_risk_samples = 8000;
    uint64_t joint_risk_seed = 0x5150C0FFEEULL;

    int mixture_sequence_samples = 512;

    double sigma_floor = 1e-6;

    bool use_entropic_allocator = false;
    double entropic_tau = 0.05;
};

struct DROConfig {
    /// Used when use_calibrated_radius == false.
    double base_radius = 0.1;

    /**
     * Engineering clamps for fixed/legacy radius operation.
     *
     * Statistically calibrated CLOPPER_PEARSON and
     * EXACT_MULTINOMIAL_GRID radii must not be clipped by these values,
     * because doing so can destroy the stated confidence coverage.
     */
    double min_radius = 0.01;
    double max_radius = 0.10;

    RadiusCalibrationSettings radius_calibration;

    DROGroundCostType ground_cost_type =
        DROGroundCostType::W2_BURES;
};

/**
 * @brief Controller-facing DRO settings.
 *
 */
struct DROControllerConfig {
    bool enabled = false;

    /// If > 0, disable calibrated rho and pin the ambiguity radius to this value.
    double fixed_rho = -1.0;

    DROConfig solver;

    void apply_fixed_rho() {
        if (fixed_rho <= 0.0) return;
        solver.radius_calibration.use_calibrated_radius = false;
        solver.base_radius = fixed_rho;
        solver.min_radius = std::min(solver.min_radius, fixed_rho);
        solver.max_radius = std::max(solver.max_radius, fixed_rho);
    }
};

// ============================================================================
// Solver
// ============================================================================

struct SolverSettings {
    int sqp_max_iterations = 5;        // Maximum SQP outer iterations.
    double sqp_convergence_tol = 1e-3; // Convergence tolerance on ||delta_u||.
    int qp_max_iterations = 200;       // acados/HPIPM interior-point iteration cap.
    double qp_tolerance = 1e-4;        // acados/HPIPM residual tolerance.
};

// ============================================================================
// Complete runtime config
// ============================================================================

/**
 * @brief Full controller runtime config: MPC + DRO + solver + world radius.
 */
struct RuntimeConfig {
    MPCConfig mpc;
    DROControllerConfig dro;
    SolverSettings solver;

    /// Seed for controller-owned random sampling. Zero selects nondeterministic seeding.
    unsigned random_seed = 0;

    /// Obstacle collision radius used with ego.radius for halfspaces.
    double obstacle_radius = 0.35;

    double combined_radius() const {
        return mpc.ego.radius + obstacle_radius + mpc.constraints.safety_margin;
    }

    double epsilon() const { return mpc.sampling.epsilon(); }

    bool enable_dro() const { return dro.enabled; }

    /// This is the number checked
    /// against the union of support scenarios and used to size S.
    int support_limit() const noexcept {
        return mpc.constraints.total_support_cap();
    }

    int scenario_removal_budget() const {return mpc.constraints.scenario_removal_budget;}

    // ---- de Groot (arXiv:2307.01070) scenario-theoretic sample complexity ----
    // de Groot's Safe-Horizon MPC sizes the sample count S from the NONCONVEX
    // scenario-optimization (NSO) bound of Campi-Garatti, NOT the convex
    // Calafiore-Campi dimension bound. The guarantee is stated through the SUPPORT
    // SIZE n (the number of scenarios that hold the solution in place), capped by a
    // support limit n̄. de Groot Eq. (7)-(8) + Theorem 1 / Algorithm 1 line 2. The
    // theorem requires the confidence tail to be at most β:
    //
    //   P^S[ V(θ*) > ε(n) ]  ≤  Σ_{k=0}^{S-1} C(S,k) (1-ε(k))^{S-k}  ≤  β,
    //   ε(n) = 1 - ( β / (S · C(S,n)) )^{1/(S-n)}   for n ≤ n̄ ,   ε(n)=1 for n>n̄.
    //
    // Each term with k ≤ n̄ contributes exactly β/S; with the piecewise cap ε(k)=1 for
    // k > n̄ the terms above the cap vanish, so the sum is (n̄+1)·β/S ≤ β (equality Σ = β
    // would need ε_S(k) applied for ALL k, i.e. no cap). Using ε(n̄) ≤ ε then certifies
    // every n ≤ n̄ since ε(n) is increasing in n.
    //
    // and S is the SMALLEST sample size with ε(n̄) ≤ ε ("bisection of Eq. 8").
    // Scenario removal (Sec. VI, Thm. 5): removed scenarios join the support, so a
    // removal budget R raises the effective support limit to n̄ + R.

    /// Realized per-decision violation risk ε(n) at sample size S and support n
    /// (de Groot Eq. 8), evaluated in log-space for numerical stability. Returns 1.0
    /// when n ≥ S (under-sampled → no guarantee).
    static double degroot_violation_risk(int S, int n, double beta) {
        if (S <= 0 || n >= S) return 1.0;
        if (n < 0) n = 0;
        const double log_binom = std::lgamma(S + 1.0)
                               - std::lgamma(n + 1.0)
                               - std::lgamma(S - n + 1.0);
        // (1-ε)^{S-n} = β / (S·C(S,n))  ⇒  ε = 1 - exp( [ln β - ln S - lnC] / (S-n) ).
        const double log_inner =
            std::log(beta) - std::log(static_cast<double>(S)) - log_binom;
        return 1.0 - std::exp(log_inner / static_cast<double>(S - n));
    }

    /// de Groot Algorithm 1, line 2: the smallest sample size S with epsilon
    /// at the specified total support cap no larger than the configured risk.
    int compute_required_scenarios(
        int nonremoved_support_cap,
        int removal_budget = 0
    ) const {
        const double eps = epsilon();
        const double beta = mpc.sampling.chance_of_certificate_violation;
        const int n = std::max(0, nonremoved_support_cap) +
                      std::max(0, removal_budget);
        int hi = std::max(n + 1, 1);                       // exponential upper bracket
        while (degroot_violation_risk(hi, n, beta) > eps) {
            hi *= 2;
            if (hi > (1 << 24)) return hi;                 // safety cap
        }
        int lo = n + 1;
        while (lo < hi) {                                  // bisection (monotone in S)
            const int mid = lo + (hi - lo) / 2;
            if (degroot_violation_risk(mid, n, beta) <= eps) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    /// Required S for the configured reference support cap n-bar + R.
    int compute_required_scenarios() const {
        return compute_required_scenarios(
            mpc.constraints.support_cap_n_bar,
            mpc.constraints.scenario_removal_budget);
    }

    /// Realized joint-risk certificate ε(n_total) actually guaranteed by S_actual drawn
    /// scenarios (de Groot Eq. 8) — the exact inverse of compute_required_scenarios.
    /// The support arguments follow the SAME non-removed convention: the total support
    /// bound is nonremoved_support_bound + R. Pass either the measured online support n̂
    /// (with R=0) or the configured n̄ + R; in both cases it must be a valid UPPER bound
    /// on the total support for the returned ε to be a valid certificate.
    double compute_effective_epsilon(int S_actual, int nonremoved_support_bound, int R = 0) const {
        return degroot_violation_risk(S_actual,
                                      std::max(0, nonremoved_support_bound) + std::max(0, R),
                                      mpc.sampling.chance_of_certificate_violation);
    }

    void normalize() {
        if (mpc.type == MPCType::SH_MPCC_DRO_FALLBACK)
            dro.enabled = !mpc.nominal_resampling_baseline;
        // Do not call mpc.sync_from_type() here — it would overwrite SH/contouring
        // overrides intentionally set after type selection.
        mpc.sampling.sync_belief();
        dro.apply_fixed_rho();
        if (mpc.uses_safe_horizon() &&
            mpc.sampling.automatically_compute_sample_size) {
            mpc.sampling.num_scenarios = compute_required_scenarios();
        }
    }

    void validate() const {
        if (mpc.nominal_resampling_baseline && mpc.type != MPCType::SH_MPCC_DRO_FALLBACK)
            throw std::invalid_argument("nominal_resampling_baseline requires sh_mpcc_dro_fallback");
        if (!std::isfinite(mpc.objective.progress_weight) || mpc.objective.progress_weight < 0.0)
            throw std::invalid_argument("progress_weight must be finite and non-negative");
        if (mpc.horizon <= 0) throw std::invalid_argument("horizon must be positive");
        if (mpc.dt <= 0) throw std::invalid_argument("dt must be positive");
        if (mpc.sampling.one_minus_chance_constraint_violation_probability <= 0 || mpc.sampling.one_minus_chance_constraint_violation_probability >= 1)
            throw std::invalid_argument("confidence_level must be in (0, 1)");
        if (mpc.sampling.chance_of_certificate_violation <= 0 || mpc.sampling.chance_of_certificate_violation >= 1)
            throw std::invalid_argument("beta must be in (0, 1)");
        if (mpc.sampling.num_scenarios <= 0)
            throw std::invalid_argument("num_scenarios must be positive");
        if (mpc.sampling.max_history_length == 0 ||
            mpc.sampling.max_history_length < -1)
            throw std::invalid_argument(
                "max_history_length must be -1 or positive");
        if (mpc.constraints.support_cap_n_bar < 0)
            throw std::invalid_argument("support_cap_n_bar must be non-negative");
        if (mpc.constraints.scenario_removal_budget < 0)
            throw std::invalid_argument("scenario_removal_budget must be non-negative");

        const auto& radius = dro.solver.radius_calibration;
        if (!is_valid_wasserstein_radius_method(
            radius.wasserstein_radius_method)) {
                throw std::invalid_argument(
                    "wasserstein_radius_method is invalid");
            }
        if (!(radius.confidence_beta > 0.0 &&
                radius.confidence_beta < 1.0)) {
                throw std::invalid_argument(
                    "confidence_beta must be in (0, 1)");
            }
        if (radius.divergence !=
                AmbiguityDivergence::WASSERSTEIN &&
            radius.wasserstein_radius_method !=
                WassersteinRadiusCalibrationMethod::BHC_DMAX) {

            throw std::invalid_argument(
                    "wasserstein_radius_method is only applicable "
                    "when divergence == WASSERSTEIN");
            }
        if (radius.exact_multinomial_grid_denominator <= 0) {
                throw std::invalid_argument(
                    "exact_multinomial_grid_denominator must be positive");
            }
        if (!is_valid_risk_scoring_model(radius.risk_scoring_model))
            throw std::invalid_argument("risk_scoring_model is invalid");
        if (radius.joint_risk_samples <= 0)
            throw std::invalid_argument("joint_risk_samples must be positive");
        if (radius.mixture_sequence_samples <= 0)
            throw std::invalid_argument("mixture_sequence_samples must be positive");
        if (radius.risk_horizon == 0 || radius.risk_horizon < -1)
            throw std::invalid_argument("risk_horizon must be -1 or positive");
    }
};

}  // namespace dro_mpc

#endif  // DRO_MPC_CONFIG_HPP
