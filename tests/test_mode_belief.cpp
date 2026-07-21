// Tests for the Bayesian mode-belief estimator and Markov mode sampling.
//
// These are written to FALSIFY the claims, not confirm them. Each asserts a
// property that would break if the estimator regressed to the pre-Bayesian
// behaviour (a hard zero-mask on unobserved modes, which made UNIFORM /
// EPSILON_GREEDY / TEMPERATURE all silently identical to FREQUENCY).

#include "mode_weights.hpp"
#include "wasserstein_dro.hpp"
#include "scenario_sampler.hpp"
#include "dynamics.hpp"
#include "types.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace dro_mpc;

static int g_failures = 0;
static void check(bool ok, const std::string& what) {
    std::cout << (ok ? "PASS: " : "FAIL: ") << what << "\n";
    if (!ok) ++g_failures;
}
static bool close_to(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol;
}

static std::map<std::string, ModeModel> make_library() {
    auto all = create_obstacle_mode_models(0.1);
    std::map<std::string, ModeModel> lib;
    for (const char* m : {"constant_velocity", "turn_left", "lane_change_left"}) {
        lib[m] = all[m];
    }
    return lib;
}

// ---------------------------------------------------------------------------
// 1. Dirichlet posterior-predictive mean, checked against the closed form.
// ---------------------------------------------------------------------------
static void test_dirichlet_posterior_mean() {
    auto lib = make_library();
    ModeHistory h(0, lib, 0);
    for (int t = 0; t < 20; ++t) h.record_observation(t, "constant_velocity");

    const double alpha = 1.0;
    auto w = compute_mode_weights(h, WeightType::FREQUENCY, 0.9, 20, alpha);

    // p_m = (n_m + a) / sum_j (n_j + a);  n_cv = 20, M = 3, a = 1
    //   cv   = 21/23,  others = 1/23
    check(close_to(w["constant_velocity"], 21.0 / 23.0, 1e-12),
          "FREQUENCY observed mode == (n+a)/(N+Ma) closed form");
    check(close_to(w["lane_change_left"], 1.0 / 23.0, 1e-12),
          "FREQUENCY unobserved mode == a/(N+Ma) closed form");
    check(w["lane_change_left"] > 0.0,
          "unobserved mode has STRICTLY POSITIVE mass (no zero-mask regression)");

    double total = 0.0;
    for (const auto& [_, v] : w) total += v;
    check(close_to(total, 1.0, 1e-12), "FREQUENCY belief normalized");
}

// ---------------------------------------------------------------------------
// 2. alpha actually moves the belief (it is a real knob, not a dead default).
// ---------------------------------------------------------------------------
static void test_alpha_is_live() {
    auto lib = make_library();
    ModeHistory h(0, lib, 0);
    for (int t = 0; t < 20; ++t) h.record_observation(t, "constant_velocity");

    auto w_laplace = compute_mode_weights(h, WeightType::FREQUENCY, 0.9, 20, 1.0);
    auto w_kt      = compute_mode_weights(h, WeightType::FREQUENCY, 0.9, 20, 0.5);

    // Krichevsky-Trofimov (a=1/2) puts LESS mass on the unseen mode than Laplace.
    check(w_kt["lane_change_left"] < w_laplace["lane_change_left"],
          "smaller alpha => less mass on unobserved mode (alpha is live)");
    check(close_to(w_kt["lane_change_left"], 0.5 / 21.5, 1e-12),
          "KT alpha=1/2 matches closed form a/(N+Ma)");
    check(w_kt["lane_change_left"] > 0.0, "KT still strictly positive");
}

// ---------------------------------------------------------------------------
// 3. The estimators are DISTINCT. Pre-fix, all four collapsed to FREQUENCY.
// ---------------------------------------------------------------------------
static void test_estimators_are_distinct() {
    auto lib = make_library();
    ModeHistory h(0, lib, 0);
    for (int t = 0; t < 20; ++t) h.record_observation(t, "constant_velocity");

    auto f = compute_mode_weights(h, WeightType::FREQUENCY, 0.9, 20, 1.0);
    auto u = compute_mode_weights(h, WeightType::UNIFORM, 0.9, 20, 1.0);
    auto e = compute_mode_weights(h, WeightType::EPSILON_GREEDY, 0.9, 20, 1.0);

    check(close_to(u["constant_velocity"], 1.0 / 3.0, 1e-12),
          "UNIFORM is genuinely uniform (1/M)");
    check(!close_to(u["lane_change_left"], f["lane_change_left"], 1e-6),
          "UNIFORM != FREQUENCY (zero-mask regression would collapse them)");
    check(e["lane_change_left"] > f["lane_change_left"],
          "EPSILON_GREEDY lifts the unobserved mode above FREQUENCY");
}

// ---------------------------------------------------------------------------
// 4. Transition matrix: row-stochastic, strictly positive, correct posterior mean.
// ---------------------------------------------------------------------------
static void test_transition_matrix() {
    auto lib = make_library();
    ModeHistory h(0, lib, 0);
    // cv, cv, cv, tl, tl, cv  =>  N[cv][cv]=2, N[cv][tl]=1, N[tl][tl]=1, N[tl][cv]=1
    const char* seq[] = {"constant_velocity", "constant_velocity", "constant_velocity",
                         "turn_left", "turn_left", "constant_velocity"};
    for (int t = 0; t < 6; ++t) h.record_observation(t, seq[t]);

    std::vector<std::string> modes = {"constant_velocity", "lane_change_left", "turn_left"};
    const double a = 1.0, kappa = 2.0;
    auto T = compute_mode_transition_matrix(h, modes, a, kappa);

    check(T.rows() == 3 && T.cols() == 3, "transition matrix is M x M");
    for (int i = 0; i < T.rows(); ++i) {
        check(close_to(T.row(i).sum(), 1.0, 1e-12),
              "transition row " + std::to_string(i) + " is stochastic");
    }
    check((T.array() > 0.0).all(),
          "every transition strictly positive (no unreachable mode)");

    // Row cv: (N+a+kappa*1{i=j}) = (2+1+2, 0+1, 1+1) = (5,1,2), Z = 8
    check(close_to(T(0, 0), 5.0 / 8.0, 1e-12), "T(cv,cv) == 5/8 closed form");
    check(close_to(T(0, 1), 1.0 / 8.0, 1e-12), "T(cv,lcl) == 1/8 closed form");
    check(close_to(T(0, 2), 2.0 / 8.0, 1e-12), "T(cv,tl) == 2/8 closed form");
    // Row lcl (never observed): (0+1, 0+1+2, 0+1) = (1,3,1), Z = 5
    check(close_to(T(1, 1), 3.0 / 5.0, 1e-12),
          "never-observed row is the pure prior: sticky self == 3/5");

    // sticky_bonus is live: kappa=0 must remove the self-transition bias.
    auto T0 = compute_mode_transition_matrix(h, modes, a, 0.0);
    check(T0(1, 1) < T(1, 1), "sticky_bonus is live (kappa=0 lowers self-transition)");
    check(close_to(T0(1, 1), 1.0 / 3.0, 1e-12),
          "kappa=0 never-observed row is uniform 1/3");
}

// ---------------------------------------------------------------------------
// 5. HMM prediction: p_{t+1} = T^T p_t, and mass reaches an unobserved mode.
// ---------------------------------------------------------------------------
static void test_prediction_reaches_unobserved_mode() {
    auto lib = make_library();
    ModeHistory h(0, lib, 0);
    const char* seq[] = {"constant_velocity", "constant_velocity", "constant_velocity",
                         "turn_left", "turn_left", "constant_velocity"};
    for (int t = 0; t < 6; ++t) h.record_observation(t, seq[t]);
    std::vector<std::string> modes = {"constant_velocity", "lane_change_left", "turn_left"};
    auto T = compute_mode_transition_matrix(h, modes, 1.0, 2.0);

    ModeDistribution b;
    b["constant_velocity"] = 1.0; b["lane_change_left"] = 0.0; b["turn_left"] = 0.0;
    auto p1 = predict_mode_belief(b, T, modes);

    // p1(j) = sum_i T(i,j) p0(i) = T(0,j) since p0 = e_cv
    check(close_to(p1["constant_velocity"], 5.0 / 8.0, 1e-12), "predict == T^T p (cv)");
    check(close_to(p1["lane_change_left"], 1.0 / 8.0, 1e-12), "predict == T^T p (lcl)");
    check(close_to(p1["turn_left"], 2.0 / 8.0, 1e-12), "predict == T^T p (tl)");

    double s = 0.0; for (const auto& [_, v] : p1) s += v;
    check(close_to(s, 1.0, 1e-12), "predicted belief normalized");

    // Iterate to (near) stationarity. Solving pi*T = pi by hand gives
    // pi = [8,5,7]/20 = [0.40, 0.25, 0.35].
    ModeDistribution p = b;
    for (int i = 0; i < 500; ++i) p = predict_mode_belief(p, T, modes);
    check(close_to(p["constant_velocity"], 0.40, 1e-6), "stationary cv == 0.40");
    check(close_to(p["lane_change_left"], 0.25, 1e-6),
          "stationary NEVER-OBSERVED mode == 0.25 (prior-driven, > 0)");
    check(close_to(p["turn_left"], 0.35, 1e-6), "stationary tl == 0.35");
}

// ---------------------------------------------------------------------------
// 6. Bayes correction recovers a mode the prior says is impossible.
// ---------------------------------------------------------------------------
static void test_bayes_update() {
    auto lib = make_library();
    ModeHistory h(0, lib, 0);
    for (int t = 0; t < 6; ++t) h.record_observation(t, "constant_velocity");
    std::vector<std::string> modes = {"constant_velocity", "lane_change_left", "turn_left"};
    auto T = compute_mode_transition_matrix(h, modes, 1.0, 2.0);

    ModeDistribution prior;
    prior["constant_velocity"] = 1.0; prior["lane_change_left"] = 0.0; prior["turn_left"] = 0.0;
    ModeDistribution like;
    like["constant_velocity"] = 1e-6; like["lane_change_left"] = 1.0; like["turn_left"] = 1e-6;

    auto post = update_mode_belief(prior, T, modes, like, 1e-12);
    check(post["lane_change_left"] > 0.99,
          "Bayes update recovers the unobserved mode from evidence");
    double s = 0.0; for (const auto& [_, v] : post) s += v;
    check(close_to(s, 1.0, 1e-9), "posterior normalized");

    // Degenerate likelihoods must fall back to the predictive, not divide by zero.
    ModeDistribution zero_like;
    for (const auto& m : modes) zero_like[m] = 0.0;
    auto fallback = update_mode_belief(prior, T, modes, zero_like, 0.0);
    double s2 = 0.0; for (const auto& [_, v] : fallback) s2 += v;
    check(close_to(s2, 1.0, 1e-9), "degenerate likelihoods fall back to a valid belief");
}

// ---------------------------------------------------------------------------
// 7. Markov sampling is WIRED: it must actually produce mode SWITCHES within a
//    single scenario, which i.i.d.-mode sampling can never do.
// ---------------------------------------------------------------------------
static void test_markov_sampling_switches_within_horizon() {
    auto lib = make_library();
    std::map<int, ObstacleState> obstacles;
    obstacles[0] = ObstacleState(5.0, 0.0, 0.5, 0.0);

    ModeHistory h(0, lib, 0);
    const char* seq[] = {"constant_velocity", "turn_left", "constant_velocity",
                         "turn_left", "constant_velocity", "turn_left"};
    for (int t = 0; t < 6; ++t) h.record_observation(t, seq[t]);
    std::map<int, ModeHistory> hists; hists[0] = h;

    std::mt19937 rng(12345);
    ModeBeliefConfig cfg;  // alpha=1, kappa=2
    auto scenarios = sample_scenarios_markov(
        obstacles, hists, nullptr, 15, 200, WeightType::FREQUENCY, cfg, &rng
    );
    check(scenarios.size() == 200, "markov sampler returns the requested budget");

    // Every trajectory carries a mode label; across 200 scenarios the sampler must
    // exercise more than one mode, including (via the prior) the unobserved one.
    std::map<std::string, int> labels;
    for (const auto& sc : scenarios) {
        for (const auto& [oid, traj] : sc.trajectories) labels[traj.mode_id]++;
    }
    check(labels.size() > 1, "markov sampling exercises multiple modes");
    check(labels.count("lane_change_left") > 0,
          "markov sampling reaches the NEVER-OBSERVED mode (prior + transition)");
}

// ---------------------------------------------------------------------------
// 8. Q* override: the caller-supplied belief must actually seed the chain.
// ---------------------------------------------------------------------------
static void test_qstar_override_seeds_chain() {
    auto lib = make_library();
    std::map<int, ObstacleState> obstacles;
    obstacles[0] = ObstacleState(5.0, 0.0, 0.5, 0.0);

    ModeHistory h(0, lib, 0);
    for (int t = 0; t < 30; ++t) h.record_observation(t, "constant_velocity");
    std::map<int, ModeHistory> hists; hists[0] = h;

    // Q* concentrated on a mode the FREQUENCY belief would barely sample.
    std::map<int, std::map<std::string, double>> qstar;
    qstar[0]["constant_velocity"] = 0.0;
    qstar[0]["lane_change_left"] = 1.0;
    qstar[0]["turn_left"] = 0.0;

    ModeBeliefConfig cfg;
    std::mt19937 rng_a(7), rng_b(7);
    auto with_q = sample_scenarios_markov(
        obstacles, hists, &qstar, 15, 200, WeightType::FREQUENCY, cfg, &rng_a);
    auto without_q = sample_scenarios_markov(
        obstacles, hists, nullptr, 15, 200, WeightType::FREQUENCY, cfg, &rng_b);

    auto count_label = [](const std::vector<Scenario>& v, const std::string& m) {
        int n = 0;
        for (const auto& sc : v)
            for (const auto& [oid, traj] : sc.trajectories)
                if (traj.mode_id == m) ++n;
        return n;
    };
    const int q_lcl = count_label(with_q, "lane_change_left");
    const int n_lcl = count_label(without_q, "lane_change_left");
    check(q_lcl > n_lcl,
          "Q* override seeds the chain (lane_change_left far more frequent under Q*)");
}

// ---------------------------------------------------------------------------
// 9. The prior selector is real and matches the cited constants.
// ---------------------------------------------------------------------------
static void test_dirichlet_prior_selector() {
    ModeBeliefConfig laplace;  laplace.prior = DirichletPrior::LAPLACE;
    ModeBeliefConfig kt;       kt.prior      = DirichletPrior::KRICHEVSKY_TROFIMOV;
    ModeBeliefConfig perks;    perks.prior   = DirichletPrior::PERKS;

    check(close_to(laplace.alpha(5), 1.0, 1e-12), "LAPLACE alpha == 1");
    check(close_to(kt.alpha(5), 0.5, 1e-12), "KRICHEVSKY_TROFIMOV alpha == 1/2");
    check(close_to(perks.alpha(5), 0.2, 1e-12), "PERKS alpha == 1/M");
    check(close_to(perks.alpha(3), 1.0 / 3.0, 1e-12), "PERKS tracks M");
    // LAPLACE/KT must NOT depend on M; PERKS must.
    check(close_to(laplace.alpha(3), laplace.alpha(50), 1e-12), "LAPLACE is M-invariant");
    check(!close_to(perks.alpha(3), perks.alpha(50), 1e-9), "PERKS is M-dependent by design");
}

// ---------------------------------------------------------------------------
// 10. kappa is DERIVED from theta and inverts E[T_ii] exactly, for any M.
//     This is the property the old hardcoded kappa=2 did not have.
// ---------------------------------------------------------------------------
static void test_sticky_kappa_derivation() {
    // E[T_ii] = (alpha + kappa) / (M*alpha + kappa) must equal theta exactly.
    for (int M : {2, 3, 5, 8}) {
        for (double theta : {0.55, 0.7, 0.8, 0.95}) {
            if (theta <= 1.0 / M) continue;
            for (auto pr : {DirichletPrior::LAPLACE,
                            DirichletPrior::KRICHEVSKY_TROFIMOV,
                            DirichletPrior::PERKS}) {
                ModeBeliefConfig cfg;
                cfg.prior = pr;
                cfg.self_persistence_prior = theta;
                const double a = cfg.alpha(M);
                const double k = cfg.kappa(M);
                const double e_tii = (a + k) / (M * a + k);
                check(close_to(e_tii, theta, 1e-10),
                      "kappa inverts E[T_ii]==theta (M=" + std::to_string(M) +
                      ", theta=" + std::to_string(theta) + ")");
            }
        }
    }

    // theta is M-INVARIANT in meaning: same theta => same E[T_ii] at any M.
    // (The old raw kappa=2 gave 0.60 at M=3 but 0.43 at M=5.)
    ModeBeliefConfig cfg;
    cfg.prior = DirichletPrior::LAPLACE;
    cfg.self_persistence_prior = 0.8;
    for (int M : {3, 5, 9}) {
        const double a = cfg.alpha(M), k = cfg.kappa(M);
        check(close_to((a + k) / (M * a + k), 0.8, 1e-10),
              "theta=0.8 means E[T_ii]=0.8 at M=" + std::to_string(M) + " (M-invariant)");
    }

    // Degenerate / disabling cases.
    ModeBeliefConfig off;
    off.self_persistence_prior = 0.0;
    check(close_to(off.kappa(5), 0.0, 1e-12), "theta=0 disables stickiness (kappa=0)");
    ModeBeliefConfig at_uniform;
    at_uniform.self_persistence_prior = 1.0 / 5.0;
    check(close_to(at_uniform.kappa(5), 0.0, 1e-12), "theta=1/M gives kappa=0 (uniform row)");
    ModeBeliefConfig degenerate;
    degenerate.self_persistence_prior = 1.0;
    check(close_to(degenerate.kappa(5), 0.0, 1e-12), "theta=1 is clamped, not infinite");
}

// ---------------------------------------------------------------------------
// 11. theta round-trips through the actual transition-matrix estimator on an
//     EMPTY history (where the row is the pure prior).
// ---------------------------------------------------------------------------
static void test_theta_roundtrip_through_estimator() {
    auto lib = make_library();  // M = 3
    ModeHistory h(0, lib, 0);
    h.record_observation(0, "constant_velocity");  // one obs; lcl row stays pure prior
    std::vector<std::string> modes = {"constant_velocity", "lane_change_left", "turn_left"};

    ModeBeliefConfig cfg;
    cfg.prior = DirichletPrior::KRICHEVSKY_TROFIMOV;
    cfg.self_persistence_prior = 0.75;
    const int M = 3;
    auto T = compute_mode_transition_matrix(h, modes, cfg.alpha(M), cfg.kappa(M));

    // Row 1 (lane_change_left) has zero observed transitions => pure prior.
    check(close_to(T(1, 1), 0.75, 1e-9),
          "unobserved row diagonal == theta exactly (prior round-trips)");
    check(close_to(T.row(1).sum(), 1.0, 1e-12), "prior row still stochastic");
}


// ---------------------------------------------------------------------------
// 12. Entropic allocator: FULL SUPPORT UNCONDITIONALLY (the pivot's core claim).
//     The raw W1-LP gives min_m q_m = 0 whenever the budget is slack (then
//     q* = e_argmax exactly), and usually when it binds. The entropic row softmax
//     is strictly positive, so q_min > 0 for EVERY tau > 0 => L <= 1/q_min < inf.
// ---------------------------------------------------------------------------
static void test_entropic_full_support() {
    std::vector<std::string> ids = {"a", "b", "c", "d"};
    std::map<std::string, double> nom, risk;
    for (size_t i = 0; i < ids.size(); ++i) { nom[ids[i]] = 0.25; }
    risk["a"] = 0.1; risk["b"] = 0.9; risk["c"] = 0.3; risk["d"] = 0.2;
    std::vector<std::vector<double>> D = {{0,1,2,3},{1,0,1,2},{2,1,0,1},{3,2,1,0}};

    for (double tau : {0.01, 0.05, 0.2, 1.0, 5.0}) {
        for (double rho : {0.05, 0.3, 10.0}) {   // 10.0 => budget deliberately slack
            auto e = solve_entropic_ot(nom, risk, D, ids, rho, tau);
            check(e.q_min > 0.0,
                  "entropic q_min > 0 (tau=" + std::to_string(tau) +
                  ", rho=" + std::to_string(rho) + ")");
            check(std::isfinite(e.likelihood_ratio_bound()),
                  "entropic L is FINITE -- certificate exists");
            double s = 0.0; for (const auto& kv : e.q) s += kv.second;
            check(close_to(s, 1.0, 1e-9), "entropic q normalized");
            check(e.transport_cost <= rho + 1e-6, "entropic respects the transport budget");
        }
    }
}

// ---------------------------------------------------------------------------
// 13. The tau frontier is monotone in the direction the theory requires:
//     protection <q_tau,r> NON-INCREASING in tau, certificate 1/q_min TIGHTENING.
// ---------------------------------------------------------------------------
static void test_entropic_frontier_monotone() {
    std::vector<std::string> ids = {"a", "b", "c", "d"};
    std::map<std::string, double> nom, risk;
    for (const auto& id : ids) nom[id] = 0.25;
    risk["a"] = 0.1; risk["b"] = 0.9; risk["c"] = 0.3; risk["d"] = 0.2;
    std::vector<std::vector<double>> D = {{0,1,2,3},{1,0,1,2},{2,1,0,1},{3,2,1,0}};
    const double rho = 0.3;

    double prev_prot = 1e300, prev_L = 1e300;
    for (double tau : {0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0}) {
        auto e = solve_entropic_ot(nom, risk, D, ids, rho, tau);
        check(e.expected_risk <= prev_prot + 1e-6,
              "protection non-increasing in tau (tau=" + std::to_string(tau) + ")");
        check(e.likelihood_ratio_bound() <= prev_L + 1e-6,
              "certificate 1/q_min tightens with tau (tau=" + std::to_string(tau) + ")");
        prev_prot = e.expected_risk; prev_L = e.likelihood_ratio_bound();
    }
    // tau -> large: rows tend to uniform, so q -> uniform and L -> M (tightest).
    auto e_big = solve_entropic_ot(nom, risk, D, ids, rho, 500.0);
    check(std::abs(e_big.likelihood_ratio_bound() - 4.0) < 0.2,
          "L -> M = 4 as tau -> inf (uniform limit, tightest possible bound)");
}

// ---------------------------------------------------------------------------
// 14. Slack budget: the RAW LP collapses to e_argmax (support 1, L = inf), while
//     the entropic allocator keeps full support. This is Theorem 2(i) vs Theorem 3.
// ---------------------------------------------------------------------------
static void test_slack_budget_collapse() {
    std::vector<std::string> ids = {"a", "b", "c", "d"};
    std::map<std::string, double> nom, risk;
    for (const auto& id : ids) nom[id] = 0.25;
    risk["a"] = 0.1; risk["b"] = 0.9; risk["c"] = 0.3; risk["d"] = 0.2;
    std::vector<std::vector<double>> D = {{0,1,2,3},{1,0,1,2},{2,1,0,1},{3,2,1,0}};

    // Slack budget, small tau: the entropic plan must approach e_argmax (= "b")
    // in protection while STILL retaining strictly positive mass everywhere.
    auto e = solve_entropic_ot(nom, risk, D, ids, /*rho=*/100.0, /*tau=*/0.01);
    check(e.q["b"] > 0.99, "slack budget + small tau concentrates on argmax r");
    check(e.q_min > 0.0, "...but q_min is STILL > 0 (certificate survives)");
    check(std::isfinite(e.likelihood_ratio_bound()), "...so L stays finite");
}

int main() {
    test_entropic_full_support();
    test_entropic_frontier_monotone();
    test_slack_budget_collapse();
    test_dirichlet_prior_selector();
    test_sticky_kappa_derivation();
    test_theta_roundtrip_through_estimator();
    test_dirichlet_posterior_mean();
    test_alpha_is_live();
    test_estimators_are_distinct();
    test_transition_matrix();
    test_prediction_reaches_unobserved_mode();
    test_bayes_update();
    test_markov_sampling_switches_within_horizon();
    test_qstar_override_seeds_chain();

    if (g_failures == 0) {
        std::cout << "All mode-belief tests passed.\n";
        return 0;
    }
    std::cout << g_failures << " mode-belief test(s) FAILED.\n";
    return 1;
}
