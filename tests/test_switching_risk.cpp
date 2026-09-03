// Switching-aware per-mode risk r[m] for Markov-jump obstacles.
//
// Covers the MIXTURE_* family (semi-analytic risk measure of the joint
// (mode sequence, noise) law) and the switching-aware JOINT_* reference, and pins
// down the reductions that make the taxonomy coherent:
//
//   (A) No transition matrix => the mixture has ONE component, so
//       MIXTURE_VAR == SURROGATE_VAR and MIXTURE_CVAR == SURROGATE_CVAR exactly.
//       (The K=1 Rockafellar-Uryasev formula must reproduce cvar_clamped_gaussian,
//       including its mu + z*sigma < 0 branch.)
//   (B) transition = I => the chain never leaves its start mode, so every
//       switching-aware estimator must reproduce its held-mode counterpart.
//   (C) Coherence / ordering: CVaR >= VaR on the same mixture, and the LEGACY
//       E_seq[max VaR] estimator UNDERSTATES the coherent joint risk -- the
//       inequality this whole family exists to fix.
//   (D) Determinism: common random numbers make r[m] identical across repeated
//       calls, so the W1 reweighting does not jitter between solves.
#include "dro.hpp"
#include "dynamics.hpp"
#include "types.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

using namespace dro_mpc;

static int fails = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++fails;
}
static bool approx(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

namespace {

struct Fixture {
    ObstacleState obs;
    std::map<std::string, ModeModel> modes;
    std::vector<std::string> ids;
    std::vector<EgoState> ego;
    double R = 1.2;
    int horizon = 12;
    int discs = 1;
    double length = 4.0;
};

Fixture make_fixture() {
    Fixture f;
    f.modes = create_obstacle_mode_models(0.1);
    for (const auto& [id, _] : f.modes) f.ids.push_back(id);

    // Obstacle approaching the ego corridor from the side.
    f.obs = ObstacleState(6.0, 1.6, -1.0, -0.35);

    // Ego drives straight along +x; the obstacle closes in over the horizon.
    for (int k = 0; k <= f.horizon; ++k) {
        EgoState e;
        e.x = 0.4 * k;
        e.y = 0.0;
        e.theta = 0.0;
        e.v = 4.0;
        f.ego.push_back(e);
    }
    return f;
}

// Sticky-ish chain: strong self-persistence, uniform leakage elsewhere.
Eigen::MatrixXd sticky_chain(int M, double theta) {
    Eigen::MatrixXd T(M, M);
    const double off = (1.0 - theta) / std::max(1, M - 1);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j) T(i, j) = (i == j) ? theta : off;
    return T;
}

std::map<std::string, double> risk_for(
    DRORiskMeasure measure, const Fixture& f, const Eigen::MatrixXd* T,
    int mixture_samples = 512, int joint_samples = 4000)
{
    DROConfig cfg;
    cfg.radius_calibration.risk_measure = measure;
    cfg.radius_calibration.mixture_sequence_samples = mixture_samples;
    cfg.radius_calibration.joint_risk_samples = joint_samples;
    DRO dro(cfg);

    std::map<std::string, double> nominal;
    for (const auto& id : f.ids) nominal[id] = 1.0 / static_cast<double>(f.ids.size());

    DROResult res = dro.compute_worst_case_weights(
        nominal, f.obs, f.modes, f.ego, f.horizon,
        /*ego_r=*/0.5, /*obs_r=*/0.5, /*margin=*/0.2,
        /*risk_horizon=*/-1, f.discs, f.length, T);
    return res.risk_per_mode;
}

double mean_of(const std::map<std::string, double>& m) {
    if (m.empty()) return 0.0;
    double s = 0.0;
    for (const auto& [_, v] : m) s += v;
    return s / static_cast<double>(m.size());
}

}  // namespace

int main() {
    Fixture f = make_fixture();
    const int M = static_cast<int>(f.ids.size());
    std::printf("fixture: %d modes, horizon %d\n", M, f.horizon);

    std::printf("\n=== (A) no chain => one-component mixture == surrogate ===\n");
    {
        auto sur_var  = risk_for(DRORiskMeasure::SURROGATE_VAR,  f, nullptr);
        auto mix_var  = risk_for(DRORiskMeasure::MIXTURE_VAR,    f, nullptr);
        auto sur_cvar = risk_for(DRORiskMeasure::SURROGATE_CVAR, f, nullptr);
        auto mix_cvar = risk_for(DRORiskMeasure::MIXTURE_CVAR,   f, nullptr);

        bool var_ok = true, cvar_ok = true;
        for (const auto& id : f.ids) {
            var_ok  = var_ok  && approx(sur_var.at(id),  mix_var.at(id));
            cvar_ok = cvar_ok && approx(sur_cvar.at(id), mix_cvar.at(id));
        }
        check(var_ok,  "MIXTURE_VAR reduces to SURROGATE_VAR when there is no chain");
        check(cvar_ok, "MIXTURE_CVAR reduces to SURROGATE_CVAR (K=1 R-U == cvar_clamped_gaussian)");
    }

    std::printf("\n=== (B) transition = I => switching reduces to held-mode ===\n");
    {
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(M, M);

        auto mix_held = risk_for(DRORiskMeasure::MIXTURE_CVAR, f, nullptr);
        auto mix_ident = risk_for(DRORiskMeasure::MIXTURE_CVAR, f, &I);
        bool mix_ok = true;
        for (const auto& id : f.ids) mix_ok = mix_ok && approx(mix_held.at(id), mix_ident.at(id));
        check(mix_ok, "MIXTURE_CVAR at T=I equals the held-mode value");

        auto joint_held  = risk_for(DRORiskMeasure::JOINT_CVAR, f, nullptr);
        auto joint_ident = risk_for(DRORiskMeasure::JOINT_CVAR, f, &I);
        bool joint_ok = true;
        for (const auto& id : f.ids)
            joint_ok = joint_ok && approx(joint_held.at(id), joint_ident.at(id), 1e-12);
        check(joint_ok, "JOINT_CVAR at T=I equals the held-mode JOINT_CVAR (noise stream untouched)");
    }

    std::printf("\n=== (C) coherence + the understatement the family fixes ===\n");
    {
        Eigen::MatrixXd T = sticky_chain(M, 0.75);

        auto legacy    = risk_for(DRORiskMeasure::SURROGATE_VAR, f, &T);  // E_seq[max VaR]
        auto mix_var   = risk_for(DRORiskMeasure::MIXTURE_VAR,   f, &T);
        auto mix_cvar  = risk_for(DRORiskMeasure::MIXTURE_CVAR,  f, &T);
        auto joint_cvar= risk_for(DRORiskMeasure::JOINT_CVAR,    f, &T);

        bool cvar_ge_var = true;
        for (const auto& id : f.ids) cvar_ge_var = cvar_ge_var && (mix_cvar.at(id) >= mix_var.at(id) - 1e-12);
        check(cvar_ge_var, "CVaR >= VaR on the same sequence mixture (coherence sanity)");

        bool legacy_understates = true;
        for (const auto& id : f.ids)
            legacy_understates = legacy_understates && (legacy.at(id) <= mix_cvar.at(id) + 1e-12);
        check(legacy_understates,
              "legacy E_seq[max VaR] <= coherent MIXTURE_CVaR on every mode (understatement)");

        std::printf("    mean r  legacy=%.4f  mixture_var=%.4f  mixture_cvar=%.4f  joint_cvar=%.4f\n",
                    mean_of(legacy), mean_of(mix_var), mean_of(mix_cvar), mean_of(joint_cvar));

        // The mixture is a surrogate for the joint reference; it should not be wildly
        // off. Compared on the SPREAD (max-min), which is all the W1 LP consumes.
        auto spread = [&](const std::map<std::string, double>& m) {
            double lo = 1e18, hi = -1e18;
            for (const auto& [_, v] : m) { lo = std::min(lo, v); hi = std::max(hi, v); }
            return hi - lo;
        };
        std::printf("    spread  mixture_cvar=%.4f  joint_cvar=%.4f\n",
                    spread(mix_cvar), spread(joint_cvar));
        check(spread(mix_cvar) > 0.0 && spread(joint_cvar) > 0.0,
              "both estimators separate the modes (non-degenerate risk vector)");
    }

    std::printf("\n=== (D) determinism under common random numbers ===\n");
    {
        Eigen::MatrixXd T = sticky_chain(M, 0.75);
        auto a = risk_for(DRORiskMeasure::MIXTURE_CVAR, f, &T);
        auto b = risk_for(DRORiskMeasure::MIXTURE_CVAR, f, &T);
        auto c = risk_for(DRORiskMeasure::JOINT_CVAR, f, &T);
        auto d = risk_for(DRORiskMeasure::JOINT_CVAR, f, &T);
        bool mix_det = true, joint_det = true;
        for (const auto& id : f.ids) {
            mix_det   = mix_det   && approx(a.at(id), b.at(id), 1e-15);
            joint_det = joint_det && approx(c.at(id), d.at(id), 1e-15);
        }
        check(mix_det,   "MIXTURE_CVAR is bit-stable across repeated calls");
        check(joint_det, "JOINT_CVAR is bit-stable across repeated calls");
    }

    std::printf("\n%s (%d checks failed)\n",
                fails == 0 ? "ALL SWITCHING-RISK TESTS PASSED" : "SOME TESTS FAILED", fails);
    return fails == 0 ? 0 : 1;
}
