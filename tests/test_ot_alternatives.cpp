/**
 * @file test_ot_alternatives.cpp
 * @brief Test 5 alternative OT approaches vs original setups.
 *
 * Compares collision rate and mode coverage (with Wilson 95% CIs) for:
 *  - Base (no OT, no DRO)
 *  - DRO(inj) (DRO injection only)
 *  - DRO(inj)+SH (DRO + safe horizon)
 *  - OT+DRO(inj)+SH original (OT as sampling weights — interference)
 *  - Alt1: DRO(inj)+OT-postDRO (OT sampling, freq for DRO Q*)
 *  - Alt2: DRO(inj)+OT-uncertainty (OT scales safety margin)
 *  - Alt3: DRO(inj)+OT-switch (OT detects mode switches)
 *  - Alt4: DRO(inj)+OT-scenario (OT prediction as extra scenario)
 *  - Alt5: DRO(inj)+OT-dynamics (OT learns mode dynamics)
 *
 * Usage: ./test_ot_alternatives [max_rollouts] [--4obs]
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <map>
#include <filesystem>

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "wasserstein_dro.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

static const std::string OUTPUT_DIR = "paper_figures/";

// ============================================================================
// Strategy configuration
// ============================================================================

struct OTAltConfig {
    std::string name;
    bool use_ot;
    bool enable_dro;
    InjectionMode injection_mode;
    bool safe_horizon;
    OTUsageMode ot_usage_mode;
    DRONominalSource dro_nominal_source;
};

static std::vector<OTAltConfig> build_configs() {
    std::vector<OTAltConfig> cfgs;

    // === Reference strategies ===
    cfgs.push_back({"Base",
        false, false, InjectionMode::NONE, false,
        OTUsageMode::DISABLED, DRONominalSource::AUTO});

    cfgs.push_back({"DRO(inj)",
        false, true, InjectionMode::DRO, false,
        OTUsageMode::DISABLED, DRONominalSource::AUTO});

    cfgs.push_back({"DRO(inj)+SH",
        false, true, InjectionMode::DRO, true,
        OTUsageMode::DISABLED, DRONominalSource::AUTO});

    // === Original OT+DRO (with interference) ===
    cfgs.push_back({"OT+DRO(inj)+SH [original]",
        true, true, InjectionMode::DRO, true,
        OTUsageMode::SAMPLING_WEIGHTS, DRONominalSource::AUTO});

    // === Alt 1: Post-DRO sampling ===
    // OT weights for sampling, frequency weights for DRO Q*
    cfgs.push_back({"DRO(inj)+OT-postDRO",
        true, true, InjectionMode::DRO, false,
        OTUsageMode::POST_DRO_SAMPLING, DRONominalSource::FREQUENCY});

    // === Alt 2: OT uncertainty margin ===
    // OT prediction error → adjusted safety margin
    cfgs.push_back({"DRO(inj)+OT-uncertainty",
        true, true, InjectionMode::DRO, false,
        OTUsageMode::UNCERTAINTY_MARGIN, DRONominalSource::AUTO});

    // === Alt 3: OT switch detection ===
    // W2 spikes → extra mode observations
    cfgs.push_back({"DRO(inj)+OT-switch",
        true, true, InjectionMode::DRO, false,
        OTUsageMode::SWITCH_DETECTION, DRONominalSource::AUTO});

    // === Alt 4: OT extra scenario ===
    // OT trajectory prediction → injected scenario constraint
    cfgs.push_back({"DRO(inj)+OT-scenario",
        true, true, InjectionMode::DRO, false,
        OTUsageMode::EXTRA_SCENARIO, DRONominalSource::AUTO});

    // === Alt 5: OT dynamics learning ===
    // OT learns mode (b, G), updates controller
    cfgs.push_back({"DRO(inj)+OT-dynamics",
        true, true, InjectionMode::DRO, false,
        OTUsageMode::DYNAMICS_LEARNING, DRONominalSource::AUTO});

    return cfgs;
}

// ============================================================================
// Rollout results
// ============================================================================

struct StrategyResult {
    std::string name;
    int n = 0;
    int collisions = 0;
    int missed_modes = 0;
    int total_mode_checks = 0;
    int rare_mode_active = 0;
    int rare_mode_missed = 0;
    double total_solve_ms = 0.0;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // Parse arguments
    int max_rollouts = 2000;
    bool four_obs = true;  // default to 4-obstacle

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--4obs") four_obs = true;
        else if (arg == "--1obs") four_obs = false;
        else max_rollouts = std::stoi(arg);
    }

    const int N_OBS = four_obs ? 4 : 1;
    const int NUM_SCENARIOS = DEFAULT_BASE_SCENARIOS;
    const int ROLLOUT_STEPS = DEFAULT_ROLLOUT_STEPS;
    const int NUM_DISCS = 1;
    const double VEHICLE_LENGTH = 1.5;
    const double SWITCH_PROB = 0.1;
    const double RARE_PROB = 0.05;

    const std::vector<std::string> OBS_MODES =
        {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    const std::string RARE_MODE = "lane_change_left";

    auto configs = build_configs();

    std::cout << "=== OT Alternative Approaches Test ===\n";
    std::cout << "Obstacles: " << N_OBS << ", Rollouts: " << max_rollouts
              << ", Scenarios: " << NUM_SCENARIOS << "\n";
    std::cout << "Strategies: " << configs.size() << "\n\n";

    // Master seeds for multi-seed cycling
    const std::vector<unsigned> master_seeds = {42, 1042, 2042, 3042, 4042};

    // Run all strategies
    std::vector<StrategyResult> results(configs.size());
    for (size_t ci = 0; ci < configs.size(); ++ci) {
        results[ci].name = configs[ci].name;
    }

    auto t0 = std::chrono::steady_clock::now();

    for (int r = 0; r < max_rollouts; ++r) {
        unsigned ms = master_seeds[r % master_seeds.size()];
        SeedBundle seeds = derive_seeds(ms, r / static_cast<int>(master_seeds.size()));

        if (r % 200 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            std::cout << "  Rollout " << r << "/" << max_rollouts;
            if (r > 0) {
                double per = elapsed / r;
                double remaining = per * (max_rollouts - r) * configs.size();
                // Actually time is per (rollout * configs), adjust:
                remaining = per * (max_rollouts - r);
                std::cout << " (~" << std::fixed << std::setprecision(0)
                          << remaining << "s remaining)";
            }
            std::cout << "\n";
        }

        for (size_t ci = 0; ci < configs.size(); ++ci) {
            const auto& cfg = configs[ci];

            ExperimentConfig ecfg;
            ecfg.horizon = DEFAULT_HORIZON;
            ecfg.num_scenarios = NUM_SCENARIOS;
            ecfg.switch_prob = SWITCH_PROB;
            ecfg.rollout_steps = ROLLOUT_STEPS;
            ecfg.obs_modes = OBS_MODES;
            ecfg.rare_mode = RARE_MODE;
            ecfg.rare_switch_prob = RARE_PROB;
            ecfg.num_discs = NUM_DISCS;
            ecfg.vehicle_length = VEHICLE_LENGTH;
            ecfg.num_obstacles = N_OBS;
            ecfg.obstacles_per_class = 1;
            if (N_OBS == 4) ecfg.obs_arc_fractions = OBS_ARC_FRACS_4;
            ecfg.path_completion_termination = true;
            ecfg.path_completion_fraction = PATH_COMPLETE_FRAC;

            // Strategy-specific settings
            ecfg.weight_type = cfg.use_ot ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
            ecfg.enable_dro = cfg.enable_dro;
            ecfg.injection_mode = cfg.injection_mode;
            ecfg.safe_horizon_enabled = cfg.safe_horizon;
            ecfg.safe_horizon_min = 3;
            ecfg.use_ot_predictor = cfg.use_ot;
            ecfg.ot_usage_mode = cfg.ot_usage_mode;
            ecfg.dro_nominal_source = cfg.dro_nominal_source;
            ecfg.method_name = cfg.name;
            ecfg.ablation = AblationVariant::NO_INJECTION;

            auto rec = run_experiment_rollout(ecfg, seeds.env);

            auto& res = results[ci];
            res.n++;
            if (rec.collision) res.collisions++;
            res.missed_modes += rec.missed_mode_steps;
            res.total_mode_checks += rec.total_mode_checks;
            res.rare_mode_active += rec.rare_mode_active;
            res.rare_mode_missed += rec.rare_mode_missed;
            res.total_solve_ms += rec.avg_solve_ms;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t1 - t0).count();

    // ============================================================================
    // Print results
    // ============================================================================

    std::cout << "\n=== Results (" << max_rollouts << " rollouts, "
              << N_OBS << " obstacles) ===\n\n";

    // Header
    std::cout << std::left << std::setw(30) << "Strategy"
              << std::right << std::setw(10) << "Coll%"
              << std::setw(18) << "Coll CI"
              << std::setw(10) << "ModeCov%"
              << std::setw(18) << "ModeCov CI"
              << std::setw(10) << "RareCov%"
              << std::setw(10) << "Solve(ms)"
              << "\n";
    std::cout << std::string(106, '-') << "\n";

    // Get Base CI for reference
    double base_coll_hi = 1.0;
    double base_mode_lo = 0.0;
    for (const auto& res : results) {
        if (res.name == "Base") {
            auto [lo, hi] = wilson_ci(res.collisions, res.n);
            base_coll_hi = hi;
            int mode_covered = res.total_mode_checks - res.missed_modes;
            auto [mlo, mhi] = wilson_ci(mode_covered, res.total_mode_checks);
            base_mode_lo = mlo;
        }
    }

    for (const auto& res : results) {
        double coll_rate = static_cast<double>(res.collisions) / res.n;
        auto [coll_lo, coll_hi] = wilson_ci(res.collisions, res.n);

        int mode_covered = res.total_mode_checks - res.missed_modes;
        double mode_cov = res.total_mode_checks > 0
            ? static_cast<double>(mode_covered) / res.total_mode_checks : 0.0;
        auto [mc_lo, mc_hi] = wilson_ci(mode_covered, res.total_mode_checks);

        int rare_covered = res.rare_mode_active - res.rare_mode_missed;
        double rare_cov = res.rare_mode_active > 0
            ? static_cast<double>(rare_covered) / res.rare_mode_active : 0.0;

        double avg_solve = res.n > 0 ? res.total_solve_ms / res.n : 0.0;

        // CI separation indicators
        std::string coll_flag = "";
        std::string mode_flag = "";
        if (res.name != "Base") {
            if (coll_hi < base_coll_hi) coll_flag = " *";
            if (mc_lo > base_mode_lo) mode_flag = " *";
        }

        std::cout << std::left << std::setw(30) << res.name
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(8) << (coll_rate * 100) << "%"
                  << " [" << std::setprecision(1) << (coll_lo * 100) << ","
                  << std::setprecision(1) << (coll_hi * 100) << "]" << coll_flag
                  << std::setw(8) << (mode_cov * 100) << "%"
                  << " [" << std::setprecision(1) << (mc_lo * 100) << ","
                  << std::setprecision(1) << (mc_hi * 100) << "]" << mode_flag
                  << std::setw(8) << (rare_cov * 100) << "%"
                  << std::setw(8) << std::setprecision(2) << avg_solve
                  << "\n";
    }

    std::cout << "\n* = CI separated from Base (better)\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(1) << total_sec << "s\n";

    // ============================================================================
    // Write CSV
    // ============================================================================

    fs::create_directories(OUTPUT_DIR);
    std::string csv_path = OUTPUT_DIR + "ot_alternatives_"
        + std::to_string(N_OBS) + "obs.csv";
    std::ofstream csv(csv_path);
    csv << "config,n,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "mode_coverage,mode_ci_lo,mode_ci_hi,"
        << "rare_coverage,avg_solve_ms\n";

    for (const auto& res : results) {
        double coll_rate = static_cast<double>(res.collisions) / res.n;
        auto [coll_lo, coll_hi] = wilson_ci(res.collisions, res.n);

        int mode_covered = res.total_mode_checks - res.missed_modes;
        double mode_cov = res.total_mode_checks > 0
            ? static_cast<double>(mode_covered) / res.total_mode_checks : 0.0;
        auto [mc_lo, mc_hi] = wilson_ci(mode_covered, res.total_mode_checks);

        int rare_covered = res.rare_mode_active - res.rare_mode_missed;
        double rare_cov = res.rare_mode_active > 0
            ? static_cast<double>(rare_covered) / res.rare_mode_active : 0.0;

        double avg_solve = res.n > 0 ? res.total_solve_ms / res.n : 0.0;

        csv << res.name << "," << res.n << ","
            << std::fixed << std::setprecision(6)
            << coll_rate << "," << coll_lo << "," << coll_hi << ","
            << mode_cov << "," << mc_lo << "," << mc_hi << ","
            << rare_cov << "," << std::setprecision(4) << avg_solve << "\n";
    }
    csv.close();
    std::cout << "\nCSV written to " << csv_path << "\n";

    // ============================================================================
    // Summary analysis
    // ============================================================================

    std::cout << "\n=== Analysis ===\n";
    std::cout << "Strategies beating Base on BOTH collision AND mode coverage "
              << "(non-overlapping CIs):\n";

    for (const auto& res : results) {
        if (res.name == "Base") continue;

        double coll_rate = static_cast<double>(res.collisions) / res.n;
        auto [coll_lo, coll_hi] = wilson_ci(res.collisions, res.n);

        int mode_covered = res.total_mode_checks - res.missed_modes;
        double mode_cov = res.total_mode_checks > 0
            ? static_cast<double>(mode_covered) / res.total_mode_checks : 0.0;
        auto [mc_lo, mc_hi] = wilson_ci(mode_covered, res.total_mode_checks);

        bool coll_better = coll_hi < base_coll_hi;
        bool mode_better = mc_lo > base_mode_lo;

        if (coll_better && mode_better) {
            int rare_covered = res.rare_mode_active - res.rare_mode_missed;
            double rare_cov = res.rare_mode_active > 0
                ? static_cast<double>(rare_covered) / res.rare_mode_active : 0.0;

            std::cout << "  " << res.name << ": "
                      << std::fixed << std::setprecision(1)
                      << (coll_rate * 100) << "% coll [" << (coll_lo * 100) << ","
                      << (coll_hi * 100) << "], "
                      << (mode_cov * 100) << "% mode [" << (mc_lo * 100) << ","
                      << (mc_hi * 100) << "], "
                      << (rare_cov * 100) << "% rare\n";
        }
    }

    // Compare alternatives against DRO(inj) baseline
    std::cout << "\nComparison against DRO(inj) baseline:\n";
    double dro_coll_lo = 0, dro_coll_hi = 1, dro_mc_lo = 0, dro_mc_hi = 1;
    for (const auto& res : results) {
        if (res.name == "DRO(inj)") {
            auto [clo, chi] = wilson_ci(res.collisions, res.n);
            dro_coll_lo = clo; dro_coll_hi = chi;
            int mc = res.total_mode_checks - res.missed_modes;
            auto [mlo, mhi] = wilson_ci(mc, res.total_mode_checks);
            dro_mc_lo = mlo; dro_mc_hi = mhi;
        }
    }

    for (const auto& res : results) {
        if (res.name == "Base" || res.name == "DRO(inj)" ||
            res.name == "DRO(inj)+SH") continue;

        auto [coll_lo, coll_hi] = wilson_ci(res.collisions, res.n);
        int mc = res.total_mode_checks - res.missed_modes;
        auto [mc_lo, mc_hi] = wilson_ci(mc, res.total_mode_checks);

        std::string coll_vs = "overlap";
        if (coll_hi < dro_coll_lo) coll_vs = "BETTER";
        else if (coll_lo > dro_coll_hi) coll_vs = "WORSE";

        std::string mode_vs = "overlap";
        if (mc_lo > dro_mc_hi) mode_vs = "BETTER";
        else if (mc_hi < dro_mc_lo) mode_vs = "WORSE";

        std::cout << "  " << std::left << std::setw(30) << res.name
                  << " coll: " << std::setw(8) << coll_vs
                  << " mode: " << mode_vs << "\n";
    }

    return 0;
}
