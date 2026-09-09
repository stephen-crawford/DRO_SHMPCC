// Regression coverage for the support-certification configuration contract:
// support-cap accounting, automatic sample sizing, YAML propagation, and the
// full-horizon Safe-Horizon semantics.
#include "experiment_config_yaml.hpp"
#include "scenario_sampler.hpp"

#include <cstdio>
#include <fstream>
#include <random>
#include <string>

using namespace dro_mpc;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", message);
    if (!condition) ++failures;
}

std::string write_overlay(const std::string& name, const std::string& contents) {
    const std::string path = "/tmp/" + name;
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

bool yaml_rejects(
    const std::string& name,
    const std::string& contents,
    const std::string& expected_message
) {
    const std::string path = write_overlay(name, contents);
    bool rejected = false;
    try {
        (void)yaml_config::load_experiment_config(path, /*strict=*/true);
    } catch (const std::exception& e) {
        rejected = std::string(e.what()).find(expected_message) != std::string::npos;
    }
    std::remove(path.c_str());
    return rejected;
}

}  // namespace

int main() {
    std::printf("=== Support-certification configuration ===\n");

    {
        RuntimeConfig cfg;
        check(cfg.mpc.constraints.support_cap_n_bar == 6,
              "the fallback n_bar matches scenario_module's default of 6");
        check(cfg.support_limit() == 6,
              "the default Safe-Horizon support cap is n_bar + removal_count");
    }

    {
        RuntimeConfig cfg;
        cfg.mpc.horizon = 17;
        check(cfg.mpc.horizon == 17,
              "the configured MPC horizon is the complete certification horizon");
    }

    {
        RuntimeConfig cfg;
        cfg.mpc.constraints.support_cap_n_bar = 2;
        cfg.mpc.constraints.scenario_removal_budget = 3;

        check(cfg.support_limit() == 5,
              "support cap is n_bar plus the conservative removal budget");
        check(cfg.compute_required_scenarios() ==
                  cfg.compute_required_scenarios(/*n_bar=*/2, /*R=*/3),
              "configured sample sizing uses the total support cap");

        cfg.mpc.sampling.automatically_compute_sample_size = true;
        cfg.mpc.sampling.num_scenarios = 7;
        cfg.normalize();
        check(cfg.mpc.sampling.num_scenarios == cfg.compute_required_scenarios(),
              "automatic sizing replaces static S with the certified sample count");

        cfg.mpc.sampling.set_manual_sample_count(19);
        cfg.normalize();
        check(cfg.mpc.sampling.num_scenarios == 19,
              "explicit static S is preserved when automatic sizing is disabled");
    }

    {
        const std::string manual_path = write_overlay(
            "dro_shmpcc_safe_horizon_manual.yaml",
            "safe_horizon_enabled: true\n"
            "support_cap_n_bar: 2\n"
            "scenario_removal_budget: 3\n"
            "automatically_compute_sample_size: false\n"
            "num_scenarios: 23\n");
        const ExperimentConfig cfg =
            yaml_config::load_experiment_config(manual_path, /*strict=*/true);
        const RuntimeConfig runtime = cfg.to_scenario_mpc_config();

        check(cfg.mpc.safe_horizon_enabled,
              "safe-horizon enablement is inherited from YAML");
        check(runtime.support_limit() == 5,
              "support cap and removal budget compose the total support cap");
        check(!runtime.mpc.sampling.automatically_compute_sample_size &&
                  runtime.mpc.sampling.num_scenarios == 23,
              "manual YAML S survives config normalization and runtime conversion");
        std::remove(manual_path.c_str());
    }

    {
        const std::string automatic_path = write_overlay(
            "dro_shmpcc_safe_horizon_automatic.yaml",
            "support_cap_n_bar: 2\n"
            "scenario_removal_budget: 3\n"
            "automatically_compute_sample_size: true\n"
            "num_scenarios: 7\n");
        const ExperimentConfig cfg =
            yaml_config::load_experiment_config(automatic_path, /*strict=*/true);
        const RuntimeConfig runtime = cfg.to_scenario_mpc_config();

        check(runtime.mpc.sampling.automatically_compute_sample_size,
              "automatic sample sizing is inherited from YAML");
        check(cfg.mpc.sampling.num_scenarios == runtime.compute_required_scenarios() &&
                  runtime.mpc.sampling.num_scenarios == runtime.compute_required_scenarios(),
              "automatic YAML S is derived once and preserved throughout its lifecycle");
        std::remove(automatic_path.c_str());
    }

    {
        // An overlay that supplies S but no automatic-sizing policy is an
        // intentional manual sweep, even when the base configuration enables
        // automatic Safe-Horizon sizing.
        const std::string implicit_manual_path = write_overlay(
            "dro_shmpcc_safe_horizon_implicit_manual.yaml",
            "num_scenarios: 29\n");
        const ExperimentConfig cfg =
            yaml_config::load_experiment_config(implicit_manual_path, /*strict=*/true);
        const RuntimeConfig runtime = cfg.to_scenario_mpc_config();
        check(!runtime.mpc.sampling.automatically_compute_sample_size &&
                  runtime.mpc.sampling.num_scenarios == 29,
              "a YAML num_scenarios override selects manual sizing unless auto sizing is explicit");
        std::remove(implicit_manual_path.c_str());
    }

    {
        const ExperimentConfig defaults = yaml_config::load_experiment_config();
        const RuntimeConfig runtime = defaults.to_scenario_mpc_config();
        check(defaults.mpc.constraints.support_cap_n_bar == 6 &&
                  runtime.mpc.constraints.support_cap_n_bar == 6,
              "default YAML inherits scenario_module's n_bar default of 6");
        check(defaults.mpc.sampling.automatically_compute_sample_size,
              "default YAML enables reference-style automatic sample sizing");
        check(defaults.mpc.sampling.num_scenarios == runtime.compute_required_scenarios(),
              "default YAML stores its certified effective scenario count");
    }

    {
        RuntimeConfig cfg;
        bool valid_default_horizon = true;
        try {
            cfg.validate();
            cfg.dro.solver.radius_calibration.risk_horizon = 1;
            cfg.validate();
        } catch (const std::exception&) {
            valid_default_horizon = false;
        }
        check(valid_default_horizon,
              "DRO count settings accept positive counts and -1 or positive risk horizons");

        cfg.dro.solver.radius_calibration.joint_risk_samples = 0;
        bool rejected = false;
        try {
            cfg.validate();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "RuntimeConfig rejects nonpositive joint-risk sample counts");

        cfg = RuntimeConfig{};
        cfg.dro.solver.radius_calibration.mixture_sequence_samples = -1;
        rejected = false;
        try {
            cfg.validate();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "RuntimeConfig rejects nonpositive mixture sequence counts");

        for (const int invalid_horizon : {0, -2}) {
            cfg = RuntimeConfig{};
            cfg.dro.solver.radius_calibration.risk_horizon = invalid_horizon;
            rejected = false;
            try {
                cfg.validate();
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            check(rejected, "RuntimeConfig rejects invalid risk-horizon sentinels");
        }
    }

    {
        check(yaml_rejects(
                  "dro_shmpcc_invalid_joint_samples.yaml",
                  "joint_risk_samples: 0\n",
                  "joint_risk_samples must be positive"),
              "YAML rejects a nonpositive joint-risk sample count");
        check(yaml_rejects(
                  "dro_shmpcc_invalid_mixture_samples.yaml",
                  "mixture_sequence_samples: 0\n",
                  "mixture_sequence_samples must be positive"),
              "YAML rejects a nonpositive mixture sequence count");
        check(yaml_rejects(
                  "dro_shmpcc_invalid_risk_horizon.yaml",
                  "risk_horizon: 0\n",
                  "risk_horizon must be -1 or positive"),
              "YAML rejects zero as a risk-horizon override");
    }

    {
        const std::string removed_path = write_overlay(
            "dro_shmpcc_removed_safe_horizon_option.yaml",
            "safe_horizon_min: 3\n");
        bool rejected = false;
        try {
            (void)yaml_config::load_experiment_config(removed_path);
        } catch (const std::exception&) {
            rejected = true;
        }
        check(rejected,
              "removed temporal safe-horizon options fail explicitly instead of being ignored");
        std::remove(removed_path.c_str());
    }

    {
        const std::string invalid_enum_path = write_overlay(
            "dro_shmpcc_invalid_enum.yaml",
            "environment: not_a_real_environment\n");
        bool rejected = false;
        try {
            (void)yaml_config::load_experiment_config(invalid_enum_path, true);
        } catch (const std::exception&) {
            rejected = true;
        }
        check(rejected,
              "unknown YAML enum values fail explicitly instead of silently selecting a default");
        std::remove(invalid_enum_path.c_str());

        const std::string invalid_bool_path = write_overlay(
            "dro_shmpcc_invalid_bool.yaml",
            "dro_enabled: perhaps\n");
        rejected = false;
        try {
            (void)yaml_config::load_experiment_config(invalid_bool_path, true);
        } catch (const std::exception&) {
            rejected = true;
        }
        check(rejected,
              "unknown YAML boolean values fail explicitly instead of silently selecting a default");
        std::remove(invalid_bool_path.c_str());
    }

    {
        // Certification identifies a joint sample by scenario_id.  If the
        // controller ever augments a static sample set, the new draws must
        // have fresh IDs rather than collapsing into existing support entries.
        std::mt19937 rng(7);
        const std::map<int, ObstacleState> no_obstacles;
        const std::map<int, ModeHistory> no_histories;
        const auto first = sample_scenarios(
            no_obstacles, no_histories, nullptr, 1, 3, {}, nullptr, &rng, 0);
        const auto second = sample_scenarios(
            no_obstacles, no_histories, nullptr, 1, 2, {}, nullptr, &rng, 3);
        check(first.size() == 3 && second.size() == 2 &&
                  first[0].scenario_id == 0 && first[2].scenario_id == 2 &&
                  second[0].scenario_id == 3 && second[1].scenario_id == 4,
              "appended samples receive unique joint scenario IDs");
    }

    std::printf("\n%s (%d checks failed)\n",
                failures == 0 ? "ALL SUPPORT-CONFIGURATION TESTS PASSED"
                              : "SUPPORT-CONFIGURATION TESTS FAILED",
                failures);
    return failures == 0 ? 0 : 1;
}
