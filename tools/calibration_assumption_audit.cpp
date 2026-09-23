// Monte Carlo assumption sensitivity through the production finite-sample API.
// This does not alter the controller's history policy or calibration formulas.
#include "wasserstein_radius_calibration.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: calibration_assumption_audit OUTPUT.csv\n";
        return 2;
    }
    if (std::filesystem::exists(argv[1])) {
        std::cerr << "refusing to overwrite an existing audit\n";
        return 2;
    }
    std::ofstream out(argv[1]);
    if (!out) throw std::runtime_error("cannot open audit output");
    out << std::setprecision(17)
        << "law,observations,replications,seed,beta,target_mode_probability,cp_region_hits,ball_hits,cp_coverage,ball_coverage,mean_radius\n";
    constexpr int replications = 2000;
    constexpr unsigned seed = 20260922;
    const std::vector<std::vector<double>> cost{{0.,1.},{1.,0.}};
    for (int n : {20,100}) {
        std::map<int,dro_mpc::FiniteSampleWassersteinRadius> cache;
        for (const std::string law : {"iid_stationary", "stationary_blocks_of_10", "mid_history_distribution_shift"}) {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> uniform(0.,1.);
            const double target = law=="mid_history_distribution_shift" ? .4 : .1;
            int cp_hits=0, ball_hits=0;
            double radius_sum=0.;
            for (int trial=0; trial<replications; ++trial) {
                int count=0;
                bool held=false;
                for (int i=0; i<n; ++i) {
                    const double probability=law=="mid_history_distribution_shift" && i>=n/2 ? .4 : .1;
                    if (law!="stationary_blocks_of_10" || i%10==0)
                        held=uniform(rng)<probability;
                    count+=held;
                }
                const double center=static_cast<double>(count)/n;
                if (!cache.count(count))
                    cache[count]=dro_mpc::finite_sample_wasserstein_radius(
                        {count,n-count},{center,1.-center},cost,.05);
                const auto& result=cache.at(count);
                const bool in_region=result.lower[0]<=target && target<=result.upper[0] &&
                                     result.lower[1]<=1.-target && 1.-target<=result.upper[1];
                // For two categories with unit off-diagonal transport cost,
                // the exact optimal transport cost is |p_1 - q_1|.
                const bool in_ball=std::abs(target-center)<=result.rho;
                cp_hits+=in_region;
                ball_hits+=in_ball;
                radius_sum+=result.rho;
            }
            out << law << ',' << n << ',' << replications << ',' << seed << ",0.05," << target
                << ',' << cp_hits << ',' << ball_hits << ',' << static_cast<double>(cp_hits)/replications
                << ',' << static_cast<double>(ball_hits)/replications << ',' << radius_sum/replications << '\n';
            std::cout << law << " n=" << n << " CP hits=" << cp_hits << '/' << replications
                      << " ball hits=" << ball_hits << '/' << replications << '\n';
        }
    }
}
