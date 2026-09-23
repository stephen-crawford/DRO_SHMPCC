// Offline adapter: call existing production CP and de Groot APIs unchanged.
#include "config.hpp"
#include "wasserstein_radius_calibration.hpp"
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(17);
        if (argc == 5 && std::string(argv[1]) == "risk") {
            const int s=std::stoi(argv[2]),support=std::stoi(argv[3]);
            const double beta=std::stod(argv[4]);
            if (s<=0 || support<0 || !(beta>0 && beta<1)) throw std::invalid_argument("invalid risk inputs");
            std::cout << dro_mpc::RuntimeConfig::degroot_violation_risk(s,support,beta) << '\n';
            return 0;
        }
        if (argc >= 4 && std::string(argv[1]) == "intervals") {
            const double beta = std::stod(argv[2]);
            if (!(beta > 0 && beta < 1)) throw std::invalid_argument("invalid beta");
            std::vector<int> counts;
            for (int i=3;i<argc;++i) {
                const int n=std::stoi(argv[i]);
                if (n<0) throw std::invalid_argument("negative count");
                counts.push_back(n);
            }
            const double total=std::accumulate(counts.begin(),counts.end(),0.);
            std::vector<double> center(counts.size());
            std::vector<std::vector<double>> cost(counts.size(),std::vector<double>(counts.size(),1.));
            for (size_t i=0;i<counts.size();++i) {
                center[i]=total ? counts[i]/total : 1./counts.size();cost[i][i]=0.;
            }
            // Only L,U are exported; the auxiliary unit-cost radius is discarded.
            const auto result=dro_mpc::finite_sample_wasserstein_radius(counts,center,cost,beta);
            for (size_t i=0;i<counts.size();++i)
                std::cout << result.lower[i] << ' ' << result.upper[i] << '\n';
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "size") {
            const double eta=std::stod(argv[2]),beta=std::stod(argv[3]);
            const int cap=std::stoi(argv[4]),removed=std::stoi(argv[5]);
            if (!(eta>0 && eta<=1 && beta>0 && beta<1) || cap<0 || removed<0)
                throw std::invalid_argument("invalid sizing inputs");
            dro_mpc::RuntimeConfig cfg;
            cfg.mpc.sampling.one_minus_chance_constraint_violation_probability=1.-eta;
            cfg.mpc.sampling.chance_of_certificate_violation=beta;
            const int s=cfg.compute_required_scenarios(cap,removed);
            const double risk=dro_mpc::RuntimeConfig::degroot_violation_risk(s,cap+removed,beta);
            // Production API has a search cap; never present its sentinel as a valid minimum.
            if (risk>eta) throw std::runtime_error("production sizing search cap reached");
            std::cout << s << ' ' << risk << ' '
                      << dro_mpc::RuntimeConfig::degroot_violation_risk(s-1,cap+removed,beta) << '\n';
            return 0;
        }
        throw std::invalid_argument("usage: certificate_numeric_backend intervals BETA COUNT... | size ETA BETA N_BAR REMOVALS | risk S TOTAL_SUPPORT BETA");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n';return 1; }
}
