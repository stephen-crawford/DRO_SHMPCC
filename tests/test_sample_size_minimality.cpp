// Independent linear enumeration of the rooted-binomial expression.
#include "config.hpp"
#include <array>
#include <cmath>
#include <iostream>

long double independent_risk(int S, int h, long double beta) {
    long double root_binomial = 1;
    for (int i = 1; i <= h; ++i)
        root_binomial *= std::pow(static_cast<long double>(S-h+i) / i,
                                  1.0L / (S-h));
    return 1 - std::pow(beta / S, 1.0L / (S-h)) / root_binomial;
}

int main() {
    dro_mpc::RuntimeConfig cfg;
    cfg.mpc.sampling.one_minus_chance_constraint_violation_probability = 0.95;
    cfg.mpc.sampling.chance_of_certificate_violation = 0.01;
    const std::array<int,11> expected{193,316,434,550,666,781,895,1009,1123,1237,1351};
    for (int h=0; h<=10; ++h) {
        int enumerated=h+1;
        while (independent_risk(enumerated,h,0.01L) > cfg.epsilon()) ++enumerated;
        const int actual=cfg.compute_required_scenarios(h,0);
        if (actual != enumerated || actual != expected[h] ||
            (h>=1 && cfg.compute_required_scenarios(h-1,1) != actual)) return 1;
        std::cout << "h=" << h << " S=" << actual
                  << " independent_minimum=" << enumerated << '\n';
    }
    return 0;
}
