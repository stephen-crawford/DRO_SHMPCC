#pragma once

#include <map>
#include <stdexcept>
#include <string>

namespace dro_mpc::test {

// Test-only synthetic evidence for fixtures that previously supplied just a
// total observation count. Spread it evenly, retaining the exact total.
inline std::map<std::string, int> uniform_mode_counts(
    const std::map<std::string, double>& nominal, int total)
{
    if (nominal.empty() || total < 0) {
        throw std::invalid_argument("Synthetic mode counts need modes and a nonnegative total");
    }
    std::map<std::string, int> counts;
    const int per_mode = total / static_cast<int>(nominal.size());
    int remainder = total % static_cast<int>(nominal.size());
    for (const auto& [mode, weight] : nominal) {
        counts[mode] = per_mode + (remainder > 0 ? 1 : 0);
        if (remainder > 0) --remainder;
    }
    return counts;
}

}  // namespace dro_mpc::test
