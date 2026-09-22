/**
 * @file exact_wasserstein_inversion.cpp
 * @brief Exact finite-sample multinomial test inversion using a discrete
 *        Wasserstein statistic.
 *
 * For an observed categorical count vector
 *
 *     n_obs = (n_1,...,n_M),   sum_i n_i = m,
 *
 * and candidate true mode distribution p, define
 *
 *     T(n,p) = W_D(n/m, p).
 *
 * The exact multinomial p-value is
 *
 *     pv(p; n_obs)
 *       =
 *       sum_{n' : T(n',p) >= T(n_obs,p)}
 *           MultinomialPMF(n'; m,p).
 *
 * The exact finite-sample confidence region is
 *
 *     C_beta =
 *       { p in Delta_M : pv(p;n_obs) > beta }.
 *
 * This file evaluates the exact p-value for candidate p and performs
 * inversion over a simplex lattice.
 *
 * IMPORTANT:
 *
 *   - The multinomial probability calculation is exact up to floating-point
 *     arithmetic.
 *
 *   - The enumeration over all count vectors is exhaustive.
 *
 *   - The final inversion is over a finite lattice approximation of the
 *     continuous probability simplex. Therefore `rho_grid` is NOT, by itself,
 *     a certified continuum outer radius.
 *
 * For the current four-mode problem this implementation is intended as:
 *
 *   1. a validation/reference implementation;
 *   2. an offline calibration tool;
 *   3. a basis for a later certified branch-and-bound inversion.
 */

#include "exact_wasserstein_inversion.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dro_mpc {

namespace {

constexpr double kProbabilityTolerance = 1e-12;
constexpr double kDistanceTolerance = 1e-12;
constexpr double kFlowTolerance = 1e-12;


/* -------------------------------------------------------------------------- */
/* Probability utilities                                                      */
/* -------------------------------------------------------------------------- */

std::vector<double> normalize_probability_vector(
    const std::vector<double>& input
) {
    if (input.empty()) {
        return {};
    }

    std::vector<double> result = input;

    double sum = 0.0;

    for (double& value : result) {

        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Probability vector contains a non-finite value.");
        }

        if (value < -kProbabilityTolerance) {
            throw std::invalid_argument(
                "Probability vector contains a negative value.");
        }

        if (value < 0.0) {
            value = 0.0;
        }

        sum += value;
    }

    if (!(sum > 0.0)) {
        throw std::invalid_argument(
            "Probability vector has zero total mass.");
    }

    for (double& value : result) {
        value /= sum;
    }

    return result;
}


void validate_ground_cost(
    const std::vector<std::vector<double>>& D,
    std::size_t M
) {
    if (D.size() != M) {
        throw std::invalid_argument(
            "Ground-cost matrix has incorrect dimension.");
    }

    for (std::size_t i = 0; i < M; ++i) {

        if (D[i].size() != M) {
            throw std::invalid_argument(
                "Ground-cost matrix must be square.");
        }

        for (std::size_t j = 0; j < M; ++j) {

            if (!std::isfinite(D[i][j])) {
                throw std::invalid_argument(
                    "Ground-cost matrix contains a non-finite value.");
            }

            if (D[i][j] < -kProbabilityTolerance) {
                throw std::invalid_argument(
                    "Ground-cost matrix contains a negative value.");
            }
        }
    }
}


/* -------------------------------------------------------------------------- */
/* Exact finite discrete Wasserstein distance                                 */
/* -------------------------------------------------------------------------- */

struct FlowEdge {
    int to = -1;
    int reverse = -1;

    double capacity = 0.0;
    double cost = 0.0;
};


void add_edge(
    std::vector<std::vector<FlowEdge>>& graph,
    int from,
    int to,
    double capacity,
    double cost
) {
    FlowEdge forward;

    forward.to = to;
    forward.reverse =
        static_cast<int>(graph[to].size());
    forward.capacity = capacity;
    forward.cost = cost;

    FlowEdge reverse;

    reverse.to = from;
    reverse.reverse =
        static_cast<int>(graph[from].size());
    reverse.capacity = 0.0;
    reverse.cost = -cost;

    graph[from].push_back(forward);
    graph[to].push_back(reverse);
}


/**
 * Exact discrete optimal-transport distance:
 *
 *   W_D(p,q)
 *     =
 *     min_Pi sum_ij D_ij Pi_ij
 *
 *     s.t.
 *         Pi 1   = p
 *         Pi^T 1 = q
 *         Pi >= 0.
 */
double discrete_wasserstein_distance(
    const std::vector<double>& p_input,
    const std::vector<double>& q_input,
    const std::vector<std::vector<double>>& D
) {
    if (p_input.size() != q_input.size()) {
        throw std::invalid_argument(
            "Wasserstein arguments have different dimensions.");
    }

    const std::size_t M =
        p_input.size();

    if (M == 0) {
        return 0.0;
    }

    validate_ground_cost(D, M);

    const std::vector<double> p =
        normalize_probability_vector(p_input);

    const std::vector<double> q =
        normalize_probability_vector(q_input);

    const int source = 0;

    const int supply_offset = 1;

    const int demand_offset =
        supply_offset +
        static_cast<int>(M);

    const int sink =
        demand_offset +
        static_cast<int>(M);

    const int node_count =
        sink + 1;

    std::vector<std::vector<FlowEdge>>
        graph(node_count);

    for (std::size_t i = 0; i < M; ++i) {

        add_edge(
            graph,
            source,
            supply_offset + static_cast<int>(i),
            p[i],
            0.0
        );
    }

    for (std::size_t i = 0; i < M; ++i) {

        for (std::size_t j = 0; j < M; ++j) {

            add_edge(
                graph,
                supply_offset + static_cast<int>(i),
                demand_offset + static_cast<int>(j),
                1.0,
                std::max(0.0, D[i][j])
            );
        }
    }

    for (std::size_t j = 0; j < M; ++j) {

        add_edge(
            graph,
            demand_offset + static_cast<int>(j),
            sink,
            q[j],
            0.0
        );
    }

    double remaining = 1.0;
    double total_cost = 0.0;

    const double infinity =
        std::numeric_limits<double>::infinity();

    while (remaining > kFlowTolerance) {

        /*
         * Reverse residual edges have negative cost, so use Bellman-Ford.
         */
        std::vector<double>
            distance(node_count, infinity);

        std::vector<int>
            previous_node(node_count, -1);

        std::vector<int>
            previous_edge(node_count, -1);

        distance[source] = 0.0;

        for (int iteration = 0;
             iteration < node_count - 1;
             ++iteration) {

            bool changed = false;

            for (int u = 0;
                 u < node_count;
                 ++u) {

                if (!std::isfinite(distance[u])) {
                    continue;
                }

                for (int e = 0;
                     e <
                         static_cast<int>(
                             graph[u].size());
                     ++e) {

                    const FlowEdge& edge =
                        graph[u][e];

                    if (edge.capacity <=
                        kFlowTolerance) {

                        continue;
                    }

                    const double candidate =
                        distance[u] +
                        edge.cost;

                    if (candidate <
                        distance[edge.to] -
                            1e-15) {

                        distance[edge.to] =
                            candidate;

                        previous_node[edge.to] =
                            u;

                        previous_edge[edge.to] =
                            e;

                        changed = true;
                    }
                }
            }

            if (!changed) {
                break;
            }
        }

        if (previous_node[sink] < 0) {
            throw std::runtime_error(
                "Wasserstein min-cost-flow problem became infeasible.");
        }

        double augment =
            remaining;

        for (int v = sink;
             v != source;
             v = previous_node[v]) {

            const int u =
                previous_node[v];

            const int e =
                previous_edge[v];

            if (u < 0 || e < 0) {
                throw std::runtime_error(
                    "Broken augmenting path in Wasserstein solver.");
            }

            augment =
                std::min(
                    augment,
                    graph[u][e].capacity
                );
        }

        if (!(augment >
              kFlowTolerance)) {

            throw std::runtime_error(
                "Wasserstein flow augmentation is numerically zero.");
        }

        for (int v = sink;
             v != source;
             v = previous_node[v]) {

            const int u =
                previous_node[v];

            const int e =
                previous_edge[v];

            FlowEdge& forward =
                graph[u][e];

            FlowEdge& reverse =
                graph[v][forward.reverse];

            forward.capacity -= augment;
            reverse.capacity += augment;
        }

        total_cost +=
            augment * distance[sink];

        remaining -= augment;

        if (remaining < 0.0 &&
            remaining > -kFlowTolerance) {

            remaining = 0.0;
        }
    }

    return std::max(
        0.0,
        total_cost
    );
}


/* -------------------------------------------------------------------------- */
/* Multinomial outcomes                                                       */
/* -------------------------------------------------------------------------- */

struct MultinomialOutcome {
    std::vector<int> counts;

    /*
     * log(m! / product_i n_i!)
     */
    double log_coefficient = 0.0;

    /*
     * Empirical categorical distribution n/m.
     */
    std::vector<double> empirical;
};


void enumerate_count_vectors_recursive(
    int remaining,
    std::size_t slots_left,
    std::vector<int>& prefix,
    std::vector<std::vector<int>>& output
) {
    if (slots_left == 0) {
        if (remaining == 0) {
            output.emplace_back(
                prefix.cbegin(),
                prefix.cend()
            );
        }
        return;
    }

    /*
     * Last coordinate is uniquely determined by the requirement
     *
     *     sum_i n_i = sample_count.
     */
    if (slots_left == 1) {
        prefix.push_back(remaining);

        /*
         * Construct explicitly from iterators instead of invoking the
         * vector copy constructor. This also avoids the GCC 13
         * -Warray-bounds false positive seen under optimization.
         */
        output.emplace_back(
            prefix.cbegin(),
            prefix.cend()
        );

        prefix.pop_back();
        return;
    }

    for (int value = 0;
         value <= remaining;
         ++value) {

        prefix.push_back(value);

        enumerate_count_vectors_recursive(
            remaining - value,
            slots_left - 1,
            prefix,
            output
        );

        prefix.pop_back();
    }
}


std::vector<std::vector<int>>
enumerate_count_vectors(
    int sample_count,
    std::size_t M
) {
    if (sample_count < 0) {
        throw std::invalid_argument(
            "sample_count cannot be negative.");
    }

    std::vector<std::vector<int>> output;

    if (M == 0) {
        return output;
    }

    std::vector<int> prefix;
    prefix.reserve(M);

    enumerate_count_vectors_recursive(
        sample_count,
        M,
        prefix,
        output
    );

    return output;
}


std::vector<MultinomialOutcome>
build_multinomial_outcomes(
    int sample_count,
    std::size_t M
) {
    const auto count_vectors =
        enumerate_count_vectors(
            sample_count,
            M
        );

    std::vector<MultinomialOutcome>
        outcomes;

    outcomes.reserve(
        count_vectors.size()
    );

    const double log_m_factorial =
        std::lgamma(
            static_cast<double>(
                sample_count) +
            1.0
        );

    for (const auto& counts :
         count_vectors) {

        MultinomialOutcome outcome;

        outcome.counts =
            counts;

        outcome.empirical.resize(
            M,
            0.0
        );

        double log_denominator =
            0.0;

        for (std::size_t i = 0;
             i < M;
             ++i) {

            log_denominator +=
                std::lgamma(
                    static_cast<double>(
                        counts[i]) +
                    1.0
                );

            if (sample_count > 0) {

                outcome.empirical[i] =
                    static_cast<double>(
                        counts[i]) /
                    static_cast<double>(
                        sample_count);
            }
        }

        outcome.log_coefficient =
            log_m_factorial -
            log_denominator;

        outcomes.push_back(
            std::move(outcome)
        );
    }

    return outcomes;
}


/* -------------------------------------------------------------------------- */
/* Multinomial PMF                                                            */
/* -------------------------------------------------------------------------- */

/**
 * log Mult(n | m,p).
 *
 * Handles p_i = 0 exactly:
 *
 *   - if n_i > 0 and p_i = 0, probability is zero;
 *   - 0 * log(0) contributes zero when n_i = 0.
 */
double multinomial_log_probability(
    const MultinomialOutcome& outcome,
    const std::vector<double>& p
) {
    double log_probability =
        outcome.log_coefficient;

    for (std::size_t i = 0;
         i < p.size();
         ++i) {

        const int n_i =
            outcome.counts[i];

        if (n_i == 0) {
            continue;
        }

        if (!(p[i] > 0.0)) {

            return
                -std::numeric_limits<double>::infinity();
        }

        log_probability +=
            static_cast<double>(
                n_i) *
            std::log(p[i]);
    }

    return log_probability;
}


/* -------------------------------------------------------------------------- */
/* Exact multinomial Wasserstein p-value                                      */
/* -------------------------------------------------------------------------- */

double exact_wasserstein_p_value(
    const std::vector<int>& observed_counts,
    const std::vector<double>& candidate_input,
    const std::vector<std::vector<double>>& D,
    const std::vector<MultinomialOutcome>& outcomes
) {
    const std::size_t M =
        candidate_input.size();

    const std::vector<double> candidate =
        normalize_probability_vector(
            candidate_input
        );

    const int sample_count =
        std::accumulate(
            observed_counts.begin(),
            observed_counts.end(),
            0
        );

    if (sample_count <= 0) {
        return 1.0;
    }

    std::vector<double>
        observed_empirical(M, 0.0);

    for (std::size_t i = 0;
         i < M;
         ++i) {

        observed_empirical[i] =
            static_cast<double>(
                observed_counts[i]) /
            static_cast<double>(
                sample_count);
    }

    /*
     * Observed Wasserstein statistic.
     */
    const double observed_statistic =
        discrete_wasserstein_distance(
            observed_empirical,
            candidate,
            D
        );

    /*
     * Sum the exact multinomial mass of all outcomes at least as
     * extreme according to the Wasserstein statistic.
     *
     * We first collect log probabilities and use log-sum-exp for
     * numerical stability.
     */
    std::vector<double>
        accepted_log_probabilities;

    accepted_log_probabilities.reserve(
        outcomes.size()
    );

    double maximum_log_probability =
        -std::numeric_limits<double>::infinity();

    for (const auto& outcome :
         outcomes) {

        const double statistic =
            discrete_wasserstein_distance(
                outcome.empirical,
                candidate,
                D
            );

        /*
         * Conservative handling of floating-point ties:
         *
         * Include outcomes whose statistic is equal to the observed
         * statistic up to numerical tolerance.
         */
        if (statistic +
                kDistanceTolerance <
            observed_statistic) {

            continue;
        }

        const double log_probability =
            multinomial_log_probability(
                outcome,
                candidate
            );

        if (!std::isfinite(
                log_probability)) {

            continue;
        }

        accepted_log_probabilities.push_back(
            log_probability
        );

        maximum_log_probability =
            std::max(
                maximum_log_probability,
                log_probability
            );
    }

    if (accepted_log_probabilities.empty()) {
        return 0.0;
    }

    double scaled_sum = 0.0;

    for (double log_probability :
         accepted_log_probabilities) {

        scaled_sum +=
            std::exp(
                log_probability -
                maximum_log_probability
            );
    }

    const double log_p_value =
        maximum_log_probability +
        std::log(scaled_sum);

    return std::clamp(
        std::exp(log_p_value),
        0.0,
        1.0
    );
}


/* -------------------------------------------------------------------------- */
/* Simplex lattice                                                            */
/* -------------------------------------------------------------------------- */

/**
 * Enumerate
 *
 *   p_i = k_i / L,
 *   k_i >= 0,
 *   sum_i k_i = L.
 */
std::vector<std::vector<double>>
enumerate_simplex_lattice(
    std::size_t M,
    int denominator
) {
    if (M == 0) {
        return {};
    }

    if (denominator <= 0) {
        throw std::invalid_argument(
            "Simplex lattice denominator must be positive.");
    }

    const auto integer_points =
        enumerate_count_vectors(
            denominator,
            M
        );

    std::vector<std::vector<double>>
        lattice;

    lattice.reserve(
        integer_points.size()
    );

    for (const auto& integer_point :
         integer_points) {

        std::vector<double>
            p(M, 0.0);

        for (std::size_t i = 0;
             i < M;
             ++i) {

            p[i] =
                static_cast<double>(
                    integer_point[i]) /
                static_cast<double>(
                    denominator);
        }

        lattice.push_back(
            std::move(p)
        );
    }

    return lattice;
}

}  // anonymous namespace


ExactWassersteinInversionResult
exact_multinomial_wasserstein_inversion(
    const std::vector<int>& observed_counts,
    const std::vector<double>& center_input,
    const std::vector<std::vector<double>>& ground_cost,
    double beta,
    int grid_denominator
) {
    ExactWassersteinInversionResult result;

    const std::size_t M =
        observed_counts.size();

    if (M == 0) {
        return result;
    }

    if (center_input.size() != M) {
        throw std::invalid_argument(
            "Observed counts and ambiguity center have different dimensions.");
    }

    validate_ground_cost(
        ground_cost,
        M
    );

    int sample_count = 0;

    for (int count :
         observed_counts) {

        if (count < 0) {
            throw std::invalid_argument(
                "Observed mode counts cannot be negative.");
        }

        sample_count +=
            count;
    }

    result.sample_count =
        sample_count;

    result.grid_denominator =
        grid_denominator;

    result.beta =
        beta;

    const std::vector<double> center =
        normalize_probability_vector(
            center_input
        );

    /*
     * With no observations every p is statistically possible.
     */
    if (sample_count == 0) {

        const auto lattice =
            enumerate_simplex_lattice(
                M,
                grid_denominator
            );

        result.candidate_count =
            lattice.size();

        result.accepted_candidate_count =
            lattice.size();

        for (const auto& p :
             lattice) {

            const double distance =
                discrete_wasserstein_distance(
                    center,
                    p,
                    ground_cost
                );

            if (distance >
                result.rho_grid) {

                result.rho_grid =
                    distance;

                result.maximizing_distribution =
                    p;
            }
        }

        return result;
    }

    const double beta_safe =
        std::clamp(
            beta,
            1e-12,
            1.0 - 1e-12
        );

    /*
     * Enumerate the ENTIRE multinomial sample space exactly.
     *
     * Number of outcomes:
     *
     *     C(m + M - 1, M - 1).
     *
     * For M=4:
     *
     *   m=50  ->  23,426
     *   m=100 -> 176,851
     *   m=140 -> 477,191
     *
     * so this should be treated as an offline/reference computation,
     * especially because each candidate p requires Wasserstein evaluation
     * for every outcome.
     */
    const auto outcomes =
        build_multinomial_outcomes(
            sample_count,
            M
        );

    result.multinomial_outcome_count =
        outcomes.size();

    /*
     * Candidate true distributions used to invert the exact test.
     */
    const auto candidates =
        enumerate_simplex_lattice(
            M,
            grid_denominator
        );

    result.candidate_count =
        candidates.size();

    result.rho_grid = 0.0;

    for (const auto& candidate :
         candidates) {

        /*
         * Exact finite-sample p-value at this candidate p.
         */
        const double p_value =
            exact_wasserstein_p_value(
                observed_counts,
                candidate,
                ground_cost,
                outcomes
            );

        /*
         * Exact-test inversion:
         *
         * reject p if p-value <= beta.
         *
         * Keep the boundary conservatively when numerical error places
         * p extremely close to beta.
         */
        if (p_value +
                1e-14 <
            beta_safe) {

            continue;
        }

        ++result.accepted_candidate_count;

        /*
         * Radius required for the ambiguity ball centered at the
         * controller's nominal distribution.
         */
        const double distance_from_center =
            discrete_wasserstein_distance(
                center,
                candidate,
                ground_cost
            );

        if (distance_from_center >
            result.rho_grid) {

            result.rho_grid =
                distance_from_center;

            result.maximizing_distribution =
                candidate;
        }
    }

    /*
     * The observed empirical distribution should ordinarily guarantee
     * a nonempty accepted set, but a very coarse lattice may not contain
     * a nearby accepted candidate.
     */
    if (result.accepted_candidate_count == 0) {

        throw std::runtime_error(
            "Exact multinomial inversion found no accepted lattice "
            "candidate. Increase grid_denominator.");
    }

    return result;
}

}  // namespace dro_mpc