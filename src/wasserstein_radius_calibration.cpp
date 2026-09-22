/**
 * @file wasserstein_radius_calibration.cpp
 * @brief Finite-sample calibration of a discrete Wasserstein ambiguity ball.
 *
 * Given:
 *
 *   - realized categorical mode counts n = (n_1, ..., n_M),
 *   - the nominal ambiguity-ball center p_hat,
 *   - a discrete ground-cost matrix D,
 *   - total confidence failure probability beta,
 *
 * construct a simultaneous finite-sample confidence region
 *
 *     C_beta =
 *       { p in Delta_M :
 *           L_i <= p_i <= U_i,  i = 1,...,M },
 *
 * where [L_i, U_i] are two-sided Clopper-Pearson intervals with
 * Bonferroni allocation beta/M.
 *
 * Then return
 *
 *     rho = max_{p in C_beta} W_D(p_hat, p).
 *
 * Consequently,
 *
 *     P[p_true in B_W(p_hat, rho)] >= 1 - beta,
 *
 * provided the count vector is generated from IID categorical observations.
 *
 * The Wasserstein distance is computed exactly as the optimal transportation
 * LP
 *
 *     W_D(p,q) =
 *       min_{Pi >= 0} sum_ij D_ij Pi_ij
 *
 *       s.t. Pi 1 = p,
 *            Pi^T 1 = q.
 *
 * Because M is small (M=4 in the current application), the transport LP is
 * solved with a small min-cost-flow algorithm and the confidence-polytope
 * vertices are enumerated explicitly.
 */

#include "wasserstein_radius_calibration.hpp"

#include <boost/math/distributions/beta.hpp>

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

constexpr double kProbabilityTolerance = 1e-10;
constexpr double kFlowTolerance = 1e-12;
constexpr double kVertexTolerance = 1e-10;

/*
 * Floating-point guard for Bellman-Ford relaxations in the residual
 * min-cost-flow graph.
 *
 * Residual reverse edges can make two mathematically equal path costs differ
 * by a few ulps. Treat improvements below this scaled threshold as numerical
 * ties so that roundoff cannot create a spurious near-zero negative cycle.
 */
constexpr double kBellmanFordToleranceFactor = 64.0;


/**
 * @brief Normalize a probability vector.
 *
 * Small negative values caused by floating-point noise are projected to zero.
 * Meaningfully negative probabilities trigger an exception.
 */
std::vector<double> normalize_probability_vector(
    const std::vector<double>& input
) {
    std::vector<double> p = input;

    double total = 0.0;

    for (double& value : p) {

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

        total += value;
    }

    if (!(total > kProbabilityTolerance)) {
        throw std::invalid_argument(
            "Probability vector must have positive total mass.");
    }

    for (double& value : p) {
        value /= total;
    }

    return p;
}


/**
 * @brief Validate dimensions and numerical values of a discrete ground-cost
 * matrix.
 *
 * W2-Bures should produce a symmetric metric matrix with zero diagonal.
 * We do not re-project or alter D here because calibration should use exactly
 * the same geometry as the DRO problem.
 */
void validate_ground_cost_matrix(
    const std::vector<std::vector<double>>& D,
    std::size_t M
) {
    if (D.size() != M) {
        throw std::invalid_argument(
            "Ground-cost matrix has incorrect number of rows.");
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
                    "Ground-cost matrix contains a negative cost.");
            }
        }
    }
}


/**
 * @brief Exact two-sided Clopper-Pearson interval.
 *
 * For X ~ Binomial(n,p), returns [L,U] satisfying
 *
 *     P_p[p in [L(X),U(X)]] >= 1 - alpha.
 *
 * Here alpha is the TOTAL failure probability assigned to this coordinate.
 * Each tail therefore receives alpha/2.
 */
std::pair<double, double> clopper_pearson_interval(
    int successes,
    int trials,
    double alpha
) {
    if (trials <= 0) {
        return {0.0, 1.0};
    }

    if (successes < 0 || successes > trials) {
        throw std::invalid_argument(
            "Invalid successes/trials in Clopper-Pearson interval.");
    }

    const double a =
        std::clamp(alpha, 1e-14, 1.0 - 1e-14);

    const double tail =
        0.5 * a;

    double lower = 0.0;
    double upper = 1.0;

    /*
     * Lower endpoint:
     *
     *   Beta^{-1}(alpha/2; x, n-x+1)
     */
    if (successes > 0) {

        boost::math::beta_distribution<double> distribution(
            static_cast<double>(successes),
            static_cast<double>(trials - successes + 1)
        );

        lower =
            boost::math::quantile(
                distribution,
                tail
            );
    }

    /*
     * Upper endpoint:
     *
     *   Beta^{-1}(1-alpha/2; x+1, n-x)
     */
    if (successes < trials) {

        boost::math::beta_distribution<double> distribution(
            static_cast<double>(successes + 1),
            static_cast<double>(trials - successes)
        );

        upper =
            boost::math::quantile(
                distribution,
                1.0 - tail
            );
    }

    lower =
        std::clamp(lower, 0.0, 1.0);

    upper =
        std::clamp(upper, 0.0, 1.0);

    return {lower, upper};
}


/**
 * @brief Residual edge for the min-cost-flow transport solver.
 */
struct FlowEdge {
    int to = -1;
    int reverse_index = -1;
    double capacity = 0.0;
    double cost = 0.0;
};


/**
 * @brief Add a directed residual-network edge.
 */
void add_flow_edge(
    std::vector<std::vector<FlowEdge>>& graph,
    int from,
    int to,
    double capacity,
    double cost
) {
    FlowEdge forward;
    forward.to = to;
    forward.reverse_index =
        static_cast<int>(graph[to].size());
    forward.capacity = capacity;
    forward.cost = cost;

    FlowEdge reverse;
    reverse.to = from;
    reverse.reverse_index =
        static_cast<int>(graph[from].size());
    reverse.capacity = 0.0;
    reverse.cost = -cost;

    graph[from].push_back(forward);
    graph[to].push_back(reverse);
}


/**
 * @brief Exact discrete Wasserstein distance for a finite ground-cost matrix.
 *
 * Solves
 *
 *   min_Pi sum_ij D_ij Pi_ij
 *
 *   s.t.
 *       sum_j Pi_ij = p_i
 *       sum_i Pi_ij = q_j
 *       Pi_ij >= 0.
 *
 * Because the number of modes is tiny, a successive shortest augmenting-path
 * min-cost-flow method with Bellman-Ford shortest paths is sufficient and
 * avoids introducing another external LP dependency.
 */
double discrete_wasserstein_distance(
    const std::vector<double>& p_input,
    const std::vector<double>& q_input,
    const std::vector<std::vector<double>>& D
) {
    if (p_input.size() != q_input.size()) {
        throw std::invalid_argument(
            "Wasserstein distributions have different dimensions.");
    }

    const std::size_t M =
        p_input.size();

    if (M == 0) {
        return 0.0;
    }

    validate_ground_cost_matrix(D, M);

    const std::vector<double> p =
        normalize_probability_vector(p_input);

    const std::vector<double> q =
        normalize_probability_vector(q_input);

    /*
     * Residual graph:
     *
     * source
     *   |
     *   v
     * supply_i ----D_ij----> demand_j
     *                         |
     *                         v
     *                        sink
     */

    const int source = 0;

    const int supply_offset = 1;

    const int demand_offset =
        supply_offset + static_cast<int>(M);

    const int sink =
        demand_offset + static_cast<int>(M);

    const int node_count =
        sink + 1;

    std::vector<std::vector<FlowEdge>>
        graph(node_count);

    /*
     * Source -> source-mode nodes.
     */
    for (std::size_t i = 0; i < M; ++i) {

        add_flow_edge(
            graph,
            source,
            supply_offset + static_cast<int>(i),
            p[i],
            0.0
        );
    }

    /*
     * Source-mode -> destination-mode transport edges.
     *
     * Capacity 1 is sufficient because total probability mass is 1.
     */
    for (std::size_t i = 0; i < M; ++i) {

        for (std::size_t j = 0; j < M; ++j) {

            add_flow_edge(
                graph,
                supply_offset + static_cast<int>(i),
                demand_offset + static_cast<int>(j),
                1.0,
                std::max(0.0, D[i][j])
            );
        }
    }

    /*
     * Destination-mode nodes -> sink.
     */
    for (std::size_t j = 0; j < M; ++j) {

        add_flow_edge(
            graph,
            demand_offset + static_cast<int>(j),
            sink,
            q[j],
            0.0
        );
    }

    double remaining_flow = 1.0;
    double total_cost = 0.0;

    const double infinity =
        std::numeric_limits<double>::infinity();

    while (remaining_flow > kFlowTolerance) {

        /*
         * Reverse residual edges have negative cost, so Bellman-Ford is used
         * rather than plain Dijkstra.
         */
        std::vector<double>
            distance(node_count, infinity);

        std::vector<int>
            previous_node(node_count, -1);

        std::vector<int>
            previous_edge(node_count, -1);

        distance[source] = 0.0;

        /*
         * Standard Bellman-Ford relaxation.
         */
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

                for (int edge_index = 0;
                     edge_index <
                         static_cast<int>(graph[u].size());
                     ++edge_index) {

                    const FlowEdge& edge =
                        graph[u][edge_index];

                    if (edge.capacity <=
                        kFlowTolerance) {
                        continue;
                    }

                    const double candidate =
                        distance[u] + edge.cost;

                    /*
                    * An unreached node must always be relaxable.
                    *
                    * For an already reached node, use a scale-aware tolerance rather than a
                    * fixed 1e-15 threshold. This prevents floating-point noise in zero-cost
                    * residual cycles from creating cyclic predecessor chains.
                    */
                    const bool target_unreached =
                        !std::isfinite(
                            distance[edge.to]
                        );

                    double relaxation_tolerance = 0.0;

                    if (!target_unreached) {

                        const double relaxation_scale =
                            std::max({
                                1.0,
                                std::abs(distance[u]),
                                std::abs(distance[edge.to]),
                                std::abs(candidate),
                                std::abs(edge.cost)
                            });

                        relaxation_tolerance =
                            kBellmanFordToleranceFactor *
                            std::numeric_limits<double>::epsilon() *
                            static_cast<double>(node_count) *
                            relaxation_scale;
                    }

                    if (target_unreached ||
                        candidate <
                            distance[edge.to] -
                                relaxation_tolerance) {

                        distance[edge.to] =
                            candidate;

                        previous_node[edge.to] =
                            u;

                        previous_edge[edge.to] =
                            edge_index;

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
                "Discrete Wasserstein min-cost-flow problem is infeasible.");
        }

    /*
    * Reconstruct and validate the shortest residual path exactly once.
    *
    * A valid predecessor chain is a simple path from sink back to source.
    * Therefore it cannot revisit a node. Explicit cycle detection prevents
    * numerical Bellman-Ford errors from creating an infinite predecessor walk.
    *
    * Each entry stores:
    *
    *     (predecessor node, outgoing edge index)
    *
    * and the entries are ordered from sink back toward source.
    */
    std::vector<std::pair<int, int>>
        augmenting_path;

    augmenting_path.reserve(
        static_cast<std::size_t>(
            std::max(0, node_count - 1)
        )
    );

    std::vector<bool>
        predecessor_visited(
            static_cast<std::size_t>(
                node_count),
            false
        );

    int path_node = sink;

    while (path_node != source) {

        if (path_node < 0 ||
            path_node >= node_count) {

            throw std::runtime_error(
                "Residual path in Wasserstein min-cost flow "
                "contains an invalid node index.");
        }

        if (predecessor_visited[
                static_cast<std::size_t>(
                    path_node)]) {

            throw std::runtime_error(
                "Cycle detected in Wasserstein min-cost-flow "
                "predecessor chain.");
        }

        predecessor_visited[
            static_cast<std::size_t>(
                path_node)] = true;

        if (augmenting_path.size() >=
            static_cast<std::size_t>(
                node_count)) {

            throw std::runtime_error(
                "Wasserstein min-cost-flow predecessor path "
                "exceeded residual-graph node count.");
        }

        const int u =
            previous_node[path_node];

        const int edge_index =
            previous_edge[path_node];

        if (u < 0 ||
            u >= node_count ||
            edge_index < 0 ||
            edge_index >=
                static_cast<int>(
                    graph[u].size())) {

            throw std::runtime_error(
                "Broken residual path in Wasserstein min-cost flow.");
        }

        const FlowEdge& edge =
            graph[u][edge_index];

        if (edge.to != path_node) {

            throw std::runtime_error(
                "Inconsistent predecessor edge in Wasserstein "
                "min-cost flow.");
        }

        augmenting_path.emplace_back(
            u,
            edge_index
        );

        path_node = u;
    }

        /*
        * Determine the maximum amount of residual flow that can be sent through
        * the validated path.
        */
        double augment =
            remaining_flow;

        for (const auto& path_entry :
            augmenting_path) {

            const int u =
                path_entry.first;

            const int edge_index =
                path_entry.second;

            augment =
                std::min(
                    augment,
                    graph[u][edge_index].capacity
                );
        }

        if (!(augment > kFlowTolerance)) {
            throw std::runtime_error(
                "Zero augmentation in Wasserstein min-cost flow.");
        }

        /*
        * Push flow through the same validated path.
        */
        for (const auto& path_entry :
            augmenting_path) {

            const int u =
                path_entry.first;

            const int edge_index =
                path_entry.second;

            FlowEdge& edge =
                graph[u][edge_index];

            const int v =
                edge.to;

            if (edge.reverse_index < 0 ||
                edge.reverse_index >=
                    static_cast<int>(
                        graph[v].size())) {

                throw std::runtime_error(
                    "Invalid reverse residual edge in Wasserstein "
                    "min-cost flow.");
            }

            FlowEdge& reverse =
                graph[v][edge.reverse_index];

            edge.capacity -= augment;
            reverse.capacity += augment;
        }

        total_cost +=
            augment * distance[sink];

        remaining_flow -=
            augment;

        if (remaining_flow < 0.0 &&
            remaining_flow >
                -kFlowTolerance) {

            remaining_flow = 0.0;
        }
    }

    /*
     * Numerical roundoff in the residual optimization can produce a tiny
     * negative result even though all original transport costs are
     * nonnegative.
     */
    return std::max(0.0, total_cost);
}


/**
 * @brief Test whether two vertices are numerically identical.
 */
bool same_vertex(
    const std::vector<double>& a,
    const std::vector<double>& b
) {
    if (a.size() != b.size()) {
        return false;
    }

    for (std::size_t i = 0;
         i < a.size();
         ++i) {

        if (std::abs(a[i] - b[i]) >
            kVertexTolerance) {

            return false;
        }
    }

    return true;
}


/**
 * @brief Add a vertex if it is not already present.
 */
void add_unique_vertex(
    std::vector<std::vector<double>>& vertices,
    const std::vector<double>& candidate
) {
    for (const auto& existing : vertices) {

        if (same_vertex(existing, candidate)) {
            return;
        }
    }

    vertices.push_back(candidate);
}


/**
 * @brief Enumerate the vertices of
 *
 *     { p :
 *         sum_i p_i = 1,
 *         lower_i <= p_i <= upper_i }.
 *
 * In an M-dimensional box intersected with the simplex equality, every
 * nondegenerate vertex has M-1 active box constraints. Therefore all but one
 * coordinate can be placed at either its lower or upper bound, and the final
 * coordinate is determined by sum p_i = 1.
 *
 * For M=4 this requires at most
 *
 *     M * 2^(M-1) = 32
 *
 * candidate constructions.
 */
std::vector<std::vector<double>>
enumerate_box_simplex_vertices(
    const std::vector<double>& lower,
    const std::vector<double>& upper
) {
    if (lower.size() != upper.size()) {
        throw std::invalid_argument(
            "Confidence interval vectors have different dimensions.");
    }

    const std::size_t M =
        lower.size();

    std::vector<std::vector<double>>
        vertices;

    if (M == 0) {
        return vertices;
    }

    for (std::size_t free_index = 0;
         free_index < M;
         ++free_index) {

        std::vector<double>
            candidate(M, 0.0);

        std::function<void(std::size_t)>
            recurse;

        recurse =
            [&](std::size_t index) {

                if (index == M) {

                    double fixed_sum = 0.0;

                    for (std::size_t i = 0;
                         i < M;
                         ++i) {

                        if (i != free_index) {
                            fixed_sum +=
                                candidate[i];
                        }
                    }

                    double free_value =
                        1.0 - fixed_sum;

                    if (free_value <
                            lower[free_index] -
                                kVertexTolerance ||
                        free_value >
                            upper[free_index] +
                                kVertexTolerance) {

                        return;
                    }

                    /*
                     * Only clamp a value that is outside because of tiny
                     * floating-point roundoff.
                     */
                    free_value =
                        std::clamp(
                            free_value,
                            lower[free_index],
                            upper[free_index]
                        );

                    candidate[free_index] =
                        free_value;

                    /*
                     * Correct any tiny sum error entirely in the free
                     * coordinate.
                     */
                    double sum =
                        std::accumulate(
                            candidate.begin(),
                            candidate.end(),
                            0.0
                        );

                    candidate[free_index] +=
                        1.0 - sum;

                    /*
                     * Final feasibility verification.
                     */
                    for (std::size_t i = 0;
                         i < M;
                         ++i) {

                        if (candidate[i] <
                                lower[i] -
                                    kVertexTolerance ||
                            candidate[i] >
                                upper[i] +
                                    kVertexTolerance) {

                            return;
                        }
                    }

                    const double final_sum =
                        std::accumulate(
                            candidate.begin(),
                            candidate.end(),
                            0.0
                        );

                    if (std::abs(final_sum - 1.0) >
                        1e-9) {

                        return;
                    }

                    add_unique_vertex(
                        vertices,
                        candidate
                    );

                    return;
                }

                if (index == free_index) {

                    recurse(index + 1);
                    return;
                }

                /*
                 * Lower-bound branch.
                 */
                candidate[index] =
                    lower[index];

                recurse(index + 1);

                /*
                 * Upper-bound branch.
                 *
                 * Avoid creating the exact same branch when the confidence
                 * interval is degenerate.
                 */
                if (std::abs(
                        upper[index] -
                        lower[index]) >
                    kVertexTolerance) {

                    candidate[index] =
                        upper[index];

                    recurse(index + 1);
                }
            };

        recurse(0);
    }

    return vertices;
}

}  // anonymous namespace


FiniteSampleWassersteinRadius
finite_sample_wasserstein_radius(
    const std::vector<int>& counts,
    const std::vector<double>& center_input,
    const std::vector<std::vector<double>>& ground_cost,
    double beta
) {
    FiniteSampleWassersteinRadius result;

    const std::size_t M =
        counts.size();

    if (M == 0) {
        return result;
    }

    if (center_input.size() != M) {
        throw std::invalid_argument(
            "Mode-count vector and Wasserstein center have different dimensions.");
    }

    validate_ground_cost_matrix(
        ground_cost,
        M
    );

    /*
     * Number of realized categorical observations.
     */
    int sample_count = 0;

    for (int count : counts) {

        if (count < 0) {
            throw std::invalid_argument(
                "Mode counts cannot be negative.");
        }

        sample_count += count;
    }

    result.sample_count =
        sample_count;

    result.lower.assign(M, 0.0);
    result.upper.assign(M, 1.0);

    /*
     * Normalize the configured ambiguity center.
     *
     * This may be the Dirichlet posterior-predictive mean rather than the
     * unsmoothed empirical frequency. That is fine: the confidence set is
     * obtained from the raw count vector, and rho is then chosen so that the
     * entire confidence set lies inside the Wasserstein ball centered at this
     * particular nominal distribution.
     */
    const std::vector<double> center =
        normalize_probability_vector(
            center_input
        );

    /*
     * No data:
     *
     * The only finite-sample-valid confidence region is the whole simplex.
     * Keeping [0,1] for every coordinate produces exactly that region.
     */
    if (sample_count > 0) {

        const double beta_safe =
            std::clamp(
                beta,
                1e-12,
                1.0 - 1e-12
            );

        /*
         * Bonferroni simultaneous confidence allocation:
         *
         *     alpha_i = beta / M.
         *
         * Each two-sided Clopper-Pearson interval has coverage
         *
         *     >= 1 - alpha_i.
         *
         * Therefore by the union bound
         *
         *     P[forall i, p_i in [L_i,U_i]]
         *       >= 1 - sum_i alpha_i
         *       = 1 - beta.
         */
        const double alpha_per_mode =
            beta_safe /
            static_cast<double>(M);

        for (std::size_t i = 0;
             i < M;
             ++i) {

            const auto interval =
                clopper_pearson_interval(
                    counts[i],
                    sample_count,
                    alpha_per_mode
                );

            result.lower[i] =
                interval.first;

            result.upper[i] =
                interval.second;
        }
    }

    /*
     * Confidence polytope:
     *
     *     C =
     *       Delta_M
     *       intersect
     *       product_i [L_i,U_i].
     */
    const auto vertices =
        enumerate_box_simplex_vertices(
            result.lower,
            result.upper
        );

    result.vertex_count =
        static_cast<int>(
            vertices.size()
        );

    if (vertices.empty()) {
        throw std::runtime_error(
            "Finite-sample confidence region has no feasible simplex vertices.");
    }

    /*
     * For fixed center p_hat,
     *
     *     p -> W_D(p_hat,p)
     *
     * is convex.
     *
     * Therefore its maximum over the compact confidence polytope is achieved
     * at an extreme point. It is sufficient to evaluate the enumerated
     * vertices.
     */
    double rho = 0.0;

    for (const auto& vertex :
         vertices) {

        const double distance =
            discrete_wasserstein_distance(
                center,
                vertex,
                ground_cost
            );

        rho =
            std::max(
                rho,
                distance
            );
    }

    result.rho =
        rho;

    return result;
}

}  // namespace dro_mpc