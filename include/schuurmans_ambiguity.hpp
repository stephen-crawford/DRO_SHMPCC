/**
 * @file schuurmans_ambiguity.hpp
 * @brief Table I of Schuurmans & Patrinos, "A General Framework for Learning-Based
 *        DR MPC of Markov Jump Systems" (arXiv:2106.00561): the five divergence-
 *        based ambiguity families over the discrete mode simplex, each with its
 *        distance D(p̂,p), calibrated radius r(m,β), and conic worst-case
 *        reformulation of  ρ[ξ] = max_{p : D(p̂,p) ≤ r, p ∈ Δ_d} ⟨p, ξ⟩.
 *
 * | Divergence          | D(p̂,p)                                  | radius        | cone        |
 * |---------------------|-----------------------------------------|---------------|-------------|
 * | Total variation     | ‖p−p̂‖₁                                  | 2√r_TV        | Linear      |
 * | Kullback–Leibler    | D_KL(p̂,p)=Σ p̂_i log(p̂_i/p_i)           | r_KL          | Exponential |
 * | Jensen–Shannon      | ½[D_KL(p̂,M)+D_KL(p,M)], M=(p̂+p)/2       | ½ r_KL        | Exponential |
 * | (Squared) Hellinger | Σ(√p_i−√p̂_i)²                           | r_KL          | Quadratic   |
 * | Wasserstein         | min_Π{Σ Π_ij K_ij : Π1=p, Πᵀ1=p̂}        | maxK·√r_TV    | Linear      |
 *
 *   r_TV(m,β) = (d·log2 − log β)/(2m)      (Bretagnolle–Huber–Carol, Eq. 12)
 *   r_KL(m,β) = (d·log m − log β)/m        (method of types,          Eq. 13)
 *   K ∈ ℝ^{d×d} symmetric distance kernel, K_ij = dist(i,j); paper uses (i−j)².
 *
 * Worst-case solvers: TV by exact water-filling; Wasserstein by the Kantorovich
 * dual (self-contained, matches dro.cpp); KL/JS/Hellinger by the
 * unified φ-divergence nested-bisection dual (per-coordinate closed forms below).
 */
#ifndef DRO_MPC_SCHUURMANS_AMBIGUITY_HPP
#define DRO_MPC_SCHUURMANS_AMBIGUITY_HPP

#include "types.hpp"   // dro_mpc::AmbiguityDivergence (Table I families)

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace dro_mpc {
namespace schuurmans {

// AmbiguityDivergence (the five Table I families) is defined in types.hpp so the
// runtime config can reference it without pulling in this solver header.

inline std::string divergence_name(AmbiguityDivergence d) {
    switch (d) {
        case AmbiguityDivergence::TOTAL_VARIATION:  return "total_variation";
        case AmbiguityDivergence::KULLBACK_LEIBLER: return "kullback_leibler";
        case AmbiguityDivergence::JENSEN_SHANNON:   return "jensen_shannon";
        case AmbiguityDivergence::HELLINGER:        return "hellinger";
        case AmbiguityDivergence::WASSERSTEIN:      return "wasserstein";
        default: return "unknown";
    }
}

inline const char* conic_representation(AmbiguityDivergence d) {
    switch (d) {
        case AmbiguityDivergence::TOTAL_VARIATION:  return "linear";
        case AmbiguityDivergence::KULLBACK_LEIBLER: return "exponential";
        case AmbiguityDivergence::JENSEN_SHANNON:   return "exponential";
        case AmbiguityDivergence::HELLINGER:        return "quadratic";
        case AmbiguityDivergence::WASSERSTEIN:      return "linear";
        default: return "unknown";
    }
}

// ---------------------------------------------------------------------------
// Radii (Table I column 3) — valid for any sample size (paper Prop. III.10/11).
// ---------------------------------------------------------------------------

/// r_TV(m,β) = (d·log2 − log β)/(2m)  — Bretagnolle–Huber–Carol (Eq. 12).
inline double r_tv(int d, int m, double beta) {
    const double mm = std::max(1, m);
    const double b = std::min(std::max(beta, 1e-12), 1.0 - 1e-12);
    return (static_cast<double>(d) * std::log(2.0) - std::log(b)) / (2.0 * mm);
}

/// r_KL(m,β) = (d·log m − log β)/m  — method of types (Eq. 13).
inline double r_kl(int d, int m, double beta) {
    const double mm = std::max(1, m);
    const double b = std::min(std::max(beta, 1e-12), 1.0 - 1e-12);
    return (static_cast<double>(d) * std::log(mm) - std::log(b)) / mm;
}

/// Per-family calibrated radius (Table I column 3). `k_max` = max_{i,j} K_ij
/// (only used by WASSERSTEIN).
inline double ambiguity_radius(AmbiguityDivergence fam, int d, int m, double beta,
                               double k_max = 1.0) {
    switch (fam) {
        case AmbiguityDivergence::TOTAL_VARIATION:  return 2.0 * std::sqrt(r_tv(d, m, beta));
        case AmbiguityDivergence::KULLBACK_LEIBLER: return r_kl(d, m, beta);
        case AmbiguityDivergence::JENSEN_SHANNON:   return 0.5 * r_kl(d, m, beta);
        case AmbiguityDivergence::HELLINGER:        return r_kl(d, m, beta);
        case AmbiguityDivergence::WASSERSTEIN:      return k_max * std::sqrt(r_tv(d, m, beta));
        default: return 0.0;
    }
}

// ---------------------------------------------------------------------------
// D(p̂,p) (Table I column 2).
// ---------------------------------------------------------------------------

namespace detail {
inline double safe_log(double x) { return std::log(std::max(x, 1e-300)); }

/// Cost of the greedy source→argmin-cost transport; an UPPER bound on W_K, so
/// "greedy ≤ r ⇒ W_K ≤ r" (sound for feasibility checks). Exact when the optimal
/// plan is deterministic, which holds for the recovered worst-case q*.
inline double wasserstein_upper(const std::vector<double>& phat,
                                const std::vector<double>& p,
                                const std::vector<std::vector<double>>& K) {
    const int d = static_cast<int>(phat.size());
    std::vector<double> supply = phat, demand = p;   // move mass phat -> p
    double cost = 0.0;
    for (int i = 0; i < d; ++i) {
        while (supply[i] > 1e-12) {
            int jbest = -1; double kbest = std::numeric_limits<double>::infinity();
            for (int j = 0; j < d; ++j)
                if (demand[j] > 1e-12 && K[i][j] < kbest) { kbest = K[i][j]; jbest = j; }
            if (jbest < 0) break;
            const double f = std::min(supply[i], demand[jbest]);
            cost += f * K[i][jbest];
            supply[i] -= f; demand[jbest] -= f;
        }
    }
    return cost;
}
}  // namespace detail

/// D(p̂,p) for the chosen family. For WASSERSTEIN, `K` is required and an upper
/// bound on W_K is returned (see wasserstein_upper).
inline double ambiguity_divergence(
    AmbiguityDivergence fam,
    const std::vector<double>& phat,
    const std::vector<double>& p,
    const std::vector<std::vector<double>>* K = nullptr) {
    const int d = static_cast<int>(phat.size());
    switch (fam) {
        case AmbiguityDivergence::TOTAL_VARIATION: {
            double s = 0.0; for (int i = 0; i < d; ++i) s += std::abs(p[i] - phat[i]);
            return s;
        }
        case AmbiguityDivergence::KULLBACK_LEIBLER: {  // D_KL(p̂,p) = Σ p̂ log(p̂/p)
            double s = 0.0;
            for (int i = 0; i < d; ++i)
                if (phat[i] > 1e-300) s += phat[i] * (detail::safe_log(phat[i]) - detail::safe_log(p[i]));
            return s;
        }
        case AmbiguityDivergence::JENSEN_SHANNON: {  // ½[D_KL(p̂,M)+D_KL(p,M)], M=(p̂+p)/2
            double s = 0.0;
            for (int i = 0; i < d; ++i) {
                const double m = 0.5 * (phat[i] + p[i]);
                if (phat[i] > 1e-300) s += 0.5 * phat[i] * (detail::safe_log(phat[i]) - detail::safe_log(m));
                if (p[i]    > 1e-300) s += 0.5 * p[i]    * (detail::safe_log(p[i])    - detail::safe_log(m));
            }
            return s;
        }
        case AmbiguityDivergence::HELLINGER: {  // Σ(√p−√p̂)²
            double s = 0.0;
            for (int i = 0; i < d; ++i) {
                const double diff = std::sqrt(std::max(0.0, p[i])) - std::sqrt(std::max(0.0, phat[i]));
                s += diff * diff;
            }
            return s;
        }
        case AmbiguityDivergence::WASSERSTEIN:
            return K ? detail::wasserstein_upper(phat, p, *K) : 0.0;
        default: return 0.0;
    }
}

// ---------------------------------------------------------------------------
// Worst-case reweighting (Table I column 4): p* = argmax_{D(p̂,p)≤r} ⟨p,ξ⟩.
// ---------------------------------------------------------------------------

struct WorstCase {
    std::vector<double> p;   // recovered worst-case distribution
    double value = 0.0;      // ⟨p*, ξ⟩ (the ambiguous expectation ρ[ξ])
    double divergence = 0.0; // D(p̂, p*) achieved (≤ r on success)
    bool feasible = false;   // D(p̂,p*) ≤ r within tolerance
};

namespace detail {

/// Total-variation worst-case (Linear): water-filling. Move mass from the
/// lowest-ξ modes to the highest-ξ modes; the L1 budget r caps total moved mass
/// at r/2 (each unit moved changes ‖p−p̂‖₁ by 2).
inline WorstCase worst_case_tv(const std::vector<double>& phat,
                               const std::vector<double>& xi, double r) {
    const int d = static_cast<int>(phat.size());
    std::vector<double> p = phat;
    double budget = 0.5 * r;   // movable mass
    std::vector<int> hi(d), lo(d);
    std::iota(hi.begin(), hi.end(), 0);
    std::iota(lo.begin(), lo.end(), 0);
    std::sort(hi.begin(), hi.end(), [&](int a, int b) { return xi[a] > xi[b]; });  // fill high ξ
    std::sort(lo.begin(), lo.end(), [&](int a, int b) { return xi[a] < xi[b]; });  // drain low ξ
    std::vector<double> add(d, 0.0);      // extra mass we can put on mode (up to 1)
    for (int i = 0; i < d; ++i) add[i] = 1.0 - p[i];
    size_t hp = 0, lp = 0;
    while (budget > 1e-15 && hp < hi.size() && lp < lo.size()) {
        int h = hi[hp], l = lo[lp];
        if (xi[h] <= xi[l] + 1e-15) break;             // no profitable move left
        const double move = std::min({budget, p[l], add[h]});
        if (move <= 1e-15) { if (p[l] <= 1e-15) ++lp; else ++hp; continue; }
        p[l] -= move; p[h] += move; add[h] -= move; budget -= move;
        if (p[l] <= 1e-15) ++lp;
        if (add[h] <= 1e-15) ++hp;
    }
    WorstCase wc; wc.p = std::move(p);
    wc.value = 0.0; for (int i = 0; i < d; ++i) wc.value += wc.p[i] * xi[i];
    wc.divergence = ambiguity_divergence(AmbiguityDivergence::TOTAL_VARIATION, phat, wc.p);
    wc.feasible = (wc.divergence <= r + 1e-6);
    return wc;
}

/// Kantorovich-dual Wasserstein worst-case (Linear): min_{λ≥0} λr + Σ p̂_i max_j(ξ_j − λ K_ij).
/// Recover p* by transporting each source i to j*(i)=argmax_j(ξ_j − λ* K_ij).
inline WorstCase worst_case_wasserstein(const std::vector<double>& phat,
                                        const std::vector<double>& xi, double r,
                                        const std::vector<std::vector<double>>& K) {
    const int d = static_cast<int>(phat.size());
    auto dual = [&](double lam) {
        double v = lam * r;
        for (int i = 0; i < d; ++i) {
            double best = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < d; ++j) best = std::max(best, xi[j] - lam * K[i][j]);
            v += phat[i] * best;
        }
        return v;
    };
    // 1D convex minimization over λ ≥ 0 (golden-section on a bracketed range).
    double lo = 0.0, hi = 1.0;
    while (dual(hi) < dual(hi * 0.5) && hi < 1e9) hi *= 2.0;   // grow until increasing
    const double gr = 0.6180339887498949;
    double a = lo, b = hi, c = b - gr * (b - a), e = a + gr * (b - a);
    for (int it = 0; it < 200; ++it) {
        if (dual(c) < dual(e)) { b = e; } else { a = c; }
        c = b - gr * (b - a); e = a + gr * (b - a);
        if (b - a < 1e-10) break;
    }
    const double lam = 0.5 * (a + b);
    std::vector<double> p(d, 0.0);
    for (int i = 0; i < d; ++i) {
        int jbest = 0; double best = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < d; ++j) { double val = xi[j] - lam * K[i][j]; if (val > best) { best = val; jbest = j; } }
        p[jbest] += phat[i];
    }
    WorstCase wc; wc.p = std::move(p);
    wc.value = 0.0; for (int i = 0; i < d; ++i) wc.value += wc.p[i] * xi[i];
    wc.divergence = wasserstein_upper(phat, wc.p, K);
    wc.feasible = (wc.divergence <= r + 1e-6);
    return wc;
}

/// Per-coordinate t_i = p̂_i/p_i for the φ-divergence stationarity condition
/// ξ_i − λ(φ(t)−tφ'(t)) − η = 0, solved in closed form per family:
///   KL:        φ(t)−tφ'(t) = −t            ⇒ t   = (η−ξ_i)/λ
///   Hellinger: φ(t)−tφ'(t) = 1−√t          ⇒ √t  = (λ+η−ξ_i)/λ
///   JS:        φ(t)−tφ'(t) = ½log(2/(1+t)) ⇒ t   = 2·exp(2(η−ξ_i)/λ) − 1
/// with p_i = p̂_i / t_i (clamped ≥ 0). In every case p_i is DECREASING in η.
inline double phi_p_i(AmbiguityDivergence fam, double phat_i, double xi_i,
                      double lambda, double eta) {
    if (phat_i <= 1e-300) return 0.0;
    const double lam = std::max(lambda, 1e-12);
    if (fam == AmbiguityDivergence::KULLBACK_LEIBLER) {
        const double t = (eta - xi_i) / lam;
        return t > 1e-12 ? phat_i / t : phat_i * 1e12;
    }
    if (fam == AmbiguityDivergence::HELLINGER) {
        const double sq = (lam + eta - xi_i) / lam;
        const double t = sq * sq;
        return t > 1e-12 ? phat_i / t : phat_i * 1e12;
    }
    // JENSEN_SHANNON
    const double t = 2.0 * std::exp(std::clamp(2.0 * (eta - xi_i) / lam, -700.0, 700.0)) - 1.0;
    return t > 1e-12 ? phat_i / t : phat_i * 1e12;
}

/// Per-family lower floor on η so every t_i > 0 (finite p_i). Below this the
/// max-ξ mode's mass diverges, so Σp > 1 there; Σp decreases as η grows.
inline double phi_eta_floor(AmbiguityDivergence fam, double xi_max, double lambda) {
    const double lam = std::max(lambda, 1e-12);
    if (fam == AmbiguityDivergence::KULLBACK_LEIBLER) return xi_max + 1e-9;
    if (fam == AmbiguityDivergence::HELLINGER)        return xi_max - lam + 1e-9;
    return xi_max - 0.34657359 * lam + 1e-9;  // JS: 2/(1+t)=exp(2(ξ−η)/λ) needs η > ξ + (λ/2)ln½
}

/// Given λ, choose the normalizer η so Σ_i p_i(λ,η) = 1 (monotone ⇒ bisection),
/// returning the normalized p.
inline std::vector<double> phi_normalize(AmbiguityDivergence fam,
                                         const std::vector<double>& phat,
                                         const std::vector<double>& xi,
                                         double lambda,
                                         std::vector<double>& p_out) {
    const int d = static_cast<int>(phat.size());
    const double xi_max = *std::max_element(xi.begin(), xi.end());
    std::vector<double> p(d, 0.0);
    auto sum_at = [&](double eta) {
        double s = 0.0;
        for (int i = 0; i < d; ++i) { p[i] = std::max(0.0, phi_p_i(fam, phat[i], xi[i], lambda, eta)); s += p[i]; }
        return s;
    };
    // Σp is DECREASING in η for all three families. At the domain floor Σp > 1
    // (max-ξ mass diverges); grow η until Σp < 1, then bisect for Σp = 1.
    const double floor = phi_eta_floor(fam, xi_max, lambda);
    double eta_lo = floor, eta_hi = floor + 1.0;
    while (sum_at(eta_hi) > 1.0 && (eta_hi - floor) < 1e12) eta_hi = floor + (eta_hi - floor) * 2.0;
    for (int it = 0; it < 300; ++it) {
        const double eta = 0.5 * (eta_lo + eta_hi);
        const double s = sum_at(eta);
        if (s > 1.0) eta_lo = eta; else eta_hi = eta;   // decreasing in η
        if (std::abs(s - 1.0) < 1e-11 || eta_hi - eta_lo < 1e-14) break;
    }
    double s = 0.0; for (double v : p) s += v;
    if (s > 1e-300) for (double& v : p) v /= s;   // defensive renormalize
    p_out = p;
    return p;
}

/// φ-divergence worst-case (KL / JS / Hellinger) via nested bisection: outer on
/// the divergence multiplier λ so D(p̂,p)=r (D decreasing in λ), inner on η.
inline WorstCase worst_case_phi(AmbiguityDivergence fam,
                                const std::vector<double>& phat,
                                const std::vector<double>& xi, double r) {
    std::vector<double> p;
    auto div_at = [&](double lambda) {
        phi_normalize(fam, phat, xi, lambda, p);
        return ambiguity_divergence(fam, phat, p);
    };
    // Large λ ⇒ p≈p̂ ⇒ D≈0; small λ ⇒ p concentrates ⇒ D large. Bracket λ.
    double lam_hi = 1.0;
    while (div_at(lam_hi) > r && lam_hi < 1e9) lam_hi *= 2.0;   // enough penalty
    double lam_lo = lam_hi;
    while (div_at(lam_lo) < r && lam_lo > 1e-9) lam_lo *= 0.5;  // too little penalty
    // If even tiny λ can't reach r, the ball contains the unconstrained optimum.
    double lam = 0.5 * (lam_lo + lam_hi);
    for (int it = 0; it < 200; ++it) {
        lam = 0.5 * (lam_lo + lam_hi);
        const double dv = div_at(lam);
        if (dv > r) lam_lo = lam; else lam_hi = lam;   // D decreasing in λ
        if (lam_hi - lam_lo < 1e-12 * std::max(1.0, lam_hi)) break;
    }
    phi_normalize(fam, phat, xi, lam, p);
    WorstCase wc; wc.p = p;
    wc.value = 0.0; for (size_t i = 0; i < p.size(); ++i) wc.value += p[i] * xi[i];
    wc.divergence = ambiguity_divergence(fam, phat, p);
    wc.feasible = (wc.divergence <= r + 1e-4);
    return wc;
}

}  // namespace detail

/**
 * @brief The Schuurmans ambiguous expectation ρ[ξ] = max_{p∈A_β} ⟨p,ξ⟩ and its
 *        maximizing p*, for the chosen Table I family. `phat` is the empirical
 *        belief, `xi` the per-mode value/risk vector, `r` the radius (from
 *        ambiguity_radius), and `K` the distance kernel (WASSERSTEIN only).
 *
 * p* is always a valid, coverage-consistent reweighting: p̂ is feasible
 * (D(p̂,p̂)=0≤r) so the returned value ≥ ⟨p̂,ξ⟩ (distributionally robust).
 */
inline WorstCase worst_case_expectation(
    AmbiguityDivergence fam,
    const std::vector<double>& phat,
    const std::vector<double>& xi,
    double r,
    const std::vector<std::vector<double>>* K = nullptr) {
    if (r <= 0.0) {  // ball is the singleton {p̂}
        WorstCase wc; wc.p = phat; wc.divergence = 0.0; wc.feasible = true;
        for (size_t i = 0; i < phat.size(); ++i) wc.value += phat[i] * xi[i];
        return wc;
    }
    switch (fam) {
        case AmbiguityDivergence::TOTAL_VARIATION: return detail::worst_case_tv(phat, xi, r);
        case AmbiguityDivergence::WASSERSTEIN:
            return K ? detail::worst_case_wasserstein(phat, xi, r, *K)
                     : detail::worst_case_tv(phat, xi, r);
        case AmbiguityDivergence::KULLBACK_LEIBLER:
        case AmbiguityDivergence::JENSEN_SHANNON:
        case AmbiguityDivergence::HELLINGER:
            return detail::worst_case_phi(fam, phat, xi, r);
        default: {
            WorstCase wc; wc.p = phat; wc.feasible = true; return wc;
        }
    }
}

}  // namespace schuurmans
}  // namespace dro_mpc

#endif  // DRO_MPC_SCHUURMANS_AMBIGUITY_HPP
