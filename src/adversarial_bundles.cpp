#include "adversarial_bundles.hpp"
#include "scenario_sampler.hpp"
#include "wasserstein_radius_calibration.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <tuple>

namespace dro_mpc {
BundleDesign design_adversarial_bundles(const ModeHistory& history,
    const std::map<std::string,double>& q, const std::map<std::string,double>& scores,
    const RuntimeConfig& config) {
    BundleDesign d;
    const auto counts=history.get_mode_counts();
    std::vector<int> ns;
    for (const auto& [mode,model] : history.available_modes) {
        d.modes.push_back(mode);
        const auto it=counts.find(mode); ns.push_back(it==counts.end()?0:it->second);
    }
    const int M=d.modes.size();
    if (M<1 || M>6) throw std::invalid_argument("bundles require 1..6 known modes");
    if (M==1) {
        d.lower={1}; d.upper={1};
    } else if (std::accumulate(ns.begin(),ns.end(),0)==0) {
        // With no history the entire simplex is the confidence region.
        d.lower.assign(M,0); d.upper.assign(M,1);
    } else {
        std::vector<double> center(M,1.0/M);
        std::vector<std::vector<double>> cost(M,std::vector<double>(M,1));
        for (int i=0;i<M;++i) cost[i][i]=0;
        const auto cp=finite_sample_wasserstein_radius(ns,center,cost,config.mpc.bundle_beta_cp);
        d.lower=cp.lower; d.upper=cp.upper;
    }
    const double sum_lower=std::accumulate(d.lower.begin(),d.lower.end(),0.0);
    double total_score=0;
    std::vector<double> allocation(M);
    for (int i=0;i<M;++i) {
        // Coordinate maximum over the box intersected with the simplex.
        d.upper[i]=std::min(d.upper[i],1-(sum_lower-d.lower[i]));
        if (!(d.upper[i]>=0 && d.upper[i]<=1)) throw std::runtime_error("invalid CP simplex upper bound");
        d.multiplicities.push_back(static_cast<int>(std::ceil(config.mpc.bundle_amplification*d.upper[i])));
        const auto qi=q.find(d.modes[i]), si=scores.find(d.modes[i]);
        allocation[i]=(qi==q.end()?0:qi->second)*std::max(0.0,si==scores.end()?0:si->second);
        total_score+=allocation[i];
    }
    // Deterministic largest-remainder extras. The coverage floor never changes.
    std::vector<std::pair<double,int>> remainder;
    int allocated=0;
    for (int i=0;i<M;++i) {
        const double share=config.mpc.bundle_extra_draws*(total_score>0?allocation[i]/total_score:1.0/M);
        const int whole=static_cast<int>(std::floor(share));
        d.multiplicities[i]+=whole; allocated+=whole;
        remainder.emplace_back(-(share-whole),i);
    }
    std::sort(remainder.begin(),remainder.end());
    for (int i=0;i<config.mpc.bundle_extra_draws-allocated;++i) ++d.multiplicities[remainder.at(i).second];
    d.amplification=std::numeric_limits<double>::infinity();
    for (int i=0;i<M;++i) {
        d.raw_per_bundle+=d.multiplicities[i];
        if (d.upper[i]>0) d.amplification=std::min(d.amplification,d.multiplicities[i]/d.upper[i]);
    }
    d.threshold=-std::expm1(d.amplification*std::log1p(-config.epsilon()));
    if (!(d.threshold>0 && d.threshold<1)) throw std::runtime_error("bundle threshold outside representable open interval");
    auto sizing=config;
    sizing.mpc.sampling.one_minus_chance_constraint_violation_probability=1-d.threshold;
    d.required_bundles=sizing.compute_required_scenarios();
    return d;
}

std::vector<Scenario> sample_adversarial_bundles(int id,const ObstacleState& state,
    const ModeHistory& history,const BundleDesign& design,int horizon,int bundles,std::mt19937& rng) {
    if (bundles<=0 || horizon<=0) throw std::invalid_argument("invalid bundle dimensions");
    std::vector<Scenario> result;
    for (int j=0;j<bundles;++j) for (std::size_t m=0;m<design.modes.size();++m) {
        const int K=design.multiplicities.at(m);
        if (K<0) throw std::invalid_argument("negative multiplicity");
        if (K==0) continue;
        const std::map<int,std::map<std::string,double>> weights={{id,{{design.modes[m],1.0}}}};
        auto draws=sample_scenarios({{id,state}},{{id,history}},&weights,horizon,K,{},nullptr,&rng,result.size());
        for (auto& draw:draws) { draw.bundle_id=j; result.push_back(std::move(draw)); }
    }
    return result;
}

int scenario_group_count(const std::vector<Scenario>& scenarios) {
    std::set<int> groups;
    for (const auto& scenario:scenarios) groups.insert(scenario.support_id());
    return groups.size();
}

std::vector<CollisionConstraint> prune_bundle_constraints_exact(const std::vector<CollisionConstraint>& constraints) {
    using Key=std::tuple<int,int,int,int,double,double,double>;
    std::map<Key,std::size_t> rows;
    std::vector<CollisionConstraint> kept;
    for (const auto& c:constraints) {
        if (c.k<=0) continue; // Same future-stage domain as the ordinary reducer.
        if (!c.a.allFinite() || !std::isfinite(c.b) || !std::isfinite(c.disc_offset))
            throw std::invalid_argument("nonfinite bundle facet");
        const Key key(c.scenario_id,c.obstacle_id,c.k,c.disc_index,c.disc_offset,c.a.x(),c.a.y());
        const auto [it,inserted]=rows.emplace(key,kept.size());
        if (inserted) kept.push_back(c);
        else if (c.b>kept[it->second].b) kept[it->second]=c;
    }
    return kept;
}
}
