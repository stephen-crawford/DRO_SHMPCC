// Observational serialization only; no RNG or controller decisions.
#ifndef DRO_MPC_CERTIFICATION_SNAPSHOT_HPP
#define DRO_MPC_CERTIFICATION_SNAPSHOT_HPP
#include "collision_constraints.hpp"
#include "config.hpp"
#include <iomanip>
#include <sstream>
namespace dro_mpc { namespace diagnostic {
inline std::string json_string(const std::string& s) {
    std::ostringstream o; o << '"';
    for (unsigned char c:s) {
        if(c=='"'||c=='\\') o << '\\' << c;
        else if(c<32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else o << c;
    }
    o << '"'; return o.str();
}
inline void number(std::ostream& o,double v) { if(std::isfinite(v)) o<<v; else o<<"null"; }
template<class T> inline void ids(std::ostream& o,const T& values) {
    o<<'['; bool first=true; for(int v:values) { if(!first)o<<','; first=false; o<<v; } o<<']';
}
inline std::string certification_snapshot(const RuntimeConfig& c,const MPCResult& r,
    const SolveAttemptDiagnostics& a,const std::map<int,ObstacleState>& obstacles,
    const std::map<int,ModeHistory>& histories,const std::set<int>& removed) {
    std::ostringstream o; o<<std::setprecision(17)<<std::boolalpha;
    o<<"{\"schema_version\":1,\"success\":"<<r.success
     <<",\"dro_enabled\":"<<a.dro_enabled<<",\"used_fallback\":"<<r.used_fallback
     <<",\"bundle_sampling\":"<<r.bundle_sampling
     <<",\"certificate_status\":"<<json_string(safe_horizon_certificate_status_name(r.certificate_status))
     <<",\"horizon\":"<<c.mpc.horizon<<",\"sample_count\":"<<r.sampled_scenarios
     <<",\"support_evaluated\":"<<(r.support_iterations_evaluated>0)
     <<",\"support_count_used_for_certificate\":"<<r.support_size
     <<",\"support_limit\":"<<r.support_limit
     <<",\"nonremoved_support_cap\":"<<c.mpc.constraints.support_cap_n_bar
     <<",\"removal_budget\":"<<c.mpc.constraints.scenario_removal_budget
     <<",\"beta_cp\":"<<c.dro.solver.radius_calibration.confidence_beta
     <<",\"beta_cert\":"<<c.mpc.sampling.chance_of_certificate_violation
     <<",\"collision_radius\":"<<c.combined_radius()
     <<",\"markov_jump_system\":"<<c.mpc.sampling.markov_jump_system
     <<",\"removed_ids\":"; ids(o,removed);
    o<<",\"support_active\":"; ids(o,r.active_scenarios);
    o<<",\"support_union\":"; ids(o,r.support_scenarios);
    o<<",\"trajectory\":[";
    for(size_t k=0;k<r.ego_trajectory.size();++k) {
        if(k)o<<',';
        const auto& x=r.ego_trajectory[k];
        o<<'['; number(o,x.x);o<<',';number(o,x.y);o<<',';number(o,x.theta);o<<',';number(o,x.v);o<<',';number(o,x.s);o<<']';
    }
    o<<"],\"disc_centers\":[";
    for(size_t k=0;k<r.ego_trajectory.size();++k) {
        if(k)o<<',';
        o<<'[';
        auto centers=compute_ego_disc_positions(r.ego_trajectory[k],c.mpc.ego.num_discs,c.mpc.ego.length);
        for(size_t d=0;d<centers.size();++d) { if(d)o<<',';o<<'[';number(o,centers[d].x());o<<',';number(o,centers[d].y());o<<']'; }
        o<<']';
    }
    o<<"],\"obstacles\":["; bool first_obstacle=true;
    for(const auto& [id,state]:obstacles) {
        if(!first_obstacle)o<<',';
        first_obstacle=false;
        o<<"{\"obstacle\":"<<id<<",\"rho\":";
        if(a.radii.count(id))number(o,a.radii.at(id));else o<<"null";
        o<<",\"modes\":[";bool first_mode=true;
        auto h=histories.find(id);
        if(h!=histories.end()) {
            auto counts=h->second.get_mode_counts();
            for(const auto& [name,model]:h->second.available_modes) {
                if(!first_mode)o<<',';
                first_mode=false;
                o<<"{\"mode\":"<<json_string(name)<<",\"count\":"<<(counts.count(name)?counts.at(name):0)
                 <<",\"held_affine_gaussian\":"<<(!c.mpc.sampling.markov_jump_system&&model.body_lateral_displacement==0)
                 <<",\"p_hat\":";
                if(a.nominal_weights.count(id)&&a.nominal_weights.at(id).count(name))number(o,a.nominal_weights.at(id).at(name));else o<<"null";
                o<<",\"q_star\":";
                if(a.sampling_weights.count(id)&&a.sampling_weights.at(id).count(name))number(o,a.sampling_weights.at(id).at(name));else o<<"null";
                o<<",\"predictions\":[";auto mean=state;Eigen::Matrix4d cov=Eigen::Matrix4d::Zero();
                for(int k=0;k<=c.mpc.horizon;++k) {
                    if(k){o<<',';mean=model.propagate(mean);model.propagate_covariance(cov);}
                    o<<"{\"mean\":[";number(o,mean.x);o<<',';number(o,mean.y);
                    o<<"],\"covariance\":[[";number(o,cov(0,0));o<<',';number(o,cov(0,1));
                    o<<"],[";number(o,cov(1,0));o<<',';number(o,cov(1,1));o<<"]]}";
                }
                o<<"]}";
            }
        }
        o<<"]}";
    }
    o<<"]}";return o.str();
}
} }
#endif
