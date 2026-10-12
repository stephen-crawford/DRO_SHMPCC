#pragma once
// Experiment-only finite vertex enumeration; delegates every OT problem to production.
#include "primal_ot.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace dro_mpc::oracle_transfer {
inline double psi(const std::vector<double>& p,const std::vector<double>& q,
                  const std::vector<double>& b,const std::vector<std::vector<double>>& D,
                  double rho,double eta) {
    const size_t M=p.size();
    if (!M || M>4 || q.size()!=M || b.size()!=M || !std::isfinite(eta) || eta<0 || eta>1)
        throw std::invalid_argument("transfer requires 1..4 modes and eta in [0,1]");
    double total=0;
    for(size_t j=0;j<M;++j) {
        if(!std::isfinite(q[j]) || q[j]<0 || !std::isfinite(b[j]) || b[j]<0 || b[j]>1 || (b[j]>0 && q[j]==0))
            throw std::invalid_argument("invalid danger bound or missing proposal support");
        total+=q[j];
    }
    if(std::abs(total-1)>1e-10)throw std::invalid_argument("proposal must sum to one");
    double worst=0;
    auto evaluate=[&](const std::vector<double>& v){
        auto result=solve_primal_ot(p,v,D,rho);
        if(!result.solved)throw std::runtime_error("primal OT failed in transfer enumeration");
        worst=std::max(worst,result.expected_risk);
    };
    for(size_t mask=0;mask<(size_t(1)<<M);++mask) {
        std::vector<double> v(M);double cost=0;
        for(size_t j=0;j<M;++j) {v[j]=(mask&(size_t(1)<<j))?b[j]:0.;cost+=q[j]*v[j];}
        if(cost<=eta)evaluate(v);
    }
    for(size_t fractional=0;fractional<M;++fractional) {
        if(q[fractional]==0 || b[fractional]==0)continue;
        for(size_t mask=0;mask<(size_t(1)<<M);++mask) {
            if(mask&(size_t(1)<<fractional))continue;
            std::vector<double> v(M);double cost=0;
            for(size_t j=0;j<M;++j)if(j!=fractional){v[j]=(mask&(size_t(1)<<j))?b[j]:0.;cost+=q[j]*v[j];}
            v[fractional]=(eta-cost)/q[fractional];
            if(v[fractional]>=0 && v[fractional]<=b[fractional])evaluate(v);
        }
    }
    return worst;
}
inline double eta_star(const std::vector<double>& p,const std::vector<double>& q,
                       const std::vector<double>& b,const std::vector<std::vector<double>>& D,
                       double rho,double epsilon) {
    if(!(epsilon>0 && epsilon<1))throw std::invalid_argument("epsilon must be in (0,1)");
    if(psi(p,q,b,D,rho,1)<=epsilon)return 1;
    double lo=0,hi=1;
    for(int iteration=0;iteration<36;++iteration) {
        const double mid=(lo+hi)/2;
        if(psi(p,q,b,D,rho,mid)<=epsilon)lo=mid;else hi=mid;
    }
    return lo; // feasible-side numerical search; no relaxed risk inequality.
}
}
