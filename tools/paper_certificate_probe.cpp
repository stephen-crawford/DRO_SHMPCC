// Experiment-only batch interface to the production CP, allocation, and SH formulas.
#include "dro.hpp"
#include "dynamics.hpp"
#include "primal_ot.hpp"
#include "wasserstein_radius_calibration.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
using namespace dro_mpc;
using Clock=std::chrono::steady_clock;
int main() {
    std::cout<<std::setprecision(17);
    RuntimeConfig cfg; cfg.normalize();
    std::string op;
    while(std::cin>>op) {
        if(op=="S") {
            double z;std::cin>>z;
            int s=cfg.compute_required_scenarios_for_risk(.05/z,6,2);
            std::cout<<s<<' '<<(cfg.compute_effective_epsilon(s,8)<=.05/z)<<std::endl;
            continue;
        }
        int m;double beta;std::cin>>m>>beta;
        std::vector<int> counts(m);for(auto& n:counts)std::cin>>n;
        const int g=std::accumulate(counts.begin(),counts.end(),0);
        std::vector<double> p(m),r(m),q(m),ub(m);
        std::map<std::string,double> nominal;std::map<std::string,int> observed;
        auto catalog=create_obstacle_mode_models(.1);
        const std::vector<std::string> names{"constant_velocity","turn_left","turn_right","decelerating","accelerating","stop"};
        std::map<std::string,ModeModel> modes;
        for(int i=0;i<m;++i) {
            std::string key="m"+std::to_string(i);
            ModeModel model;
            if(i<6) model=catalog.at(names[i]);
            else { model=catalog.at("constant_velocity");model.b(1)=.02*(i-5); }
            model.mode_id=key;modes[key]=model;
            p[i]=(counts[i]+.5)/(g+.5*m);nominal[key]=p[i];observed[key]=counts[i];
        }
        DROConfig dc;dc.radius_calibration.confidence_beta=beta;DRO dro(dc);
        std::vector<EgoState> frozen;
        for(int k=0;k<=20;++k)frozen.emplace_back(.15*k,0,0,1.5);
        auto preliminary=dro.compute_worst_case_weights(nominal,observed,ObstacleState(3.0,1.5,-.6,0),modes,frozen,20,.6,.35,.0,20,3,4.);
        auto D=preliminary.transport_cost_matrix;
        for(int i=0;i<m;++i)r[i]=preliminary.risk_per_mode.at("m"+std::to_string(i));
        auto start=Clock::now();
        auto cp=finite_sample_wasserstein_radius(counts,p,D,beta);
        double cp_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        std::vector<double> u(m);
        start=Clock::now();
        for(int i=0;i<m;++i){double rest=0;for(int j=0;j<m;++j)if(j!=i)rest+=cp.lower[j];u[i]=std::min(cp.upper[i],1-rest);}
        double u_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        start=Clock::now();auto a=solve_dominating_ot(p,r,D,cp.rho,.01,u);
        double q_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        if(!a.transport.solved)return 2;
        for(int j=0;j<m;++j){std::vector<double> objective(m,0);objective[j]=1;auto b=solve_primal_ot(p,objective,D,cp.rho);if(!b.solved)return 3;ub[j]=b.expected_risk;for(int i=0;i<m;++i)q[j]+=a.transport.plan[i][j];}
        std::cout<<cp.rho<<' '<<cp.vertex_count<<' '<<cp_ms<<' '<<u_ms<<' '<<q_ms;
        for(const auto& vec:{p,r,cp.lower,cp.upper,u,ub,q})for(double v:vec)std::cout<<' '<<v;
        for(const auto& row:D)for(double v:row)std::cout<<' '<<v;
        std::cout<<std::endl;
    }
}
