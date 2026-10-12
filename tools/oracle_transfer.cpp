#include "oracle_transfer.hpp"
#include "config.hpp"
#include <iomanip>
#include <iostream>
int main(){
    using namespace dro_mpc;
    std::cout<<std::setprecision(17);
    std::string command;
    try {while(std::cin>>command) {
        if(command=="samples") {
            double epsilon;int h;std::cin>>epsilon>>h;RuntimeConfig cfg;
            cfg.mpc.sampling.chance_of_certificate_violation=.01;
            // eta=1 makes the transfer vacuous; evaluate the limit below 1 with the existing API.
            if(epsilon==1)epsilon=std::nextafter(1.,0.);
            std::cout<<cfg.compute_required_scenarios_for_risk(epsilon,h,0)<<std::endl;continue;
        }
        int M;double rho,value,eta;std::cin>>M>>rho>>value>>eta;
        if(M<1 || M>4)throw std::invalid_argument("mode dimension");
        std::vector<double> p(M),q(M),b(M);std::vector<std::vector<double>> D(M,std::vector<double>(M));
        for(auto& x:p)std::cin>>x;for(auto& x:q)std::cin>>x;for(auto& x:b)std::cin>>x;
        for(auto& row:D)for(auto& x:row)std::cin>>x;
        if(!std::cin)throw std::runtime_error("malformed transfer input");
        if(command=="psi")std::cout<<oracle_transfer::psi(p,q,b,D,rho,eta)<<std::endl;
        else if(command=="eta")std::cout<<oracle_transfer::eta_star(p,q,b,D,rho,value)<<std::endl;
        else if(command=="dro") {
            auto result=solve_primal_ot(p,b,D,rho);if(!result.solved)throw std::runtime_error("OT failed");
            std::vector<double> weights(M,0);for(int i=0;i<M;++i)for(int j=0;j<M;++j)weights[j]+=result.plan[i][j];
            for(auto x:weights)std::cout<<x<<' ';std::cout<<std::endl;
        } else throw std::invalid_argument("unknown command");
    }}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
