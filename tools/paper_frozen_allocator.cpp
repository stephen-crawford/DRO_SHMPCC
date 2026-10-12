#include "primal_ot.hpp"
#include <iomanip>
#include <iostream>
using namespace dro_mpc;
int main(){std::cout<<std::setprecision(17);int m;while(std::cin>>m){double rho;std::cin>>rho;std::vector<double>p(m),r(m),u(m);for(auto&x:p)std::cin>>x;for(auto&x:r)std::cin>>x;for(auto&x:u)std::cin>>x;std::vector<std::vector<double>>D(m,std::vector<double>(m));for(auto&row:D)for(auto&x:row)std::cin>>x;auto a=solve_dominating_ot(p,r,D,rho,.01,u);std::cout<<a.transport.solved;if(a.transport.solved){std::cout<<' '<<a.domination;for(int j=0;j<m;++j){double q=0;for(int i=0;i<m;++i)q+=a.transport.plan[i][j];std::cout<<' '<<q;}}std::cout<<std::endl;}}
