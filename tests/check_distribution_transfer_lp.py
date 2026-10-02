#!/usr/bin/env python3
"""Independent SciPy/HiGHS audit of the production Q_risk LP (optional SciPy)."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
DRIVER = r'''#include "primal_ot.hpp"
#include <iostream>
#include <iomanip>
int main() {
    int m;
    while (std::cin >> m) {
        std::vector<double> p(m), r(m);
        for (auto& x:p) std::cin >> x;
        for (auto& x:r) std::cin >> x;
        double rho, threshold; std::cin >> rho >> threshold;
        std::vector<std::vector<double>> d(m,std::vector<double>(m));
        for(auto& row:d) for(auto& x:row) std::cin >> x;
        auto a=dro_mpc::solve_dominating_ot(p,r,d,rho,threshold);
        std::cout << std::setprecision(17) << a.transport.solved;
        if(a.transport.solved) {
            std::cout << ' ' << a.transport.expected_risk << ' ' << a.domination;
            for(double u:a.envelope) std::cout << ' ' << u;
            for(int j=0;j<m;++j) {
                double q=0;for(int i=0;i<m;++i)q+=a.transport.plan[i][j];
                std::cout << ' ' << q;
            }
        }
        std::cout << '\n';
    }
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=int,default=240)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(20261001)
    problems=[]
    for case in range(args.cases):
        m=2+case%5
        p=rng.dirichlet(np.ones(m))
        r=rng.uniform(0,2,m)
        points=rng.normal(size=(m,2))
        d=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
        if case%12==0: d[:]=0
        rho=0.0 if case%11==0 else rng.uniform(0,1)*d.max()
        if case%13==0: r[:]=0
        problems.append((p,r,d,rho,0.5))
    text=''.join(' '.join(map(str,[len(p),*p,*r,rho,t,*d.ravel()]))+'\n'
                 for p,r,d,rho,t in problems)
    with tempfile.TemporaryDirectory(prefix='acc-transfer-lp-') as tmp:
        source=Path(tmp)/'probe.cpp';source.write_text(DRIVER)
        binary=Path(tmp)/'probe'
        subprocess.run(['c++','-std=c++17','-O2','-I',str(ROOT/'include'),
                        str(source),str(ROOT/'src/primal_ot.cpp'),'-o',str(binary)],check=True)
        rows=subprocess.run([str(binary)],input=text,text=True,capture_output=True,check=True).stdout.splitlines()
    errors=[];max_objective_error=0.;max_envelope_error=0.;max_constraint_error=0.
    for case,(problem,line) in enumerate(zip(problems,rows)):
        p,r,d,rho,threshold=problem;m=len(p)
        output=np.fromstring(line,sep=' ')
        if not output.size or output[0]!=1:
            errors.append(dict(case=case,error='production solver did not solve'));continue
        objective,zeta=output[1:3];u=output[3:3+m];q=output[3+m:]
        source=np.zeros((m,m*m));destination=np.zeros_like(source)
        for i in range(m):
            source[i,i*m:(i+1)*m]=1
            destination[i,i::m]=1
        def lp(c,lower=None):
            a=d.reshape(1,-1);b=np.array([rho])
            if lower is not None:
                a=np.concatenate((a,-destination));b=np.concatenate((b,-lower))
            result=linprog(c,A_ub=a,b_ub=b,A_eq=source,b_eq=p,bounds=(0,None),method='highs')
            if not result.success: raise RuntimeError(result.message)
            return result
        envelope=np.array([-lp(-destination[j]).fun for j in range(m)])
        lower=envelope/np.max(envelope/p)
        lower=np.maximum(lower,np.where(r>=threshold,p,0.))
        optimum=-lp(-np.tile(r,m),lower).fun
        oe=abs(objective-optimum);ue=float(np.max(abs(u-envelope)))
        ce=max(float(np.max(lower-q)),abs(float(q.sum())-1),float(np.max(envelope-zeta*q)),0.)
        max_objective_error=max(max_objective_error,oe)
        max_envelope_error=max(max_envelope_error,ue)
        max_constraint_error=max(max_constraint_error,ce)
        if max(oe,ue,ce)>1e-7: errors.append(dict(case=case,objective_error=oe,envelope_error=ue,constraint_error=ce))
    if len(rows)!=len(problems): errors.append(dict(error='missing output rows'))
    report=dict(cases=len(problems),seed=20261001,mode_counts=[2,3,4,5,6],tolerance=1e-7,
                max_objective_error=max_objective_error,max_envelope_error=max_envelope_error,
                max_constraint_error=max_constraint_error,errors=errors)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return int(bool(errors))

if __name__=='__main__':
    raise SystemExit(main())
