#!/usr/bin/env python3
"""Independent CP/polytope/transport audit of the updated paper implementation."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta as beta_distribution

ROOT = Path(__file__).resolve().parents[1]
DRIVER = r'''
#include "primal_ot.hpp"
#include "wasserstein_radius_calibration.hpp"
#include <iostream>
#include <iomanip>
#include <numeric>
int main() {
    int m;
    while (std::cin >> m) {
        std::vector<int> counts(m);
        for (auto& x:counts) std::cin >> x;
        double beta, threshold; std::cin >> beta >> threshold;
        std::vector<double> risk(m),p(m);
        for (auto& x:risk) std::cin >> x;
        std::vector<std::vector<double>> d(m,std::vector<double>(m));
        for (auto& row:d) for (auto& x:row) std::cin >> x;
        double n=std::accumulate(counts.begin(),counts.end(),0.0);
        for (int j=0;j<m;++j) p[j]=(counts[j]+0.5)/(n+0.5*m);
        auto cp=dro_mpc::finite_sample_wasserstein_radius(counts,p,d,beta);
        auto a=dro_mpc::solve_dominating_ot(p,risk,d,cp.rho,threshold,cp.coordinate_envelope);
        std::cout << std::setprecision(17) << a.transport.solved << ' ' << cp.rho
                  << ' ' << a.transport.expected_risk << ' ' << a.domination;
        for (double x:cp.lower) std::cout << ' ' << x;
        for (double x:cp.upper) std::cout << ' ' << x;
        for (double x:cp.coordinate_envelope) std::cout << ' ' << x;
        for (int j=0;j<m;++j) {
            double q=0; if (a.transport.solved) for (int i=0;i<m;++i) q+=a.transport.plan[i][j];
            std::cout << ' ' << q;
        }
        std::cout << '\n';
    }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=120)
    args = parser.parse_args()
    rng = np.random.default_rng(20261002)
    problems = []
    for k in range(args.cases):
        m = 1 + k % 5
        counts = rng.multinomial(0 if k % 11 == 0 else int(rng.integers(1, 200)), rng.dirichlet(np.ones(m)))
        points = rng.normal(size=(m, 2))
        d = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        if k % 7 == 0:
            d[:] = 0
        risk = rng.uniform(0, 2, m)
        if k % 13 == 0:
            risk[:] = 0
        problems.append((counts, d, risk, 1e-16 if k % 17 == 0 else .025, .5))
    payload = ''.join(' '.join(map(str, [len(c), *c, b, t, *r, *d.ravel()])) + '\n'
                      for c, d, r, b, t in problems)
    with tempfile.TemporaryDirectory(prefix='paper-cp-audit-') as td:
        source = Path(td) / 'driver.cpp'
        binary = Path(td) / 'driver'
        source.write_text(DRIVER)
        subprocess.run(['c++', '-std=c++17', '-O2', '-I', str(ROOT/'include'), str(source),
                        str(ROOT/'src/primal_ot.cpp'), str(ROOT/'src/wasserstein_radius_calibration.cpp'),
                        '-o', str(binary)], check=True)
        output = subprocess.run([str(binary)], input=payload, text=True, capture_output=True, check=True)
    lines = output.stdout.splitlines()
    assert len(lines) == len(problems)
    maxima = dict(cp_endpoints=0., envelope=0., radius=0., risk_objective=0., feasibility=0.)
    errors = []
    for k, ((counts, d, risk, beta, threshold), line) in enumerate(zip(problems, lines)):
        m = len(counts)
        n = sum(counts)
        p = (counts + .5)/(n + m*.5)
        values = np.fromstring(line, sep=' ')
        solved, rho, objective, zeta = values[:4]
        lower, upper, envelope, q = values[4:].reshape(4, m)
        if solved != 1:
            errors.append(dict(case=k, error='production allocation unsolved'))
            continue
        tail = beta/(2*m)
        lo = np.array([0. if x == 0 else beta_distribution.ppf(tail, x, n-x+1) for x in counts])
        hi = np.array([1. if x == n else beta_distribution.isf(tail, x+1, n-x) for x in counts])
        reference_u = []
        for j in range(m):
            c = -np.eye(m)[j]
            res = linprog(c, A_eq=np.ones((1,m)), b_eq=[1.], bounds=list(zip(lo,hi)), method='highs')
            assert res.success, res.message
            reference_u.append(-res.fun)
        reference_u = np.array(reference_u)
        source = np.zeros((m,m*m)); destination = np.zeros_like(source)
        for i in range(m):
            source[i,i*m:(i+1)*m] = 1
            destination[i,i::m] = 1
        # Independently enumerate each possible free coordinate; evaluate W_D
        # with HiGHS instead of the production residual-network transport solver.
        radius = 0.
        for free in range(m):
            fixed = [i for i in range(m) if i != free]
            for bits in itertools.product((0,1), repeat=m-1):
                vertex = np.zeros(m)
                for j, bit in zip(fixed,bits): vertex[j] = hi[j] if bit else lo[j]
                vertex[free] = 1 - vertex.sum()
                if vertex[free] < lo[free]-1e-12 or vertex[free] > hi[free]+1e-12: continue
                res = linprog(d.ravel(), A_eq=np.vstack((source,destination)),
                              b_eq=np.r_[p,vertex], bounds=(0,None), method='highs')
                assert res.success, res.message
                radius = max(radius,res.fun)
        floors = reference_u / max(reference_u/p)
        floors = np.maximum(floors,np.where(risk >= threshold,p,0))
        res = linprog(-np.tile(risk,m), A_ub=np.vstack((d.ravel(),-destination)),
                      b_ub=np.r_[radius,-floors], A_eq=source, b_eq=p, bounds=(0,None), method='highs')
        assert res.success, res.message
        transport = linprog(d.ravel(), A_eq=np.vstack((source,destination)),
                            b_eq=np.r_[p,q], bounds=(0,None), method='highs')
        assert transport.success, transport.message
        residuals = dict(cp_endpoints=max(abs(lower-lo).max(),abs(upper-hi).max()),
                         envelope=abs(envelope-reference_u).max(), radius=abs(rho-radius),
                         risk_objective=abs(objective+res.fun),
                         feasibility=max(0., max(floors-q), abs(q.sum()-1),
                                         max(reference_u-zeta*q), transport.fun-radius,
                                         zeta-max(reference_u/p)))
        for key,value in residuals.items(): maxima[key] = max(maxima[key],float(value))
        if max(residuals.values()) > 1e-7:
            errors.append(dict(case=k, **{key:float(value) for key,value in residuals.items()}))
    report = dict(cases=len(problems), seed=20261002, mode_counts=[1,2,3,4,5],
                  tolerance=1e-7, maxima=maxima, failures=errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return int(bool(errors))

if __name__ == '__main__':
    raise SystemExit(main())
