#!/usr/bin/env python3
"""Certificate-design and obstacle-scaling studies using production C++ calculations."""
import argparse,csv,json,subprocess,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1]

def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seeds',type=int,default=100);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen([str(ROOT/'build-base/paper_certificate_probe')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def query(line):
        proc.stdin.write(line+'\n');proc.stdin.flush();line=proc.stdout.readline()
        if not line:raise RuntimeError('production probe terminated')
        return np.fromstring(line,sep=' ')
    def samples(z):
        s,ok=query(f'S {z:.17g}');return int(s),int(ok)
    def instance(m,g,seed,beta):
        rng=np.random.default_rng(seed);truth=.65**np.arange(m);truth/=truth.sum();counts=rng.multinomial(g,truth)
        v=query('I '+str(m)+' '+str(beta)+' '+' '.join(map(str,counts)))
        rho,vertices,cp_ms,u_ms,q_ms=v[:5];p,r,lo,hi,u,ub,q=v[5:5+7*m].reshape(7,m);D=v[5+7*m:].reshape(m,m)
        return dict(m=m,g=g,seed=seed,beta=beta,truth=truth,counts=counts,rho=rho,vertices=int(vertices),cp_ms=cp_ms,u_ms=u_ms,q_ms=q_ms,p=p,r=r,lo=lo,hi=hi,u=u,ub=ub,q=q,D=D)
    mode_rows=[];cost_rows=[];arm_rows=[];radius_rows=[];scaling=[];scaling_obs=[]
    for m in [2,3,5]:
        for g in [10,25,50,100,250,1000]:
            for seed in range(a.seeds):
                x=instance(m,g,100000+m*10000+g*100+seed,.05);p,r,u,D=x['p'],x['r'],x['u'],x['D'];minimum=u.sum()
                ident=f'cal_m{m}_g{g}_s{seed}'
                for j in range(m):mode_rows.append(dict(instance=ident,M=m,g=g,seed=seed,mode=j,count=x['counts'][j],p_true=x['truth'][j],p_nominal=p[j],risk=r[j],L=x['lo'][j],U=x['hi'][j],u_C=u[j],u_B=x['ub'][j]))
                for i in range(m):
                    for j in range(m):cost_rows.append(dict(instance=ident,source=i,destination=j,D=D[i,j]))
                # t=1/zeta: maximizing t makes q_j >= t*u_j linear.
                source=np.zeros((m,m*m+1));destination=np.zeros_like(source)
                for j in range(m):source[j,j*m:(j+1)*m]=1;destination[j,j:m*m:m]=1
                budget=np.r_[D.ravel(),0.];ineq=-destination.copy();ineq[:,-1]=u
                start=time.perf_counter();objective=np.zeros(m*m+1);objective[-1]=-1
                lp=linprog(objective,A_ub=np.vstack([budget,ineq]),b_ub=np.r_[x['rho'],np.zeros(m)],A_eq=source,b_eq=p,bounds=(0,None),method='highs')
                water_ms=1000*(time.perf_counter()-start)
                if not lp.success:raise RuntimeError(lp.message)
                water=destination@lp.x
                start=time.perf_counter();env=u/minimum;env_ms=1000*(time.perf_counter()-start)
                for arm,q,ms in [('nominal',p,0.),('envelope_optimal',env,env_ms),('wasserstein_min_zeta',water,water_ms),('full_proposed',x['q'],x['q_ms'])]:
                    z=max(u/q);S,ok=samples(z)
                    if arm=='wasserstein_min_zeta':assert abs(z-1/lp.x[-1])<1e-6
                    assert z>=minimum-1e-7
                    if arm=='full_proposed':assert z<=max(u/p)+1e-7 and min(q[r>=.01]-p[r>=.01],default=0)>=-1e-7
                    arm_rows.append(dict(instance=ident,M=m,g=g,seed=seed,arm=arm,beta_DRO=.05,epsilon=.05,rho=x['rho'],vertices=x['vertices'],minimum_zeta=minimum,zeta=z,normalized_zeta=z/minimum,epsilon_SH=.05/z,required_S=S,S_bound_satisfied=ok,old_new_envelope_ratio=x['ub'].sum()/minimum,cp_covered=int(np.all(x['truth']>=x['lo']) and np.all(x['truth']<=x['hi'])),risk_expectation=q@r,risk_lift=(q-p)@r,cp_radius_total_ms=x['cp_ms'],envelope_formula_ms=x['u_ms'],allocation_ms=ms,construction_total_ms=x['cp_ms']+ms,q=json.dumps(q.tolist())))
            print(f'certificate M={m} g={g}: {a.seeds} seeds',flush=True)
    for m in range(2,9):
        for seed in range(a.seeds):
            x=instance(m,100,800000+m*1000+seed,.05)
            radius_rows.append(dict(M=m,g=100,seed=seed,vertices=x['vertices'],rho=x['rho'],cp_radius_total_ms=x['cp_ms'],envelope_formula_ms=x['u_ms'],allocation_ms=x['q_ms'],library='production_first_'+str(min(m,6))+'_plus_affine_drifts' if m>6 else 'production_modes'))
    # Fixed beta_v=.05/6 across all V prevents a changing confidence allocation
    # from confounding the product scaling. Prefix pairing across V is retained.
    for seed in range(a.seeds):
        zs=np.ones(3)
        for v in range(1,7):
            x=instance(3,100,900000+seed*10+v,.05/6);u=x['u'];factors=[u.sum(),max(u/x['p']),max(u/x['q'])];zs*=factors
            scaling_obs.append(dict(seed=seed,obstacle=v,g=100,M=3,beta_obstacle=.05/6,minimum_zeta=factors[0],nominal_zeta=factors[1],risk_zeta=factors[2],rho=x['rho'],counts=json.dumps(x['counts'].tolist())))
            for arm,z in zip(['envelope_minimum','nominal','full_proposed'],zs):
                S,ok=samples(z)
                scaling.append(dict(seed=seed,V=v,M=3,g=100,arm=arm,beta_obstacle=.05/6,beta_joint_used=v*.05/6,zeta_joint=z,log_zeta_joint=np.log(z),epsilon_SH=.05/z,required_S=S,S_bound_satisfied=ok))
    proc.stdin.close();assert proc.wait()==0
    for name,data in [('certificate_instances.csv',arm_rows),('certificate_modes.csv',mode_rows),('certificate_ground_costs.csv',cost_rows),('radius_runtime.csv',radius_rows),('scaling_instances.csv',scaling),('scaling_obstacles.csv',scaling_obs)]:write(a.output/name,data)
    def summarize(data,keys,values):
        groups={}
        for r in data:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
        result=[]
        for k,group in sorted(groups.items()):
            row=dict(zip(keys,k));row['instances']=len(group)
            for value in values:
                vals=[r[value] for r in group]
                for suffix,pct in [('p10',10),('median',50),('p90',90),('max',100)]:row[value+'_'+suffix]=float(np.percentile(vals,pct))
            result.append(row)
        return result
    write(a.output/'table1_certificate_summary.csv',summarize(arm_rows,['M','g','arm'],['normalized_zeta','zeta','minimum_zeta','epsilon_SH','required_S','construction_total_ms','old_new_envelope_ratio']))
    write(a.output/'radius_runtime_summary.csv',summarize(radius_rows,['M'],['vertices','cp_radius_total_ms','allocation_ms']))
    write(a.output/'figure1_scaling_summary.csv',summarize(scaling,['V','arm'],['zeta_joint','log_zeta_joint','epsilon_SH','required_S']))
    print('Finished certificate and scaling CSVs',flush=True)
if __name__=='__main__':main()
