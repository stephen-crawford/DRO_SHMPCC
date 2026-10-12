#!/usr/bin/env python3
"""Fixed-direction lexicographic replay and canonical-row/support audit."""
import argparse,csv,json,re,statistics
from pathlib import Path


def ids(value):return {int(x) for x in value.split(';') if x}


def analyze(root,tightening=0.0):
    summaries=[];native_equal=0;active_checks=0
    for seed in (77,78):
        directory=root/f'seed_{seed}'
        rows=list(csv.DictReader((directory/'slack_replays.csv').open()))
        ext=list(csv.DictReader((directory/'extrema.csv').open()));native=list(csv.DictReader((directory/'native_rows.csv').open()))
        old={r['replicate']:r for r in csv.DictReader((Path('results/fixed-normal-hard-replay')/f'seed_{seed}'/'replays.csv').open()) if r['variant']=='fixed'}
        old_groups=list(csv.DictReader((Path('results/fixed-normal-hard-replay')/f'seed_{seed}'/'groups.csv').open()))
        cardinalities=[];native_gaps=[];qp_calls=0;qp_converged=0
        assert len(rows)==100 and len({r['replicate'] for r in rows})==100
        for row in rows:
            rep=row['replicate'];before=old[rep]
            assert row['sampling_seed']==before['sampling_seed'] and row['candidate']==before['candidate']
            if tightening==0:assert row['baseline_certified']==before['certified']
            assert row['baseline_qp']==before['entered_qp']
            canonical=[r for r in ext if r['replicate']==rep];actual=[r for r in native if r['replicate']==rep]
            prior=[r for r in old_groups if r['replicate']==rep]
            assert len(canonical)==20
            expected={(r['k'],r['disc'],r['obstacle'],r['mode']):r for r in prior}
            for r in canonical:
                target=expected[(r['k'],r['disc'],r['obstacle'],r['mode'])]
                assert r['id']==target['extreme_id'] and float(r['b'])==float(target['max_b'])+tightening
                assert r['a_x']==target['a_x'] and r['a_y']==target['a_y'] and r['sample_count']=='1351'
            # Equality of complete coefficient sets proves feasible-set equality
            # for these recorded instances, without an approximate LP tolerance.
            def coefficients(data):return {(r['k'],r['disc'],float(r['a_x']),float(r['a_y']),float(r['b'])) for r in data}
            equal=coefficients(actual)==coefficients(canonical)
            native_equal+=int(equal)
            if not equal:native_gaps.append(int(rep))
            reconstruction={int(r['id']) for r in canonical};cardinalities.append(len(reconstruction))
            log=(directory/f'rep{rep}_production.log').read_text()
            for value in re.findall(r'^\[FIXED ACTIVE\] iter=\d+ ids=(.*)$',log,re.MULTILINE):
                assert ids(value)<=reconstruction,(seed,rep,'native active outside extrema',value)
                active_checks+=1
            union=re.search(r'^\[SUPPORT UNION IDS\](.*)$',log,re.MULTILINE)
            if union:assert set(map(int,union[1].split()))<=reconstruction
            trace=list(csv.DictReader((directory/f'rep{rep}_slack.csv').open()));qp_calls+=len(trace)
            qp_converged+=sum(r['converged']=='1' for r in trace)
            for qp in trace:
                if qp['converged']=='1':
                    assert ids(qp['active_ids'])<=reconstruction
                    active_checks+=1
            if row['zero_slack_candidate']=='1':
                assert row['all_qps_converged']=='1' and float(row['final_violation'])==0
        feasible=[r for r in rows if r['augmented_feasible']=='1']
        summary=dict(context_seed=seed,batches=100,qp_calls=qp_calls,converged_qp_calls=qp_converged,
                     **{key:sum(r[key]=='1' for r in rows) for key in ('all_qps_converged','augmented_feasible','strict_zero','slack_le_1e4','zero_slack_candidate')},
                     raw_slack_nonpositive=sum(r['all_qps_converged']=='1' and float(r['slack'])<=0 for r in rows),
                     median_final_active=statistics.median(int(r['n_active']) for r in feasible),
                     median_cumulative_active=statistics.median(int(r['union_active']) for r in feasible),
                     max_cumulative_active=max(int(r['union_active']) for r in rows),
                     reconstruction_size_min=min(cardinalities),reconstruction_size_median=statistics.median(cardinalities),
                     reconstruction_size_max=max(cardinalities),reconstruction_size_over_h10=sum(n>10 for n in cardinalities),
                     native_coefficient_mismatch_replicates=native_gaps,
                     tightening=tightening,
                     baseline_certified=sum(r['baseline_certified']=='1' for r in rows))
        if 'margin_candidate' in rows[0]:
            candidates=[r for r in rows if r['margin_candidate']=='1']
            summary['margin_candidates']=len(candidates)
            summary['min_candidate_projection_reserve']=min((float(r['projection_reserve']) for r in candidates),default=None)
            for r in candidates:
                assert float(r['lifted_slack'])<=tightening and float(r['projection_reserve'])>=0
                assert float(r['physical_clearance'])>=0
                assert float(r['final_violation'])<=float(r['lifted_slack'])
        summaries.append(summary)
    output=dict(contexts=summaries,native_equal_batches=native_equal,active_membership_checks=active_checks)
    (root/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    with (root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--tightening',type=float,default=0.0)
    args=parser.parse_args();analyze(args.root,args.tightening)
