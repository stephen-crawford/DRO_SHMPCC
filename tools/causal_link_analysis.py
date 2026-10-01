"""Read-only causal follow-up statistics. Unit of outcome inference is a seed."""
import csv
import math
from collections import defaultdict
from pathlib import Path


def read(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def undercoverage(p, k, size):
    """P(Binomial(size,p) < k), including degenerate probabilities."""
    if not 0 <= p <= 1 or size < 0 or int(size) != size:
        raise ValueError('invalid binomial parameters')
    return sum(math.comb(size, j) * p**j * (1-p)**(size-j)
               for j in range(max(0, min(k, size+1))))


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return '', ''
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return max(0., center-radius), min(1., center+radius)


def select_separation(rows):
    """Selection uses only frozen risk/coverage, never closed-loop outcomes."""
    dangerous = sorted((r for r in rows if float(r['r']) > 0),
                       key=lambda r: (float(r['delta_u2']), r['candidate']))
    if not dangerous:
        raise ValueError('no candidate with positive dangerous-mode risk')
    return [dict(dangerous[-1], stratum='high'),
            dict(dangerous[len(dangerous)//2], stratum='medium'),
            dict(min(rows, key=lambda r: (abs(float(r['delta_u2'])), r['candidate'])), stratum='near_zero')]


def qualify_braking(rows, expected_seeds, reach_threshold=.95):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['arm']].append(r)
    if any(len(grouped[a]) != expected_seeds or
           any(r.get('status') != 'OK' for r in grouped[a]) or
           sum(int(r['switch_reached']) for r in grouped[a])/expected_seeds < reach_threshold
           for a in ['sh_mpcc', 'sh_mpcc_dro']):
        return False
    nominal = grouped['sh_mpcc']
    failure = sum(int(r['collision']) or r['termination'] == 'no_admissible_control' for r in nominal)/expected_seeds
    return .2 <= failure <= .5


def frozen_report(folder):
    folder = Path(folder)
    across = next(r for r in read(folder/'weights.csv') if r['mode'] == 'across')
    grouped = defaultdict(list)
    for row in read(folder/'draws.csv'):
        grouped[row['law']].append(row)
    coverage = []
    for law, rows in grouped.items():
        p = float(across['q' if law == 'wdro' else 'p'])
        for k in [1, 2, 3]:
            n = len(rows); hits = sum(int(r['n_d']) < k for r in rows)
            theory = undercoverage(p, k, int(rows[0]['S']))
            lo, hi = wilson(hits, n)
            # Simultaneous, distribution-free tolerance for the six planned coverage checks.
            tolerance = math.sqrt(math.log(2*6/.01)/(2*n))
            coverage.append(dict(law=law,k=k,trials=n,undercovered=hits,empirical=hits/n,
                theory=theory,ci95_low=lo,ci95_high=hi,simultaneous99_tolerance=tolerance,
                coverage_check_pass=int(abs(hits/n-theory) <= tolerance)))
    write(folder/'coverage_check.csv', coverage)
    trials = read(folder/'conditional.csv')
    estimates = []
    for law in ['nominal', 'wdro']:
        for n in range(int(across['S'])+1):
            rr = [r for r in trials if r['law'] == law and int(r['n_d']) == n]
            for metric in ['collision','refusal','horizon_completed']:
                successes = sum(int(r[metric]) for r in rr)
                lo, hi = wilson(successes, len(rr))
                estimates.append(dict(law=law,n_d=n,metric=metric,seeds=len(rr),events=successes,
                    rate=successes/len(rr) if rr else '',ci95_low=lo,ci95_high=hi))
    write(folder/'conditional_estimates.csv', estimates)
    # Quota sampling over-represents rare Nd. Reweight strata by the binomial law;
    # never report the unweighted quota pool as gamma_bad/gamma_good.
    gamma = []
    for law in ['nominal','wdro']:
        p = float(across['q' if law == 'wdro' else 'p']); size = int(across['S'])
        for k in [1,2,3]:
            for group in ['bad','good']:
                ns = range(k) if group == 'bad' else range(k,size+1)
                denominator = undercoverage(p,k,size) if group == 'bad' else 1-undercoverage(p,k,size)
                for metric in ['collision','refusal']:
                    lower=upper=estimate=missing=0.
                    for n in ns:
                        mass = math.comb(size,n)*p**n*(1-p)**(size-n)
                        rr = [r for r in trials if r['law']==law and int(r['n_d'])==n]
                        if not rr:
                            missing += mass; upper += mass; continue
                        events=sum(int(r[metric]) for r in rr)
                        # Bonferroni simultaneous Wilson intervals are approximate, not exact bounds.
                        from statistics import NormalDist
                        z=NormalDist().inv_cdf(1-.05/(2*(size+1)))
                        lo,hi=wilson(events,len(rr),z)
                        estimate += mass*events/len(rr); lower += mass*lo; upper += mass*hi
                    gamma.append(dict(law=law,k=k,group=group,metric=metric,
                        observed_mass=denominator-missing,missing_mass=missing,
                        estimate_observed_contribution=estimate/denominator if denominator else '',
                        partial_identification_low=estimate/denominator if denominator else '',
                        partial_identification_high=(estimate+missing)/denominator if denominator else '',
                        approximate_simultaneous95_low=lower/denominator if denominator else '',
                        approximate_simultaneous95_high=upper/denominator if denominator else ''))
    write(folder/'gamma_estimates.csv', gamma)
    return coverage


def paired_summary(root):
    output=[]
    for name in ['selected_p50','selected_p20','negative_control']:
        grouped=defaultdict(dict)
        for row in read(Path(root)/name/'report/stress_outcomes.csv'):
            grouped[row['pair_case'],row['seed']][row['solver_style']]=row
        counts=defaultdict(lambda: [0,0,0,0])
        for (case,seed), arms in grouped.items():
            if not all(a in arms and arms[a]['trial_status']=='OK' for a in ['sh_mpcc','sh_mpcc_dro']):continue
            a,b=(int(arms[s]['collision']) for s in ['sh_mpcc','sh_mpcc_dro'])
            counts[case][2*a+b]+=1
        for case,(neither,wdro_only,nominal_only,both) in counts.items():
            discordant=wdro_only+nominal_only
            p=min(1.,2*sum(math.comb(discordant,i) for i in range(min(wdro_only,nominal_only)+1))/2**discordant) if discordant else 1.
            output.append(dict(case=case,paired_seeds=neither+wdro_only+nominal_only+both,
                neither_collision=neither,wdro_only_collision=wdro_only,nominal_only_collision=nominal_only,
                both_collision=both,exact_mcnemar_p=p))
    write(Path(root)/'paired_collision_summary.csv',output)
    return output
