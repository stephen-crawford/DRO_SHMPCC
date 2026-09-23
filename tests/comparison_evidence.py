"""Complete comparison evidence; matches remain qualitative case selection only."""
from collections import defaultdict
import json
import math
import statistics
import random
import run_analysis_matrix as analysis

ARM_LABELS = {'sh_mpcc': 'A', 'sh_mpcc_extra': 'B', 'sh_mpcc_dro': 'C',
              'sh_mpcc_resample': 'D', 'sh_mpcc_dro_fallback': 'E'}


def flat_yaml(path):
    result={}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        key,value=line.split(':',1)
        try: value=json.loads(value.strip())
        except json.JSONDecodeError: value=value.strip().strip('"')
        result[key]=value
    return result


def run_checked_trial(case,seed,args,settings,identity):
    result=analysis.run_trial(case,seed,args,settings,identity)
    if result['status']!='OK':return result
    try:
        requested=flat_yaml(args.output/'configs'/(case['case']+'.yaml'))
        if len(result['repeats']) != settings['repeats']:
            raise ValueError('recorded repeat count differs from requested count')
        signatures=[]
        for repeat in range(settings['repeats']):
            bundle=analysis.trial_root(args.output,case,seed)/f'repeat_{repeat}'
            # Re-read evidence even on resume; cached OK status is not an integrity check.
            signature=analysis.repeat_signature(bundle,analysis.analyze(bundle,case))
            if signature != result['repeats'][repeat]['signature']:
                raise ValueError('saved numerical evidence differs from recorded repeat signature')
            signatures.append(signature)
            resolved=flat_yaml(bundle/'resolved_config.yaml')
            for key in ['shift_psi','shift_boost','boosted_mode','rare_mode','rare_mode_probability',
                        'nominal_resampling_baseline','dro_enabled','artifact_capture_attempt_diagnostics',
                        'safe_horizon_enabled','automatically_compute_sample_size']:
                if requested[key]!=resolved[key]:raise ValueError(f'resolved {key} differs from requested profile')
            if not requested['automatically_compute_sample_size'] and requested['num_scenarios']!=resolved['num_scenarios']:
                raise ValueError('resolved scenario budget differs from requested budget')
            if requested['artifact_capture_attempt_diagnostics']:
                concentration_rows(bundle)  # Requires complete, labelled transport geometry.
                attempts=analysis.rows(bundle/'attempts.csv')
                mechanism=analysis.rows(bundle/'mode_mechanism.csv')
                decisions=analysis.rows(bundle/'decisions.csv')
                if not attempts or not mechanism:raise ValueError('missing attempt/mechanism evidence')
                by_step=defaultdict(list);by_mode=defaultdict(list)
                for a in attempts:by_step[a['step']].append(a)
                for m in mechanism:by_mode[m['step'],m['attempt'],int(m['obstacle_id'])].append(m)
                for d in decisions:
                    if not requested['safe_horizon_enabled'] and (int(d['certificate_requested']) or int(d['certified'])):
                        raise ValueError('Safe Horizon certification active in disabled configuration')
                    aa=by_step[d['step']]
                    if len(aa)!=1+int(d['nominal_fallback_attempted']):raise ValueError('outer-attempt count mismatch')
                    if aa[-1]['success']!=d['success']:raise ValueError('final attempt result mismatch')
                    for a in aa:
                        if not requested['safe_horizon_enabled'] and not requested['automatically_compute_sample_size'] and int(a['scenario_count'])!=requested['num_scenarios']:
                            raise ValueError('uncertified fixed-budget attempt changed scenario count')
                        for obstacle in range(case['obstacles']):
                            mm=by_mode[a['step'],a['attempt'],obstacle]
                            if len(mm)!=case['modes_per_class']:raise ValueError('incomplete mode support evidence')
                            if sum(int(m['sampled_count']) for m in mm)!=int(a['scenario_count']):
                                raise ValueError('sample counts do not sum to actual batch size')
                            for column in ['nominal_probability','sampling_probability']:
                                if abs(sum(float(m[column]) for m in mm)-1)>1e-9:
                                    raise ValueError('unnormalized logged mode law')
        if len(set(signatures)) != 1:
            raise ValueError('same-seed numerical evidence differs across repeats')
    except (OSError,KeyError,ValueError) as error:
        result['status']='ERROR';result['error']='evidence: '+str(error)
        analysis.dump_json(analysis.trial_root(args.output,case,seed)/'result.json',result)
    return result


def exact_discordance_p(a,b):
    n=a+b
    return min(1.,2*sum(math.comb(n,k) for k in range(min(a,b)+1))/2**n) if n else 1.


def paired_interval(differences):
    if len(differences)<2:return None,None
    rng=random.Random(1729)
    estimates=sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(2000))
    return estimates[49],estimates[1949]


def nominal_prefix_matches(a,b):
    def evidence(bundle):
        return {(int(r['step']),r['obstacle_id'],r['mode']):r['nominal_probability']
                for r in analysis.rows(bundle/'mode_mechanism.csv') if r['attempt']=='0'}
    left,right=evidence(a),evidence(b)
    if not left or not right:return False
    end=min(max(k[0] for k in left),max(k[0] for k in right))
    return {k:v for k,v in left.items() if k[0]<=end}=={k:v for k,v in right.items() if k[0]<=end}


def concentration_rows(bundle):
    """Observational vertex reachability; never changes the allocator or its tolerances."""
    groups=defaultdict(dict)
    for r in analysis.rows(bundle/'mode_mechanism.csv'):
        if r['rho']!='':groups[r['step'],r['attempt'],r['obstacle_id']][r['mode']]=r
    costs=defaultdict(dict)
    for r in analysis.rows(bundle/'transport_costs.csv'):
        key=r['step'],r['attempt'],r['obstacle_id']
        edge=r['source_mode'],r['target_mode']
        if edge in costs[key]:raise ValueError('duplicate transport cost')
        costs[key][edge]=r
    if set(costs)!=set(groups):raise ValueError('transport geometry does not match WDRO attempts')
    per_mode=[];per_solve=[]
    for key,mm in sorted(groups.items()):
        dd=costs[key]
        if set(dd)!={(i,j) for i in mm for j in mm}:raise ValueError('incomplete transport matrix')
        counts={int(r['radius_observation_count']) for r in dd.values()}
        if len(counts)!=1:raise ValueError('inconsistent radius observation count')
        rho=float(next(iter(mm.values()))['rho'])
        distance={j:sum(float(mm[i]['nominal_probability'])*float(dd[i,j]['cost']) for i in mm) for j in mm}
        if any(not math.isfinite(d) or d<0 for d in distance.values()):raise ValueError('invalid vertex distance')
        minimum=min(distance.values());maximum_q=max(float(r['sampling_probability']) for r in mm.values())
        common=dict(step=int(key[0]),attempt=int(key[1]),obstacle_id=int(key[2]),
                    radius_observation_count=next(iter(counts)),rho=rho)
        for mode,d in distance.items():
            per_mode.append(dict(**common,mode=mode,vertex_distance=d,
                rho_over_vertex_distance=rho/d if d>0 else None,
                zero_vertex_distance=int(d==0),vertex_reachable=int(rho>=d),
                nominal_probability=float(mm[mode]['nominal_probability']),
                sampling_probability=float(mm[mode]['sampling_probability']),
                risk_score=float(mm[mode]['risk_score'])))
        per_solve.append(dict(**common,min_vertex_distance=minimum,
            rho_over_min_vertex_distance=rho/minimum if minimum>0 else None,
            zero_min_vertex_distance=int(minimum==0),any_vertex_reachable=int(rho>=minimum),
            max_q=maximum_q,q_gt_0_9=int(maximum_q>.9),q_near_one=int(maximum_q>=1-1e-9),
            q_exactly_one=int(maximum_q==1)))
    return per_mode,per_solve


def write_evidence_reports(output,cases,settings,results,comparisons):
    primary=[]
    grouped=defaultdict(list)
    for r in comparisons:grouped[r['pair'],r['controller_a'],r['controller_b']].append(r)
    for (pair,a,b), group in sorted(grouped.items()):
        good=[r for r in group if r['status']=='OK']
        n=len(good)
        # Collision-free here includes early stops: separately report safe completion.
        counts={(i,j):sum(bool(r['nondro_collision'])==i and bool(r['dro_collision'])==j for r in good)
                for i in [False,True] for j in [False,True]}
        delta=[int(r['dro_collision'])-int(r['nondro_collision']) for r in good]
        low,high=paired_interval(delta)
        primary.append(dict(pair=pair,profile=group[0]['profile'],controller_a=a,controller_b=b,
            arm_a=ARM_LABELS[a],arm_b=ARM_LABELS[b],
            scenario_budget=group[0].get('scenario_budget','auto'),
            expected_pairs=len(group),measured_pairs=n,error_or_pending_pairs=len(group)-n,
            neither_collides=counts[False,False],only_a_collides=counts[True,False],
            only_b_collides=counts[False,True],both_collide=counts[True,True],
            collision_rate_a=sum(r['nondro_collision'] for r in good)/n if n else None,
            collision_rate_b=sum(r['dro_collision'] for r in good)/n if n else None,
            safe_completion_a=sum(r['nondro_safe_completion'] for r in good)/n if n else None,
            safe_completion_b=sum(r['dro_safe_completion'] for r in good)/n if n else None,
            collision_difference_b_minus_a=statistics.mean(delta) if delta else None,
            paired_bootstrap95_low=low,paired_bootstrap95_high=high,
            exact_mcnemar_p=exact_discordance_p(counts[True,False],counts[False,True]) if n else None))
    analysis.write_csv(output/'primary_summary.csv',primary)
    diagnostics=[]
    vertices=[];concentration=[]
    by_name={c['case']:c for c in cases}
    for trial in sorted(results.values(),key=lambda r:(r['case'],r['seed'])):
        if trial['status']!='OK':continue
        case=by_name[trial['case']]
        bundle=analysis.trial_root(output,case,trial['seed'])/'repeat_0'
        if not (bundle/'attempts.csv').exists():continue
        vv,cc=concentration_rows(bundle)
        identity=dict(case=case['case'],pair=case['pair'],profile=case['profile'],
                      controller=case['solver_style'],seed=trial['seed'],scenario_budget=case.get('scenario_budget','auto'))
        vertices.extend(dict(**identity,**r) for r in vv)
        concentration.extend(dict(**identity,**r) for r in cc)
        attempts=analysis.rows(bundle/'attempts.csv'); mechanism=analysis.rows(bundle/'mode_mechanism.csv')
        decisions=analysis.rows(bundle/'decisions.csv')
        first=[a for a in attempts if a['attempt']=='0']
        profile=next(p for p in settings['mismatch_profiles'] if p['name']==case['profile'])
        modes=settings['mode_catalog'][:case['modes_per_class']]
        focus=settings.get('focus_mode') or modes[profile.get('boosted_mode',-1)]
        mm=[m for m in mechanism if m['attempt']=='0' and m['mode']==focus]
        actual=trial['repeats'][0]['metrics']
        resolved=flat_yaml(bundle/'resolved_config.yaml')
        diagnostics.append(dict(case=case['case'],pair=case['pair'],profile=case['profile'],
            controller=case['solver_style'],arm=ARM_LABELS[case['solver_style']],
            environment=case['environment'],obstacles=case['obstacles'],classes=case['classes'],
            scenario_budget=case.get('scenario_budget','auto'),
            seed=trial['seed'],focus_mode=focus,
            safe_horizon_enabled=resolved['safe_horizon_enabled'],
            automatic_sample_sizing=resolved['automatically_compute_sample_size'],
            certification_status='not_requested' if not resolved['safe_horizon_enabled'] else 'see_decision_evidence',
            certificate_requested_decisions=sum(int(d['certificate_requested']) for d in decisions),
            certified_decisions=sum(int(d['certified']) for d in decisions),
            collision=actual['collision'],safe_completion=int(not actual['collision'] and actual['completed_path']),
            no_admissible_control=int(actual['termination_reason']=='no_admissible_control'),
            minimum_safety_margin_m=actual['max_conservatism_m'],executed_steps=actual['executed_steps'],
            decisions=len(decisions),first_attempt_inadmissible=sum(a['success']=='0' for a in first),
            final_inadmissible=sum(d['success']=='0' for d in decisions),
            fallback_invocations=len(attempts)-len(first),failed_outer_attempts=sum(a['success']=='0' for a in attempts),
            total_scenario_draws=sum(int(a['scenario_count']) for a in attempts),
            outer_attempts=len(attempts),qp_calls=sum(int(a['qp_calls']) for a in attempts),
            cycle_total_ms=sum(float(d['solve_ms']) for d in decisions),
            focus_checks=len(mm),focus_represented=sum(int(m['sampled_count'])>0 for m in mm),
            focus_realized=sum(m['true_mode']==focus for m in mm),
            mean_nominal_probability=statistics.mean(float(m['nominal_probability']) for m in mm),
            mean_sampling_probability=statistics.mean(float(m['sampling_probability']) for m in mm),
            mean_focus_scenarios=statistics.mean(int(m['sampled_count']) for m in mm),
            nominal_probability_sum=sum(float(m['nominal_probability']) for m in mm),
            sampling_probability_sum=sum(float(m['sampling_probability']) for m in mm)))
    analysis.write_csv(output/'mechanism_per_seed.csv',diagnostics)
    aggregate=[]
    for pair,method in sorted({(d['pair'],d['controller']) for d in diagnostics}):
        rr=[d for d in diagnostics if d['pair']==pair and d['controller']==method]
        checks=sum(d['focus_checks'] for d in rr);cycles=sum(d['decisions'] for d in rr)
        aggregate.append(dict(pair=pair,profile=rr[0]['profile'],controller=method,arm=ARM_LABELS[method],
            safe_horizon_enabled=rr[0]['safe_horizon_enabled'],automatic_sample_sizing=rr[0]['automatic_sample_sizing'],
            certification_status=rr[0]['certification_status'],
            certificate_requested_decisions=sum(d['certificate_requested_decisions'] for d in rr),
            certified_decisions=sum(d['certified_decisions'] for d in rr),
            environment=rr[0]['environment'],obstacles=rr[0]['obstacles'],classes=rr[0]['classes'],
            scenario_budget=rr[0]['scenario_budget'],expected_rollouts=len(settings['seeds']),
            measured_rollouts=len(rr),missing_or_error_rollouts=len(settings['seeds'])-len(rr),
            collision_rate=sum(d['collision'] for d in rr)/len(rr),
            safe_completion_rate=sum(d['safe_completion'] for d in rr)/len(rr),
            no_admissible_control_rate=sum(d['no_admissible_control'] for d in rr)/len(rr),
            first_attempt_inadmissibility=sum(d['first_attempt_inadmissible'] for d in rr)/cycles,
            first_attempt_admissibility=1-sum(d['first_attempt_inadmissible'] for d in rr)/cycles,
            final_inadmissibility=sum(d['final_inadmissible'] for d in rr)/cycles,
            final_admissibility=1-sum(d['final_inadmissible'] for d in rr)/cycles,
            fallback_rate=sum(d['fallback_invocations'] for d in rr)/cycles,
            mean_draws_per_cycle=sum(d['total_scenario_draws'] for d in rr)/cycles,
            mean_qp_calls_per_cycle=sum(d['qp_calls'] for d in rr)/cycles,
            minimum_safety_margin_m=min(d['minimum_safety_margin_m'] for d in rr),
            mean_rollout_minimum_safety_margin_m=statistics.mean(d['minimum_safety_margin_m'] for d in rr),
            mean_cycle_ms=sum(d['cycle_total_ms'] for d in rr)/cycles,
            focus_inclusion_rate=sum(d['focus_represented'] for d in rr)/checks,
            focus_realized_frequency=sum(d['focus_realized'] for d in rr)/checks,
            mean_nominal_probability=sum(d['nominal_probability_sum'] for d in rr)/checks,
            mean_sampling_probability=sum(d['sampling_probability_sum'] for d in rr)/checks))
    analysis.write_csv(output/'mechanism_summary.csv',aggregate)
    analysis.write_csv(output/'vertex_reachability.csv',vertices)
    analysis.write_csv(output/'concentration_per_solve.csv',concentration)
    concentration_groups=defaultdict(list)
    for r in concentration:
        concentration_groups[r['pair'],r['controller']].append(r)
    summaries=[]
    for (pair,method),rr in sorted(concentration_groups.items()):
        summaries.append(dict(pair=pair,controller=method,wdro_obstacle_solves=len(rr),
            q_gt_0_9_fraction=statistics.mean(r['q_gt_0_9'] for r in rr),
            q_near_one_fraction=statistics.mean(r['q_near_one'] for r in rr),
            q_exactly_one_fraction=statistics.mean(r['q_exactly_one'] for r in rr),
            any_vertex_reachable_fraction=statistics.mean(r['any_vertex_reachable'] for r in rr),
            min_max_q=min(r['max_q'] for r in rr),median_max_q=statistics.median(r['max_q'] for r in rr),
            max_max_q=max(r['max_q'] for r in rr),
            min_observation_count=min(r['radius_observation_count'] for r in rr),
            max_observation_count=max(r['radius_observation_count'] for r in rr),
            min_rho=min(r['rho'] for r in rr),max_rho=max(r['rho'] for r in rr)))
    analysis.write_csv(output/'concentration_summary.csv',summaries)
    analysis.dump_json(output/'evidence_notes.json',dict(
        population='All predetermined trials are retained; primary_summary uses complete pairs per comparison.',
        mechanism_population='Each controller OK trial, stratified by fixed setup/profile/budget; these marginals can have different denominators. Use per-seed rows for matched effects.',
        arms=ARM_LABELS,
        concentration='Per WDRO obstacle/attempt, repeat zero only. d_j=sum_i p_i D_ij; reachable uses rho>=d_j without tolerance. Near-one means max(q)>=1-1e-9 (diagnostic only). Zero distances yield blank ratios and explicit flags. Raw per-solve g/rho/q records support stratification; no monotonic contraction or non-collapse guarantee is asserted. Entropic allocation may remain interior even when a vertex is reachable.',
        claims='Frozen coverage: mechanism only. A/C: same-S shift robustness. B/C: extra nominal sampling. D/E: retry architecture attribution. All pair comparisons retained.',
        repeatability='Same-seed numerical traces, decisions, metrics, attempts, and p/q/r/rho/counts must match exactly. Resume rechecks saved signatures. Timing is excluded; repeats are not independent seeds.',
        warning='A focus mode is not proven dangerous by its name. Inspect logged risk and geometry. Histories adapt to observed plant modes.',
        hypothesis_tests='Exact two-sided McNemar per fixed setup over seeds; not pooled over dependent setups. Exploratory, unadjusted for multiple comparisons.',
        intervals='Descriptive 2000-resample paired seed bootstrap per fixed setup, seed 1729. Few seeds/zero events give unreliable or degenerate intervals, not zero-risk guarantees.',
        meaning='Collision-free is not successful completion; no-admissible-control and step-limit outcomes remain separate.',
        budget='Extra nominal draws 2S in one attempt; two-attempt policies draw at most 2S. Actual counts/latency are reported; CPU time is not equalized.'))
