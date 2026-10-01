import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_control_response import divergence,seed_events,flatten,category


def decision(t,a,success='1'):
    return dict(step=str(t),success=success,acceleration=str(a) if a is not None else '',omega='0' if a is not None else '',ego_speed=str(2+t*.1),actual_clearance=str(3-t*.1))
def mode(t,n,p='.02',q='.04'):
    return dict(step=str(t),sampled_count=str(n),nominal_probability=p,sampling_probability=q,risk_score='.5',rho='.01',true_mode='continue')

class ControlResponseTests(unittest.TestCase):
    def test_material_threshold_and_refusal_separate(self):
        n={0:decision(0,.5),1:decision(1,.5),2:decision(2,.5)}
        d={0:decision(0,.49),1:decision(1,.2),2:decision(2,None,'0')}
        result=divergence(n,d)
        self.assertEqual(result['first_material_control_divergence'],1)
        self.assertEqual(result['first_materially_lower_wdro_acceleration'],1)
        self.assertEqual(result['first_control_availability_divergence'],2)
    def test_sample_before_same_solve_control_and_strict_brake(self):
        n={0:decision(0,.5),1:decision(1,-.1),2:decision(2,-.2)}
        d={0:decision(0,.5),1:decision(1,-.4),2:decision(2,-.3)}
        m={'nominal':{t:mode(t,0,q='.02') for t in n},'wdro':{t:mode(t,int(t>=1)) for t in n}}
        e=seed_events(1,'test',n,d,m)
        self.assertEqual(e['nominal_first_brake'],2)
        self.assertEqual(e['wdro_first_brake'],1)
        self.assertEqual(e['braking_advance_steps'],1)
        self.assertEqual(e['risk_shift_extra_sample_before_lower_control'],1)
        self.assertEqual(e['extra_sample_at_material_control_divergence'],1)
    def test_absent_braking_is_not_zero_and_refusal_not_braking(self):
        n={0:decision(0,.5),1:decision(1,.5)}
        d={0:decision(0,.5),1:decision(1,None,'0')}
        m={'nominal':{t:mode(t,0) for t in n},'wdro':{t:mode(t,1) for t in n}}
        e=seed_events(1,'test',n,d,m)
        self.assertIsNone(e['wdro_first_brake'])
        self.assertIsNone(e['braking_advance_steps'])
        self.assertEqual(e['availability_only_divergence'],1)
        self.assertEqual(e['first_decision_response_kind'],'availability')
    def test_precontrol_state_and_terminal_missingness(self):
        n={0:decision(0,.5),1:decision(1,None,'0')};d={0:decision(0,.3)}
        m={'nominal':{t:mode(t,0) for t in n},'wdro':{t:mode(t,1) for t in d}}
        rows=flatten(1,'test',n,d,m)
        self.assertEqual(rows[0]['nominal_v'],'2.0')
        self.assertEqual(rows[0]['nominal_next_v'],'2.1')
        self.assertNotIn('wdro_next_v',rows[0])
        self.assertEqual(rows[1]['matched'],0)
    def test_outcome_subgroup_does_not_treat_all_noncollisions_as_refusal(self):
        n=dict(collision='1',termination='collision');d=dict(collision='0',termination='completed')
        self.assertEqual(category(n,d),'other')
        d['termination']='no_admissible_control'
        self.assertEqual(category(n,d),'nominal_collision_wdro_refusal')
if __name__=='__main__':unittest.main()
