import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_control_response import seed_events
from analyze_control_response_expansion import extend_event, summarize_cohort, PRIMARY_FIELDS
from test_control_response import decision, mode

class ExpansionTests(unittest.TestCase):
    def event(self, extra=1, response=2, risk='.5'):
        n={t:decision(t,.5) for t in range(3)}
        d={t:decision(t,.5 if t<response else -.5) for t in range(3)}
        modes={'nominal':{t:mode(t,1,q='.02') for t in n},
               'wdro':{t:mode(t,1+int(t>=extra)) for t in n}}
        for r in modes['wdro'].values():r['risk_score']=risk
        return extend_event(seed_events(1,'neither_collision',n,d,modes))

    def test_extra_not_first_sample_and_no_forced_alignment(self):
        e=self.event()
        self.assertEqual(e['wdro_first_across_sample'],0)
        self.assertEqual(e['first_extra_wdro_across_sample'],1)
        self.assertEqual(e['first_extra_equals_first_response'],0)
        self.assertEqual(e['first_extra_before_first_response'],1)
        self.assertEqual(e['positive_chain_at_first_response'],1)

    def test_risk_required_for_chain(self):
        self.assertEqual(self.event(risk='0')['positive_chain_at_first_response'],0)

    def test_no_response_preserves_missing_denominator(self):
        e=self.event(response=4)
        self.assertIsNone(e['positive_chain_at_first_response'])
        summary=next(r for r in summarize_cohort([e,self.event()]) if r['group']=='all_seeds')
        self.assertEqual(summary['seeds'],2)
        self.assertEqual(summary['positive_chain_at_first_response_eligible'],1)

    def test_availability_response_has_chain_without_executed_brake(self):
        n={0:decision(0,.5),1:decision(1,.5)}
        d={0:decision(0,.5),1:decision(1,None,'0')}
        m={'nominal':{t:mode(t,0) for t in n},'wdro':{t:mode(t,t) for t in n}}
        e=extend_event(seed_events(1,'test',n,d,m))
        self.assertEqual(e['first_decision_response_kind'],'availability')
        self.assertEqual(e['positive_chain_at_first_response'],1)
        self.assertIsNone(e['wdro_first_brake'])
        self.assertIsNone(e['first_extra_equals_first_material'])

    def test_primary_uses_extra_and_supplement_only_first_samples(self):
        names=[r[0] for r in PRIMARY_FIELDS]
        self.assertIn('first_extra_wdro_across_sample',names)
        self.assertNotIn('wdro_first_across_sample',names)
        self.assertIn('first_control_availability_divergence',names)

if __name__=='__main__':unittest.main()
