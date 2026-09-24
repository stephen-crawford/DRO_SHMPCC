#!/usr/bin/env python3
import copy
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from compute_mode_collision_bounds import compute, support_audit
from summarize_mode_bounds import summarize


class ModeBoundsTests(unittest.TestCase):
    def fixture(self):
        return dict(horizon=1, trajectory=[[0,0,0,0,0], [4,0,0,0,0]],
                    disc_centers=[[[0,0]], [[4,0]]], beta_cp=.05, collision_radius=1.,
                    bundle_sampling=False, markov_jump_system=False,
                    obstacles=[dict(obstacle=7, rho=.04, modes=[
                        dict(mode=name, count=count, p_hat=p, q_star=q, held_affine_gaussian=True,
                             predictions=[dict(mean=mean, covariance=[[0,0],[0,0]]) for k in range(2)])
                        for name,count,p,q,mean in [('away',90,.9,.8,[0,0]), ('near',10,.1,.2,[4,0])]])])

    def test_actual_final_center_and_full_pair_evidence(self):
        bounds, pairs = compute(self.fixture(), ROOT/'build-certificate/certificate_numeric_backend')
        self.assertEqual([r['b_mode'] for r in bounds], [0,1])
        self.assertEqual([r['mass_shift'] for r in bounds], ['decreased','increased'])
        self.assertEqual(len(pairs), 2)
        self.assertEqual([r['k'] for r in pairs], [1,1])
        self.assertTrue(all(0 <= r['L'] <= r['U'] <= 1 for r in bounds))
        summary = summarize([dict(case='fixture', **r) for r in bounds])
        self.assertEqual(summary[0]['fraction_b_lt_one'], .5)
        self.assertEqual(summary[0]['median'], .5)

    def test_support_removal_is_union_not_addition(self):
        s = dict(support_active=[2], removed_ids=[3,17], support_union=[2,3,17],
                 support_count_used_for_certificate=3, support_evaluated=True)
        a = support_audit(s)
        self.assertTrue(a['removed_in_union'] and a['active_in_union'] and a['count_matches'])
        s['support_count_used_for_certificate'] = 5
        self.assertFalse(support_audit(s)['count_matches'])
        s['support_union'] = [2]
        self.assertFalse(support_audit(s)['removed_in_union'])

    def test_no_silent_model_or_horizon_substitution(self):
        for key in ('bundle_sampling','markov_jump_system'):
            s = self.fixture(); s[key] = True
            with self.assertRaises(ValueError): compute(s, ROOT/'build-certificate/certificate_numeric_backend')
        s = self.fixture(); s['disc_centers'].pop()
        with self.assertRaises(ValueError): compute(s, ROOT/'build-certificate/certificate_numeric_backend')
        s = self.fixture(); s['obstacles'].append(copy.deepcopy(s['obstacles'][0]))
        with self.assertRaises(ValueError): compute(s, ROOT/'build-certificate/certificate_numeric_backend')

if __name__ == '__main__': unittest.main()
