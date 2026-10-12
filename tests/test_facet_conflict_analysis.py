"""Analytical fixtures for the offline infeasibility/removal diagnostic."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_facet_conflicts import conflict,removals,horizon_removals


def row(sid,x,y,b,k=1):
    return dict(id=sid,a_x=x,a_y=y,b=b,k=k,disc=0)


class ConflictAnalysisTests(unittest.TestCase):
    def test_feasible_and_opposing_pair(self):
        rows=[row(0,1.,0.,1.),row(1,-1.,0.,0.)]
        self.assertIsNone(conflict(rows[:1]))
        self.assertEqual(conflict(rows)['size'],2)
        self.assertEqual(removals(rows)['minimum'],1)

    def test_three_facets(self):
        rows=[row(0,1.,0.,1.),row(1,0.,1.,1.),row(2,-1.,-1.,-1.)]
        self.assertEqual(conflict(rows)['size'],3)
        self.assertEqual(removals(rows)['minimum'],1)

    def test_joint_scenario_deletions(self):
        rows=[row(0,1.,0.,1.),row(1,-1.,0.,0.),row(2,1.,0.,1.),row(3,-1.,0.,0.)]
        self.assertEqual(removals(rows)['minimum'],2)
        grouped=[row(0,1.,0.,1.),row(0,1.,0.,2.),row(1,-1.,0.,0.)]
        self.assertEqual(removals(grouped)['minimum'],1)

    def test_horizon_and_budget_lower_bound(self):
        rows=[]
        for k in range(1,4):
            rows.extend([row(2*k,1.,0.,1.,k),row(2*k+1,-1.,0.,0.,k)])
        self.assertEqual(horizon_removals(rows[:4])['minimum'],2)
        self.assertEqual(horizon_removals(rows)['minimum'],'>=3')


if __name__=='__main__':unittest.main()
