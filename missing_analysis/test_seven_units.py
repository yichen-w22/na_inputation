"""Reuse the seven core protocol tests and test multi-unit generalization."""
import sys
import unittest
import numpy as np
import pandas as pd
import seven_units as m

sys.modules.setdefault('daily_modes',m.core)
from jqrd_gt1.test_daily_modes import ProtocolTests


class MultiUnitTests(unittest.TestCase):
    def test_selected_sensor_scope_and_order(self):
        for unit in m.UNITS:
            points,_=m.points(unit)
            self.assertEqual(len(points),30 if unit.startswith('gard') else 23)
            self.assertEqual(points[0]['semantic'],'fuel_volume_flow')
            self.assertEqual(sum(p['semantic']=='CH4' for p in points),1)
            self.assertFalse(any(p['semantic'] in m.core.EXCLUDE for p in points))

    def test_status_and_boundary_rules(self):
        origin=m.core.time_ns('2025-01-01')
        source=dict(sensor='test',name='test',semantic='test',first_ns=origin+m.core.DAY,
                    last_ns=origin+4*m.core.DAY)
        counts=np.zeros((6,52),dtype=np.int64)
        counts[:,21]=[0,100,100,1,0,0]
        z=dict(counts=counts,observations=np.array([0,101,100,1,1,0]),longest=np.zeros(6),
               long_gap_seconds=np.zeros(6),unfinished=np.array([0,0,0,0,1,0]))
        f=m.make_daily_frame(source,0,origin,6,z)
        self.assertEqual(f.status.tolist(),['outside_coverage','boundary_assignable','eligible',
                          'insufficient_intervals','insufficient_intervals','outside_coverage'])
        np.testing.assert_array_equal(f.train_eligible,[False,False,True,False,False,False])
        np.testing.assert_array_equal(f.assignable,[False,True,True,False,False,False])

    def test_transitions_scale_with_sensor_count_and_do_not_bridge(self):
        for sensor_count in [2,23,30]:
            f=pd.DataFrame({'sensor':np.repeat(np.arange(sensor_count),4)})
            labels=np.tile([0,0,-1,1],sensor_count)
            all_t,stable_t=m.matrices(labels,np.tile([True,False,False,True],sensor_count),f,2)
            np.testing.assert_array_equal(all_t,[[sensor_count,0],[0,0]])
            self.assertEqual(stable_t.sum(),0)


if __name__=='__main__':
    unittest.main(verbosity=2)
