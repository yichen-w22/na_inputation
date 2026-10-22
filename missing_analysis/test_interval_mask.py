"""Boundary and end-to-end checks for interval-only learning and masking."""
import unittest
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from interval_mask import (Settings, fit_distribution, generate_mask, high_frequency, learn,
                           plot_top3, transition_matrix, valid_times, window_features)


def settings(**overrides):
    args = dict(high_window_seconds=100, high_p95_seconds=1.1, high_cv=.05,
                windows_seconds=(100, 200), step_seconds=100,
                classes=(1,), min_intervals=5, min_class_windows=2,
                repeats=2, fit_sample_limit=1000)
    args.update(overrides)
    return Settings(**args)


class IntervalMaskTests(unittest.TestCase):
    def test_invalid_records_and_order(self):
        times = pd.date_range('2020-01-01', periods=4, freq='s')
        ns, ids = valid_times(times, [1, np.nan, 2, 3])
        self.assertEqual(ids.tolist(), [0, 2, 3])
        self.assertEqual(np.diff(ns).tolist(), [2000000000, 1000000000])
        with self.assertRaises(ValueError):
            valid_times([times[1], times[0]])
        with self.assertRaises(ValueError):
            valid_times([times[0], times[0]])
        ns, _ = valid_times(['2020-01-01 00:00:00.100', '2020-01-01 00:00:01'])
        self.assertEqual(int(np.diff(ns)[0]), 900000000)

    def test_high_frequency_and_crossing_interval(self):
        t = np.r_[np.arange(101), 150., 250.]
        excluded, blocks = high_frequency(t, np.diff(t), settings())
        self.assertEqual(blocks, [[0., 100.]])
        self.assertTrue(excluded[:100].all())
        self.assertFalse(excluded[100:].any())
        # An otherwise dense burst that does not span a block is not high quality.
        t = np.r_[0., np.arange(40., 61.), 200.]
        excluded, blocks = high_frequency(t, np.diff(t), settings())
        self.assertEqual(blocks, [])

    def test_window_right_endpoint_and_no_zero_empty(self):
        t = np.array([0., 1., 201., 202., 203., 204., 205., 206.])
        cfg = settings(windows_seconds=(100,), min_intervals=2, step_seconds=100)
        f, ids, ends = window_features(t, np.diff(t), np.zeros(7, bool), 100, cfg)
        self.assertEqual(len(f), 0)  # long gap is one interval, not 100 zero intervals
        cfg = settings(windows_seconds=(100,), min_intervals=2, step_seconds=5)
        f, ids, ends = window_features(t, np.diff(t), np.zeros(7, bool), 100, cfg)
        self.assertIn(1, ids[-1])  # 1->201 belongs by its later endpoint

    def test_no_stable_result_does_not_generate(self):
        times = pd.date_range('2020-01-01', periods=1001, freq='s')
        model = learn(times, np.ones(len(times)), settings())
        self.assertEqual(model['status'], 'no_stable_model')
        with self.assertRaises(ValueError):
            generate_mask(model, times, seed=1, tolerance_seconds=.5)

    def test_constant_sparse_end_to_end_and_mapping(self):
        times = pd.date_range('2020-01-01', periods=1001, freq='10s')
        model = learn(times, np.ones(len(times)), settings())
        self.assertEqual(model['status'], 'stable')
        self.assertEqual(model['distributions'][0]['family'], 'constant')
        target = pd.date_range('2021-01-01', periods=101, freq='s')
        a, info = generate_mask(model, target, seed=3, tolerance_seconds=0)
        b, _ = generate_mask(model, target, seed=3, tolerance_seconds=0)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(np.flatnonzero(a).tolist(), list(range(10, 101, 10)))
        self.assertEqual(info['retained'], 10)
        with self.assertRaises(RuntimeError):
            generate_mask(model, target, seed=3, tolerance_seconds=0, max_draws=1)

    def test_matching_tolerance_no_invented_records(self):
        model = dict(status='stable', mode_runs=[dict(label=0, duration_seconds=100)],
                     distributions=[dict(family='constant', parameters=[2.5])])
        target = pd.date_range('2021-01-01', periods=11, freq='s')
        mask, _ = generate_mask(model, target, seed=1, tolerance_seconds=0)
        self.assertEqual(np.flatnonzero(mask).tolist(), [5, 10])
        mask, _ = generate_mask(model, target, seed=1, tolerance_seconds=.5)
        self.assertEqual(np.flatnonzero(mask).tolist(), [2, 5, 7, 10])

    def test_distribution_and_multiclass_search(self):
        rng = np.random.default_rng(42)
        fitted = fit_distribution(rng.lognormal(2., .4, 3000))
        self.assertEqual(fitted['family'], 'lognorm')
        self.assertEqual(len(fitted['candidates']), 4)
        # Two repeated sparse regimes, with no high-frequency observations.
        intervals = np.tile(np.r_[np.full(1000, 5.), np.full(1000, 20.)], 4)
        times = pd.Timestamp('2020-01-01') + pd.to_timedelta(np.r_[0., intervals.cumsum()], unit='s')
        model = learn(times, np.ones(len(times)), settings(classes=(1, 2, 3),
                      windows_seconds=(300, 600, 900), step_seconds=300))
        self.assertEqual(model['status'], 'stable')
        self.assertGreater(model['selected']['k'], 1)
        self.assertEqual(len(model['top3']), 3)
        self.assertEqual([s['selected']['silhouette'] for s in model['top3']],
                         sorted([s['selected']['silhouette'] for s in model['top3']], reverse=True))
        # Overlapping windows cannot double-count intervals for distribution fitting.
        n = sum(d['intervals'] for d in model['distributions'])
        self.assertLessEqual(n, len(intervals))
        self.assertTrue(all(np.isfinite(d['parameters']).all() for d in model['distributions']))
        target = pd.date_range('2021-01-01', periods=2001, freq='s')
        mask, _ = generate_mask(model, target, seed=12, tolerance_seconds=.5)
        self.assertGreater(mask.sum(), 0)
        self.assertLess(mask.sum(), len(target))
        with tempfile.TemporaryDirectory() as folder:
            names = plot_top3(model, folder, 'Synthetic Top3')
            self.assertEqual(len(list(Path(folder).iterdir())), 3)
            for name in names:
                root = ET.parse(Path(folder)/name).getroot()
                self.assertTrue(root.tag.endswith('svg'))

    def test_transition_gaps_and_unobserved_rows(self):
        counts, p = transition_matrix([10, 20, 40, 50], [0, 1, 0, 0], 10, 3)
        np.testing.assert_array_equal(counts, [[1, 1, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(p[0], [.5, .5, 0])
        self.assertTrue(np.isnan(p[1:]).all())


if __name__ == '__main__':
    unittest.main()
