"""Protocol-critical synthetic tests; no raw data changes."""
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import daily_modes_core as m


class ProtocolTests(unittest.TestCase):
    def test_bin_boundaries(self):
        x = np.array([.001, .01, m.EDGES[1], 999., 1000., 1000.001])
        np.testing.assert_array_equal(m.bin_ids(x), [0, 1, 2, 50, 50, 51])
        counts = np.bincount(m.bin_ids(x), minlength=52)
        self.assertAlmostEqual((counts/counts.sum()).sum(), 1)

    def test_stream_day_chunk_duplicates_invalid_and_censoring(self):
        origin = m.time_ns('2025-01-01')
        # The last completed gap spans an entirely empty calendar day.
        t = ['2025-01-01 23:59:59', '2025-01-02 00:00:00',
             '2025-01-02 00:00:00', '2025-01-02 00:00:01',
             '2025-01-02 00:00:02', '2025-01-04 00:00:02']
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'synthetic.csv'
            pd.DataFrame(dict(timestamp=t, value=[1, 2, 2, 'bad', 3, 4])).to_csv(path, index=False)
            a, audit = m.scan_source(path, origin, 4, 2)
            b, other = m.scan_source(path, origin, 4, 100)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(audit, other)
            self.assertEqual(audit['positive_intervals'], 3)
            self.assertEqual(audit['duplicate_valid_timestamps'], 1)
            self.assertEqual(audit['invalid_values'], 1)
            np.testing.assert_array_equal(a['counts'].sum(axis=1), [1, 2, 0, 0])
            np.testing.assert_array_equal(a['observations'], [1, 3, 0, 1])
            np.testing.assert_allclose(a['long_gap_seconds'], [0, 86398, 86400, 2])
            np.testing.assert_array_equal(a['unfinished'], [0, 0, 0, 1])
            # Descending invalid records must also be detected, including across chunks.
            pd.DataFrame(dict(timestamp=t[:2]+[t[0]], value=[1, 1, 'bad'])).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                m.scan_source(path, origin, 4, 2)

    def test_timezone_and_subsecond(self):
        t = m.timestamps(pd.Series(['2025-01-01T16:00:00.010+00:00']))
        self.assertEqual(t[0], m.time_ns('2025-01-02 00:00:00.010'))

    def test_hellinger(self):
        p = np.array([[1., 0], [0, 1.], [.5, .5]])
        d = m.hellinger(p)
        np.testing.assert_allclose(d, d.T)
        np.testing.assert_array_equal(np.diag(d), 0)
        self.assertAlmostEqual(d[0, 1], 1)

    def test_true_medoids_exhaustive_optimum_and_seed(self):
        p = np.array([[1., 0, 0], [.9, .1, 0], [0, 1., 0], [0, .9, .1], [0, 0, 1.]])
        d = m.hellinger(p)
        optimum = min(d[:, medoids].min(axis=1).sum() for medoids in itertools.combinations(range(len(p)), 2))
        fits = [m.fit_once(d, 2, m.seeded(99, i)) for i in range(10)]
        fit = min(fits, key=lambda f: f['loss'])
        self.assertAlmostEqual(fit['loss'], optimum)
        self.assertTrue(set(fit['medoids']).issubset(range(len(p))))
        repeat = m.fit_once(d, 2, fit['seed'])
        np.testing.assert_array_equal(fit['labels'], repeat['labels'])

    def test_date_split_alignment_and_ari(self):
        keep, blocks = m.date_split(334, 0)
        np.testing.assert_array_equal(keep, m.date_split(334, 0)[0])
        self.assertEqual(len(blocks), round(.8*48))
        # Same calendar mask applied to distinct sensors; no train/check date overlap.
        dates = np.tile(np.arange(334), 3)
        mask = keep[dates]
        self.assertFalse(set(dates[mask]) & set(dates[~mask]))
        for i in range(3):
            np.testing.assert_array_equal(mask[i*334:(i+1)*334], keep)
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([2, 2, 0, 0, 1, 1])
        self.assertEqual(adjusted_rand_score(a, b), 1)
        mapping, _ = m.match_labels(a, b, 3)
        np.testing.assert_array_equal(mapping[b], a)
        # The mapping remains based only on retained labels, regardless of check labels.
        np.testing.assert_array_equal(mapping[[2, 0, 1]], [0, 1, 2])

    def test_temporal_no_bridging_and_censoring(self):
        labels = np.array([0, 0, -1, 0, 1, 1, 0])
        t = m.transitions(labels, 2)
        np.testing.assert_array_equal(t, [[1, 1], [1, 1]])
        r = m.visible_runs(labels)
        self.assertEqual([x['days'] for x in r], [2, 1, 2, 1])
        self.assertTrue(r[0]['left_censored'] and r[0]['right_censored'])
        self.assertTrue(r[1]['left_censored'])
        self.assertFalse(r[1]['right_censored'])
        self.assertFalse(r[2]['left_censored'] or r[2]['right_censored'])
        self.assertTrue(r[3]['right_censored'])
        r = m.visible_runs(np.array([0, 1, 1, 2]), np.array([False, True, True, False]))
        self.assertTrue(r[1]['left_censored'] and r[1]['right_censored'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
