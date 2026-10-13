"""Shared daily interval modes. Run `python missing_analysis/daily_modes.py --help`.

Protocol: Shanghai calendar; completed intervals belong to their left endpoint;
50 log bins plus tails; equal sensor-day weights; no operating-state filter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import kmedoids
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, silhouette_samples

ROOT = Path(__file__).resolve().parents[1]
EXECUTED_SOURCE = Path(__file__).read_bytes()
SECOND = 1_000_000_000
DAY = 86400 * SECOND
EDGES = np.geomspace(.01, 1000., 51)
EXCLUDE = {'H2', 'N2', 'CO2', 'CO', 'C2H6', 'C3H8', 'iC4H10', 'nC4H10',
           'iC5H12', 'nC5H12', 'oxygen_channel', 'C6plus'}
CONFIG = dict(protocol='jqrd_gt1_daily_v1', timezone='Asia/Shanghai', unit='jqrd_gt1',
              seed=20261013, bins=50, lower_seconds=.01, upper_seconds=1000.,
              min_intervals=50, k_values=[2, 3, 4, 5, 6], initializations=10,
              perturbations=20, perturbation_initializations=3, block_days=7,
              retain_fraction=.8, permutations=500, chunk_size=350000,
              state_filter='all', boundary_training=False, n_cpu=1,
              thresholds=dict(init_ari=.90, block_ari=.80, block_ari_p10=.60,
                              class_jaccard=.60, silhouette=0., silhouette_tolerance=.02),
              interval_assignment='left_endpoint', probability_smoothing=False)


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv(path, table):
    pd.DataFrame(table).to_csv(path, index=False, encoding='utf-8-sig')


def metadata(path):
    s = path.stat()
    return dict(path=str(path.resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(run, stage):
    versions = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'kmedoids', 'matplotlib', 'openpyxl']}
    assert versions['kmedoids'] == '0.5.5', 'Protocol requires kmedoids==0.5.5'
    codehash = hashlib.sha256(EXECUTED_SOURCE).hexdigest()
    target = run / 'code' / f'daily_modes_{codehash[:12]}.py'
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        target.write_bytes(EXECUTED_SOURCE)
    save_json(run / f'provenance_{stage}.json', dict(code_sha256=codehash, versions=versions,
              python=sys.version, platform=platform.platform(), command=sys.argv,
              utc_time=pd.Timestamp.now(tz='UTC').isoformat(), config_sha256=digest(run/'config.json')))
    status = read_json(run/'status.json') if (run/'status.json').exists() else {}
    status[stage] = dict(state='complete', utc_time=pd.Timestamp.now(tz='UTC').isoformat())
    save_json(run/'status.json', status)


def time_ns(x):
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_convert('Asia/Shanghai').tz_localize(None)
    return t.value


def timestamps(series):
    ts = pd.to_datetime(series, format='mixed', errors='raise')
    if ts.isna().any():
        raise ValueError('Empty timestamp')
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        ts = ts.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    return ts.dt.as_unit('ns').astype('int64').to_numpy()


def source_extent(path):
    import csv as csvlib
    first = pd.read_csv(path, nrows=1)
    if first.empty:
        raise ValueError(f'Empty source: {path}')
    with path.open('rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell()-65536))
        last = next(x for x in reversed(f.read().decode('utf-8-sig').splitlines()) if x.strip())
    last = next(csvlib.reader([last]))
    return time_ns(first.iloc[0]['timestamp']), time_ns(last[first.columns.tolist().index('timestamp')])


def selected_points():
    from openpyxl import load_workbook
    files = list((ROOT/'raw_data'/'jqrd_gt1').glob('*选取点表.xlsx'))
    assert len(files) == 1
    wb = load_workbook(files[0], read_only=True, data_only=True)
    rows = [dict(sensor=r[0], name=r[1], semantic=r[2])
            for r in wb.worksheets[0].iter_rows(min_row=2, values_only=True)
            if r[0] and r[2] not in EXCLUDE]
    wb.close()
    assert len(rows) == 23 and sum(r['semantic'] == 'CH4' for r in rows) == 1
    return rows, files[0]


def bin_ids(seconds):
    b = np.searchsorted(EDGES, seconds, side='right')
    b[seconds == EDGES[-1]] = 50
    return b


def read_chunks(path, size):
    with pd.read_csv(path, usecols=['timestamp', 'value'], chunksize=size) as reader:
        yield from reader


def scan_source(path, origin, n_days, chunk_size):
    """Count every completed positive valid-to-valid interval exactly once."""
    counts = np.zeros((n_days, 52), dtype=np.int64)
    observations = np.zeros(n_days, dtype=np.int64)
    longest = np.zeros(n_days)
    overlap = np.zeros(n_days)
    full_days = np.zeros(n_days+1, dtype=np.int64)
    raw_previous = valid_previous = None
    audit = dict(rows=0, invalid_values=0, duplicate_valid_timestamps=0, positive_intervals=0)
    first = last = None
    for c in read_chunks(path, chunk_size):
        t = timestamps(c['timestamp'])
        v = pd.to_numeric(c['value'], errors='coerce').to_numpy(dtype=float)
        if np.any(np.diff(t) < 0) or (raw_previous is not None and t[0] < raw_previous):
            raise ValueError(f'Descending timestamp: {path}, chunk beginning row {audit["rows"]+1}')
        first = int(t[0]) if first is None else first
        last = raw_previous = int(t[-1])
        audit['rows'] += len(t)
        ok = np.isfinite(v)
        audit['invalid_values'] += int((~ok).sum())
        t = t[ok]
        np.add.at(observations, (t-origin)//DAY, 1)
        if not len(t):
            continue
        if valid_previous is not None:
            t = np.r_[valid_previous, t]
        valid_previous = int(t[-1])
        left, right = t[:-1], t[1:]
        positive = right > left
        audit['duplicate_valid_timestamps'] += int((~positive).sum())
        left, right = left[positive], right[positive]
        dt = (right-left)/SECOND
        d = (left-origin)//DAY
        counts += np.bincount(d*52 + bin_ids(dt), minlength=n_days*52).reshape(n_days, 52)
        np.maximum.at(longest, d, dt)
        audit['positive_intervals'] += len(dt)
        big = dt > 1000
        a, b, duration = left[big], right[big], dt[big]
        da, db = (a-origin)//DAY, (b-origin)//DAY
        same = da == db
        np.add.at(overlap, da[same], duration[same])
        da, db, a, b = da[~same], db[~same], a[~same], b[~same]
        np.add.at(overlap, da, (origin+(da+1)*DAY-a)/SECOND)
        np.add.at(overlap, db, (b-origin-db*DAY)/SECOND)
        np.add.at(full_days, da+1, 1)
        np.add.at(full_days, db, -1)
    overlap += np.cumsum(full_days[:-1])*86400
    assert counts.sum() == audit['positive_intervals']
    assert np.all(overlap <= 86400+1e-6)
    unfinished = np.zeros(n_days, dtype=np.int64)
    if valid_previous is not None:
        unfinished[(valid_previous-origin)//DAY] = 1
    audit.update(first_ns=first, last_ns=last, unclosed_last_valid_ns=valid_previous)
    return dict(counts=counts, observations=observations, longest=longest,
                long_gap_seconds=overlap, unfinished=unfinished), audit


def prepare(run):
    run.mkdir(parents=True, exist_ok=True)
    if (run/'config.json').exists():
        assert read_json(run/'config.json') == CONFIG, 'Use a new run ID for protocol changes'
    else:
        save_json(run/'config.json', CONFIG)
    points, workbook = selected_points()
    sources = []
    for p in points:
        path = ROOT/'raw_data'/'jqrd_gt1'/'csv'/f'{p["sensor"]}.csv'
        a, b = source_extent(path)
        sources.append(dict(**p, **metadata(path), first_ns=a, last_ns=b))
    inventory = dict(workbook=metadata(workbook), workbook_sha256=digest(workbook), sources=sources)
    if (run/'sources.json').exists():
        assert read_json(run/'sources.json') == inventory, 'Sources changed; start a new run'
    save_json(run/'sources.json', inventory)
    origin = min(s['first_ns'] for s in sources)//DAY*DAY
    end = max(s['last_ns'] for s in sources)//DAY*DAY
    n_days = int((end-origin)//DAY+1)
    dates = pd.date_range(pd.Timestamp(origin), periods=n_days, freq='D').strftime('%Y-%m-%d')
    cache = run/'cache'
    cache.mkdir(exist_ok=True)
    frames, histograms, audits = [], [], []
    for j, s in enumerate(sources):
        tag = s['sensor']
        cp, ap = cache/f'{tag}.npz', cache/f'{tag}.json'
        if cp.exists() and ap.exists():
            z, audit = dict(np.load(cp)), read_json(ap)
        else:
            print(f'prepare {j+1}/{len(sources)} {tag}', flush=True)
            z, audit = scan_source(Path(s['path']), origin, n_days, CONFIG['chunk_size'])
            assert metadata(Path(s['path'])) == {k: s[k] for k in ['path', 'bytes', 'mtime_ns']}
            np.savez_compressed(cp, **z)
            save_json(ap, audit)
        assert audit['first_ns'] == s['first_ns'] and audit['last_ns'] == s['last_ns']
        total = z['counts'].sum(axis=1)
        assert total.sum() == audit['positive_intervals']
        d = np.arange(n_days)
        first_day, last_day = (s['first_ns']-origin)//DAY, (s['last_ns']-origin)//DAY
        covered = (d >= first_day) & (d <= last_day)
        boundary = (d == first_day) | (d == last_day)
        status = np.full(n_days, 'eligible', dtype=object)
        status[total < CONFIG['min_intervals']] = 'insufficient_intervals'
        status[z['observations'] == 0] = 'no_valid_observations'
        status[~covered] = 'outside_coverage'
        status[boundary & (total >= CONFIG['min_intervals'])] = 'boundary_assignable'
        frame = pd.DataFrame(dict(sensor=tag, name=s['name'], semantic=s['semantic'],
                 sensor_order=j, date=dates, day_index=d, status=status, boundary=boundary,
                 valid_observations=z['observations'], completed_intervals=total,
                 longest_interval_seconds=z['longest'], long_gap_overlap_seconds=z['long_gap_seconds'],
                 unfinished_intervals=z['unfinished'], assignable=covered & (total >= CONFIG['min_intervals']),
                 train_eligible=covered & ~boundary & (total >= CONFIG['min_intervals'])))
        for name, pos in [('underflow_probability', 0), ('overflow_probability', 51)]:
            frame[name] = np.divide(z['counts'][:, pos], total, out=np.full(n_days, np.nan), where=total > 0)
        # Independent earlier full-source audit, when available; bin definitions differ.
        old_svg = ROOT/'figures'/'interval_scattering'/f'{tag}_interval_probability.svg'
        if old_svg.exists():
            import xml.etree.ElementTree as ET
            desc = ET.parse(old_svg).getroot().find('.//{http://purl.org/dc/elements/1.1/}description')
            old = json.loads(desc.text)['source_audit']
            assert old['positive_intervals'] == audit['positive_intervals']
            assert old['rows'] == audit['rows']
            audit['historical_audit_match'] = True
        frames.append(frame)
        histograms.append(z['counts'])
        audits.append(dict(sensor=tag, **audit))
    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, 'sample_id', np.arange(len(frame)))
    counts = np.vstack(histograms)
    total = counts.sum(axis=1)
    probability = np.divide(counts, total[:, None], out=np.full(counts.shape, np.nan), where=total[:, None] > 0)
    np.testing.assert_allclose(probability[total > 0].sum(axis=1), 1)
    np.savez_compressed(run/'daily_profiles.npz', counts=counts, probabilities=probability, edges=EDGES)
    csv(run/'daily_metadata.csv', frame)
    save_json(run/'source_audit.json', audits)
    verify_sources(run)
    provenance(run, 'prepare')
    print(f'Prepared {len(frame)} sensor-days; {frame.train_eligible.sum()} learning profiles', flush=True)


def verify_sources(run):
    inv = read_json(run/'sources.json')
    for s in inv['sources']:
        assert metadata(Path(s['path'])) == {k: s[k] for k in ['path', 'bytes', 'mtime_ns']}
    assert metadata(Path(inv['workbook']['path'])) == inv['workbook']
    assert digest(Path(inv['workbook']['path'])) == inv['workbook_sha256']


def load_data(run):
    assert read_json(run/'config.json') == CONFIG, 'Protocol mismatch'
    return pd.read_csv(run/'daily_metadata.csv'), np.load(run/'daily_profiles.npz')['probabilities']


def hellinger(p, q=None):
    q = p if q is None else q
    return cdist(np.sqrt(p), np.sqrt(q), 'euclidean')/np.sqrt(2)


def seeded(*parts):
    return int(np.random.SeedSequence([CONFIG['seed'], *parts]).generate_state(1)[0])


def fit_once(distance, k, seed):
    r = kmedoids.fasterpam(distance, k, random_state=seed, max_iter=300, n_cpu=1)
    m = np.array(r.medoids, dtype=int)
    labels = np.argmin(distance[:, m], axis=1)
    loss = float(distance[np.arange(len(distance)), m[labels]].sum(dtype=np.float64))
    assert len(np.unique(m)) == k and len(np.unique(labels)) == k, 'Empty or duplicate cluster'
    return dict(medoids=m, labels=labels, loss=loss, seed=seed, iterations=r.n_iter, swaps=r.n_swap)


def order_medoids(medoids, profiles, frame):
    def key(i):
        p = profiles[i]
        return (int(np.searchsorted(np.cumsum(p), .5)), float(p@np.arange(52)),
                frame.iloc[i]['sensor'], frame.iloc[i]['date'])
    return np.array(sorted(medoids, key=key), dtype=int)


def date_split(n_days, rep):
    blocks = np.arange((n_days+6)//7)
    rng = np.random.default_rng(seeded(20, rep))
    selected = rng.choice(blocks, int(round(.8*len(blocks))), replace=False)
    return np.isin(np.arange(n_days)//7, selected), selected


def match_labels(reference, prediction, k):
    overlap = np.zeros((k, k), dtype=int)
    np.add.at(overlap, (reference, prediction), 1)
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.full(k, -1, dtype=int)
    mapping[cols] = rows
    return mapping, overlap


def quantile(x, q):
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x[np.isfinite(x)], q)) if np.any(np.isfinite(x)) else np.nan


def evaluate(run):
    frame, p = load_data(run)
    train = np.flatnonzero(frame.train_eligible)
    assign = np.flatnonzero(frame.assignable)
    n_days = int(frame.day_index.max()+1)
    distance_path = run/'hellinger_train.npy'
    if not distance_path.exists():
        distance = np.lib.format.open_memmap(distance_path, mode='w+', dtype='float64', shape=(len(train), len(train)))
        for start in range(0, len(train), 512):
            distance[start:start+512] = hellinger(p[train[start:start+512]], p[train])
        np.fill_diagonal(distance, 0)
        distance.flush()
    distance = np.load(distance_path, mmap_mode='r')
    assert distance.shape == (len(train), len(train))
    np.testing.assert_allclose(distance, distance.T, atol=1e-14)
    split_table = []
    for rep in range(20):
        keep, blocks = date_split(n_days, rep)
        split_table.append(dict(repetition=rep, seed=seeded(20, rep), retained_blocks=blocks.tolist(),
                                retained_days=np.flatnonzero(keep), omitted_days=np.flatnonzero(~keep)))
    save_json(run/'date_splits.json', split_table)
    summaries = []
    for k in CONFIG['k_values']:
        out = run/f'k{k}'
        out.mkdir(exist_ok=True)
        if (out/'evaluation.json').exists():
            summaries.append(read_json(out/'evaluation.json'))
            print(f'K={k}: using completed evaluation', flush=True)
            continue
        print(f'K={k}: 10 full-data initializations', flush=True)
        starts = [fit_once(distance, k, seeded(10, k, i)) for i in range(10)]
        best = min(range(10), key=lambda i: starts[i]['loss'])
        medoids = order_medoids(train[starts[best]['medoids']], p, frame)
        d_all = hellinger(p[assign], p[medoids])
        labels = np.full(len(frame), -1, dtype=int)
        labels[assign] = np.argmin(d_all, axis=1)
        baseline = labels[train]
        silhouette = silhouette_samples(distance, baseline, metric='precomputed')
        ari_init = [adjusted_rand_score(a['labels'], b['labels']) for a, b in itertools.combinations(starts, 2)]
        save_json(out/'initializations.json', [dict(**s, medoid_sample_ids=train[s['medoids']]) for s in starts])
        np.savez_compressed(out/'baseline.npz', medoid_ids=medoids, labels=labels,
                            train_ids=train, silhouette=silhouette, distances=d_all, assign_ids=assign)
        checks = np.zeros(len(frame), dtype=int)
        agrees = np.zeros(len(frame), dtype=int)
        block_rows, class_rows = [], []
        for rep in range(20):
            keep, _ = date_split(n_days, rep)
            train_keep = keep[frame.day_index.to_numpy()[train]]
            retained = np.flatnonzero(train_keep)
            test_ids = assign[~keep[frame.day_index.to_numpy()[assign]]]
            assert not set(frame.iloc[train[retained]].day_index) & set(frame.iloc[test_ids].day_index)
            sub_d = np.ascontiguousarray(distance[np.ix_(retained, retained)])
            fits = [fit_once(sub_d, k, seeded(30, k, rep, i)) for i in range(3)]
            fit = min(fits, key=lambda s: s['loss'])
            sub_medoids = train[retained[fit['medoids']]]
            mapping, overlap = match_labels(baseline[retained], fit['labels'], k)
            prediction = np.argmin(hellinger(p[test_ids], p[sub_medoids]), axis=1)
            mapped = mapping[prediction]
            ari = adjusted_rand_score(labels[test_ids], prediction)
            checks[test_ids] += 1
            agrees[test_ids] += mapped == labels[test_ids]
            block_rows.append(dict(repetition=rep, ari=ari, train_samples=len(retained),
                                   check_samples=len(test_ids), seed=seeded(30, k, rep, 0)))
            for c in range(k):
                a, b = labels[test_ids] == c, mapped == c
                union = int((a | b).sum())
                class_rows.append(dict(repetition=rep, cluster=c+1,
                         jaccard=float((a & b).sum()/union) if union else np.nan,
                         baseline_absent=not a.any(), resampled_absent=not b.any(),
                         train_baseline_absent=not (baseline[retained] == c).any()))
            save_json(out/f'perturbation_{rep:02d}.json', dict(
                repetition=rep, medoid_sample_ids=sub_medoids, mapping=mapping, retained_overlap=overlap,
                initializations=[dict(**s, medoid_sample_ids=train[retained[s['medoids']]]) for s in fits]))
            np.savez_compressed(out/f'perturbation_{rep:02d}_checks.npz', sample_ids=test_ids,
                                predicted=prediction, aligned=mapped)
            print(f'K={k}: date-block check {rep+1}/20 ARI={ari:.4f}', flush=True)
        agreement = np.divide(agrees, checks, out=np.full(len(frame), np.nan), where=checks > 0)
        stability = np.full(len(frame), 'unclassified', dtype=object)
        stability[assign] = 'insufficient_evidence'
        stability[(checks >= 3) & (agreement < .8)] = 'unstable'
        stability[(checks >= 3) & (agreement >= .8)] = 'stable'
        table = frame.copy()
        table['cluster'] = np.where(labels >= 0, labels+1, np.nan)
        table['check_count'], table['agreement_fraction'], table['stability'] = checks, agreement, stability
        table['nearest_distance'] = np.nan
        table.loc[assign, 'nearest_distance'] = d_all.min(axis=1)
        nearest2 = np.argsort(d_all, axis=1)[:, :2]+1
        table['second_cluster'] = np.nan
        table.loc[assign, 'second_cluster'] = nearest2[:, 1]
        table['distance_margin'] = np.nan
        table.loc[assign, 'distance_margin'] = np.sort(d_all, axis=1)[:, 1]-d_all.min(axis=1)
        csv(out/'daily_labels.csv', table)
        csv(out/'block_stability.csv', block_rows)
        csv(out/'class_stability.csv', class_rows)
        csv(out/'initialization_ari.csv', [dict(pair=i, ari=a) for i, a in enumerate(ari_init)])
        class_stats, representatives = [], []
        for c, medoid in enumerate(medoids):
            member = train[baseline == c]
            cj = [r['jaccard'] for r in class_rows if r['cluster'] == c+1]
            class_stats.append(dict(cluster=c+1, training_days=len(member),
                 assigned_days=int((labels == c).sum()), distinct_dates=frame.iloc[member].date.nunique(),
                 sensor_count=frame.iloc[member].sensor.nunique(),
                 silhouette_mean=float(silhouette[baseline == c].mean()),
                 block_jaccard_median=quantile(cj, .5), block_jaccard_p10=quantile(cj, .1),
                 missing_check_repetitions=sum(r['baseline_absent'] or r['resampled_absent']
                       for r in class_rows if r['cluster'] == c+1),
                 medoid_sample_id=int(medoid), medoid_sensor=frame.iloc[medoid].sensor,
                 medoid_date=frame.iloc[medoid].date,
                 median_bin=int(np.searchsorted(np.cumsum(p[medoid]), .5))))
            representatives.append(dict(cluster=c+1, sample_id=int(medoid), probabilities=p[medoid],
                   p10=np.quantile(p[member], .1, axis=0), p90=np.quantile(p[member], .9, axis=0)))
        csv(out/'class_summary.csv', class_stats)
        save_json(out/'representatives.json', representatives)
        composition = table[table.cluster.notna()].groupby(['cluster', 'sensor', 'name']).size().reset_index(name='days')
        csv(out/'sensor_composition.csv', composition)
        block_ari = [r['ari'] for r in block_rows]
        summary = dict(k=k, best_initialization=best, silhouette_mean=float(silhouette.mean()),
                       mean_hellinger=float(distance[np.arange(len(train)), np.searchsorted(train, medoids)[baseline]].mean()),
                       init_ari_median=quantile(ari_init, .5), init_ari_p10=quantile(ari_init, .1),
                       block_ari_median=quantile(block_ari, .5), block_ari_p10=quantile(block_ari, .1),
                       min_class_jaccard_median=min(c['block_jaccard_median'] for c in class_stats),
                       unstable_days=int((stability == 'unstable').sum()),
                       insufficient_evidence_days=int((stability == 'insufficient_evidence').sum()))
        t = CONFIG['thresholds']
        gates = dict(init_ari=summary['init_ari_median'] >= t['init_ari'],
                     block_ari=summary['block_ari_median'] >= t['block_ari'],
                     block_ari_p10=summary['block_ari_p10'] >= t['block_ari_p10'],
                     class_jaccard=summary['min_class_jaccard_median'] >= t['class_jaccard'],
                     silhouette=summary['silhouette_mean'] > 0)
        summary.update(passes=all(gates.values()), gates=gates)
        save_json(out/'evaluation.json', summary)
        summaries.append(summary)
    csv(run/'k_comparison.csv', [{k: v for k, v in s.items() if k != 'gates'} for s in summaries])
    eligible = [s for s in summaries if s['passes']]
    recommendation = None
    if eligible:
        top = max(s['silhouette_mean'] for s in eligible)
        recommendation = min(s['k'] for s in eligible if s['silhouette_mean'] >= top-.02)
    save_json(run/'recommendation.json', dict(recommended_k=recommendation,
            status='pending_human_confirmation' if recommendation else 'no_stable_shared_classification',
            candidates=summaries))
    if not (run/'selection.json').exists():
        save_json(run/'main_result.json', dict(k=recommendation, status='pending_human_confirmation'
                  if recommendation else 'no_stable_shared_classification', directory=f'k{recommendation}' if recommendation else None))
    verify_sources(run)
    provenance(run, 'evaluate')
    print(f'Recommended K: {recommendation}', flush=True)


def transitions(labels, k):
    counts = np.zeros((k, k), dtype=int)
    a, b = labels[:-1], labels[1:]
    ok = (a >= 0) & (b >= 0)
    np.add.at(counts, (a[ok], b[ok]), 1)
    return counts


def visible_runs(labels, boundary=None):
    """Censor on unavailable neighbours and source boundary days."""
    boundary = np.zeros(len(labels), dtype=bool) if boundary is None else boundary
    rows = []
    i = 0
    while i < len(labels):
        if labels[i] < 0:
            i += 1
            continue
        j = i+1
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        rows.append(dict(cluster=int(labels[i])+1, start_index=i, end_index=j-1, days=j-i,
                    left_censored=i == 0 or labels[i-1] < 0 or bool(boundary[i]),
                    right_censored=j == len(labels) or labels[j] < 0 or bool(boundary[j-1])))
        i = j
    return rows


def temporal_metrics(labels, k):
    t = transitions(labels, k)
    lengths = [r['days'] for r in visible_runs(labels)]
    return [float(np.trace(t)/t.sum()) if t.sum() else np.nan,
            float(np.mean(lengths)) if lengths else np.nan,
            float(max(lengths)) if lengths else np.nan]


def temporal(run, k):
    out = run/f'k{k}'
    table = pd.read_csv(out/'daily_labels.csv')
    all_trans = np.zeros((k, k), dtype=int)
    stable_trans = np.zeros((k, k), dtype=int)
    runs, sensor_stats, month_stats, null_rows = [], [], [], []
    null_values = np.full((23, 500, 3), np.nan)
    for si, (sensor, group) in enumerate(table.groupby('sensor', sort=False)):
        group = group.sort_values('day_index').reset_index(drop=True)
        labels = group.cluster.fillna(0).to_numpy(dtype=int)-1
        stable = np.where(group.stability == 'stable', labels, -1)
        tm, sm = transitions(labels, k), transitions(stable, k)
        all_trans += tm
        stable_trans += sm
        rr = visible_runs(labels, group.boundary.to_numpy())
        for r in rr:
            runs.append(dict(sensor=sensor, name=group.iloc[0]['name'], **r,
                 start_date=group.iloc[r['start_index']].date, end_date=group.iloc[r['end_index']].date))
        for c in range(k):
            sensor_stats.append(dict(sensor=sensor, name=group.iloc[0]['name'], cluster=c+1,
                   days=int((labels == c).sum()), classified_days=int((labels >= 0).sum()),
                   proportion=float((labels == c).sum()/max(1, (labels >= 0).sum())),
                   adjacent_pairs=int(tm.sum()), switches=int(tm.sum()-np.trace(tm)),
                   switching_rate=float(1-np.trace(tm)/tm.sum()) if tm.sum() else np.nan))
        months = group.date.str[:7].to_numpy()
        for month in np.unique(months):
            sel = months == month
            lm = labels[sel]
            # Transitions assigned to destination month, including month boundary.
            pair = (labels[:-1] >= 0) & (labels[1:] >= 0) & sel[1:]
            switches = int(((labels[1:] != labels[:-1]) & pair).sum())
            for c in range(k):
                month_stats.append(dict(sensor=sensor, month=month, cluster=c+1,
                     days=int((lm == c).sum()), classified_days=int((lm >= 0).sum()),
                     proportion=float((lm == c).sum()/max(1, (lm >= 0).sum())),
                     adjacent_pairs=int(pair.sum()), switches=switches,
                     switching_rate=float(switches/pair.sum()) if pair.any() else np.nan))
        observed = temporal_metrics(labels, k)
        rng = np.random.default_rng(seeded(40, k, si))
        ids = np.flatnonzero(labels >= 0)
        shuffled = labels.copy()
        for rep in range(500):
            shuffled[ids] = rng.permutation(labels[ids])
            null_values[si, rep] = temporal_metrics(shuffled, k)
        for j, name in enumerate(['self_transition_fraction', 'mean_visible_run_days', 'max_visible_run_days']):
            v = null_values[si, :, j]
            null_rows.append(dict(sensor=sensor, metric=name, observed=observed[j],
                       null_median=quantile(v, .5), null_p025=quantile(v, .025), null_p975=quantile(v, .975),
                       p_greater=(1+int((v >= observed[j]).sum()))/501 if np.isfinite(observed[j]) else np.nan,
                       seed=seeded(40, k, si)))
    csv(out/'visible_runs.csv', runs)
    csv(out/'sensor_proportions_switches.csv', sensor_stats)
    csv(out/'monthly_proportions_switches.csv', month_stats)
    csv(out/'temporal_permutation.csv', null_rows)
    np.savez_compressed(out/'temporal_permutations.npz', values=null_values,
                        sensor_order=table.sensor.drop_duplicates().to_numpy(dtype=str))
    for name, counts in [('all', all_trans), ('stable', stable_trans)]:
        den = counts.sum(axis=1)
        prob = np.divide(counts, den[:, None], out=np.full((k, k), np.nan), where=den[:, None] > 0)
        csv(out/f'transitions_{name}.csv', [dict(from_cluster=i+1, to_cluster=j+1,
              count=int(counts[i, j]), probability=prob[i, j]) for i in range(k) for j in range(k)])
    uncertainty = table[table.stability.isin(['unstable', 'insufficient_evidence'])].copy()
    uncertainty = uncertainty.sort_values(['stability', 'distance_margin', 'sensor', 'date'])
    csv(out/'uncertain_dates.csv', uncertainty)
    return table


def plot_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'svg.fonttype': 'path',
                         'font.size': 10, 'savefig.bbox': 'tight'})
    return plt


def save_svg(fig, path):
    fig.savefig(path, format='svg', metadata={'Date': None, 'Creator': 'daily_modes.py'})
    import matplotlib.pyplot as plt
    plt.close(fig)


def distribution_axis(ax):
    ax.set_xticks([0, 11, 21, 31, 41, 51],
                  ['<0.01', '0.1', '1', '10', '100', '>1000'], rotation=35)
    ax.set_xlabel('间隔箱 / 秒（含上下溢箱）')
    ax.set_ylabel('箱内概率')
    ax.set_xlim(-.5, 51.5)


def bin_description(index):
    index = int(index)
    if index == 0:
        return '<0.01秒'
    if index == 51:
        return '>1000秒'
    return f'[{EDGES[index-1]:.4g}, {EDGES[index]:.4g}' + (']秒' if index == 50 else ')秒')


def figures(run, k, table, p, plt):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    out = run/f'k{k}'
    reps = read_json(out/'representatives.json')
    base = np.load(out/'baseline.npz')
    colors = plt.get_cmap('tab10').colors[:k]
    fig, axes = plt.subplots(k, 1, figsize=(11, 2.7*k), sharex=True, layout='constrained')
    for ax, rep in zip(axes, reps):
        c = rep['cluster']-1
        row = table.iloc[rep['sample_id']]
        ax.fill_between(np.arange(52), rep['p10'], rep['p90'], color=colors[c], alpha=.2, step='mid', label='类内10%—90%')
        ax.plot(rep['probabilities'], color=colors[c], lw=1.8, drawstyle='steps-mid', label='真实代表日')
        ax.set_title(f'C{c+1} | {row.sensor} | {row["name"]} | {row.date}')
        distribution_axis(ax)
        ax.legend(loc='upper right')
    save_svg(fig, out/'representative_distributions.svg')
    n_days = int(table.day_index.max()+1)
    data = np.full((23, n_days), -3, dtype=int)
    names = []
    for i, (_, group) in enumerate(table.groupby('sensor', sort=False)):
        names.append(group.iloc[0]['name'])
        vals = np.where(group.status == 'outside_coverage', -3,
                np.where(group.status == 'no_valid_observations', -2, -1))
        classified = group.cluster.notna().to_numpy()
        vals[classified] = group.cluster[classified].to_numpy(dtype=int)-1
        data[i] = vals
    fig, ax = plt.subplots(figsize=(16, 10), layout='constrained')
    palette = ['#f4f4f4', '#9e9e9e', '#d4d4d4']+list(colors)
    ax.imshow(data, aspect='auto', interpolation='nearest', cmap=ListedColormap(palette),
              norm=BoundaryNorm(np.arange(-3.5, k+.5), len(palette)))
    ax.set_yticks(np.arange(23), names)
    dates = table[table.sensor_order == 0].date.to_numpy()
    ticks = np.r_[0, [i for i in range(1, n_days) if dates[i][8:] == '01']]
    ax.set_xticks(ticks, [dates[i][:7] for i in ticks], rotation=40, ha='right')
    ax.set_xlabel('上海时间 / 日期')
    ax.set_title(f'JQRD GT1 每日记录间隔模式 | K={k}')
    labels = ['覆盖外', '无有效记录', '间隔不足']+[f'C{i+1}' for i in range(k)]
    ax.legend(handles=[Patch(color=col, label=lab) for col, lab in zip(palette, labels)],
              ncol=k+3, loc='upper center', bbox_to_anchor=(.5, 1.10), fontsize=9)
    save_svg(fig, out/'class_timeline.svg')
    # Deterministic uncertain examples: smallest margins, unstable first, max one per sensor/date.
    examples = table[table.stability == 'unstable'].sort_values(['distance_margin', 'sample_id']).head(6)
    if examples.empty:
        examples = table[table.stability == 'insufficient_evidence'].sort_values(['distance_margin', 'sample_id']).head(6)
    csv(out/'uncertain_examples.csv', examples)
    if len(examples):
        fig, axes = plt.subplots(len(examples), 1, figsize=(11, 2.7*len(examples)), layout='constrained')
        for ax, (_, r) in zip(np.atleast_1d(axes), examples.iterrows()):
            ax.plot(p[int(r.sample_id)], color='black', label='当日分布')
            for c in [int(r.cluster), int(r.second_cluster)]:
                ax.plot(p[base['medoid_ids'][c-1]], '--', color=colors[c-1], label=f'C{c} 代表')
            ax.set_title(f'{r.sensor} | {r["name"]} | {r.date} | 一致比例 {r.agreement_fraction:.0%}')
            distribution_axis(ax)
            ax.legend()
        save_svg(fig, out/'uncertain_distributions.svg')


def main_figures(run, k, plt):
    out = run/f'k{k}'
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, scope, title in zip(axes, ['all', 'stable'], ['全部已分类日期', '仅稳定日期（其余断开）']):
        t = pd.read_csv(out/f'transitions_{scope}.csv')
        prob = t.probability.to_numpy().reshape(k, k)
        im = ax.imshow(prob, vmin=0, vmax=1, cmap='Blues')
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f'{prob[i,j]:.2f}' if np.isfinite(prob[i,j]) else '—',
                        ha='center', va='center', color='white' if prob[i,j] > .5 else 'black')
        ax.set_xticks(range(k), [f'C{i+1}' for i in range(k)])
        ax.set_yticks(range(k), [f'C{i+1}' for i in range(k)])
        ax.set_xlabel('次日类别'); ax.set_ylabel('当日类别'); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
    save_svg(fig, out/'transition_matrices.svg')
    runs = pd.read_csv(out/'visible_runs.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    colors = plt.get_cmap('tab10').colors
    for c in range(1, k+1):
        r = runs[runs.cluster == c]
        for ax, v in [(axes[0], r.days), (axes[1], r.loc[~r.left_censored & ~r.right_censored, 'days'])]:
            if len(v):
                x = np.sort(v)
                ax.step(x, np.arange(1, len(x)+1)/len(x), where='post', label=f'C{c} (n={len(x)})', color=colors[c-1])
    for ax, title in zip(axes, ['全部连续可见段（包含删失段）', '两端均未删失的可见段']):
        ax.set_title(title); ax.set_xlabel('连续可见天数'); ax.set_ylabel('经验累计比例')
        ax.set_xscale('log'); ax.legend()
    save_svg(fig, out/'visible_duration.svg')


def report(run):
    frame, p = load_data(run)
    rec = read_json(run/'recommendation.json')
    main = read_json(run/'main_result.json')
    plt = plot_setup()
    comparison = pd.read_csv(run/'k_comparison.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, field, title, threshold in zip(axes.flat,
            ['silhouette_mean', 'init_ari_median', 'block_ari_median', 'min_class_jaccard_median'],
            ['平均轮廓系数', '初始化 ARI 中位数', '日期扰动 ARI', '最弱类别 Jaccard 中位数'], [0, .9, .8, .6]):
        ax.plot(comparison.k, comparison[field], 'o-', label=title)
        ax.axhline(threshold, color='gray', ls='--', label=f'门槛 {threshold}')
        if field == 'block_ari_median':
            ax.plot(comparison.k, comparison.block_ari_p10, 's--', label='10%分位数')
            ax.axhline(.6, color='silver', ls=':')
        ax.set_xticks(CONFIG['k_values']); ax.set_xlabel('类别数 K'); ax.set_title(title); ax.legend()
    save_svg(fig, run/'k_stability_comparison.svg')
    for k in CONFIG['k_values']:
        print(f'report K={k}: temporal analysis and 500 permutations per sensor', flush=True)
        table = temporal(run, k)
        figures(run, k, table, p, plt)
    if main['k'] is not None:
        main_figures(run, main['k'], plt)
    write_report(run)
    verify_sources(run)
    provenance(run, 'report')


def write_report(run):
    frame, _ = load_data(run)
    rec = read_json(run/'recommendation.json')
    main = read_json(run/'main_result.json')
    audits = read_json(run/'source_audit.json')
    text = ['JQRD GT1 每日记录间隔共享聚类实验', '',
            f'运行编号：{run.name}；主种子：20261013。',
            '范围：23测点、全部源记录时段、上海时间自然日；燃料组分仅CH4。',
            '每日50个对数箱（0.01—1000秒）加上下溢，共52维；Hellinger + FasterPAM。',
            '这是记录间隔模式的探索性分类，不代表故障原因、质量等级或填补准确率。', '',
            f'来源行数：{sum(a["rows"] for a in audits):,}；正间隔：{sum(a["positive_intervals"] for a in audits):,}。',
            f'统一日历：{frame.date.min()} 至 {frame.date.max()}；共{len(frame)}个测点日。',
            f'学习样本：{int(frame.train_eligible.sum())}；可分配标签：{int(frame.assignable.sum())}。',
            f'源首末边界日：{int(frame.boundary.sum())}（不参与中心学习；可分配日期独立标记）。',
            '日期状态比例：']
    for name, n in frame.status.value_counts().items():
        text.append(f'  {name}: {n} ({n/len(frame):.2%})')
    text += ['', '类别数比较（预先固定门槛为工程规则，并非通用统计定律）：',
             'K | silhouette | init ARI median | block ARI median / p10 | min class Jaccard | pass']
    for s in rec['candidates']:
        text.append(f'{s["k"]} | {s["silhouette_mean"]:.4f} | {s["init_ari_median"]:.4f} | '
            f'{s["block_ari_median"]:.4f} / {s["block_ari_p10"]:.4f} | {s["min_class_jaccard_median"]:.4f} | {s["passes"]}')
        if not s['passes']:
            text.append('  未通过：'+', '.join(n for n, passed in s['gates'].items() if not passed))
    if rec['recommended_k'] is None:
        text += ['', '结论：当前设置尚不能获得稳定共享分类。未自动修改分箱或扩大类别数。']
    else:
        text += ['', f'规则推荐 K={rec["recommended_k"]}，待人工确认。',
                 '推荐依据：先通过全部稳定性门槛，再在合格模型最高轮廓系数0.02以内选择最小K。']
        best = max((s for s in rec['candidates'] if s['passes']), key=lambda s: s['silhouette_mean'])
        chosen = next(s for s in rec['candidates'] if s['k'] == rec['recommended_k'])
        text.append(f'合格模型中最高轮廓系数来自K={best["k"]}（{best["silhouette_mean"]:.4f}）；'
                    f'推荐方案差值{best["silhouette_mean"]-chosen["silhouette_mean"]:.4f}，优先较小K。')
    if main['k']:
        k = main['k']
        table = pd.read_csv(run/f'k{k}'/'daily_labels.csv')
        cs = pd.read_csv(run/f'k{k}'/'class_summary.csv')
        text += [f'当前主结果 K={k}；状态：{main["status"]}。', '', '主结果类别（编号按代表中位间隔箱排序）：']
        for _, r in cs.iterrows():
            text.append(f'  C{int(r.cluster)}: {int(r.assigned_days)}天，{int(r.sensor_count)}测点，'
                f'代表 {r.medoid_sensor} / {r.medoid_date}，中位间隔箱 {bin_description(r.median_bin)}，'
                f'Jaccard中位数 {r.block_jaccard_median:.3f}，轮廓 {r.silhouette_mean:.3f}。')
        for flag in ['stable', 'unstable', 'insufficient_evidence']:
            text.append(f'  {flag}: {int((table.stability == flag).sum())} 天')
        weakest = cs.loc[cs.silhouette_mean.idxmin()]
        text.append(f'C{int(weakest.cluster)}的类内轮廓系数最低（{weakest.silhouette_mean:.3f}）；'
                    '共享类别稳定不等于类内分布完全同质，应结合10%—90%范围和更细K的代表分布判断。')
        text.append('不稳定日期的类别组成：'+', '.join(
                    f'C{int(c)}={n}' for c, n in table.loc[table.stability == 'unstable', 'cluster'].value_counts().sort_index().items()))
        tm = pd.read_csv(run/f'k{k}'/'transitions_all.csv')
        sm = pd.read_csv(run/f'k{k}'/'transitions_stable.csv')
        for title, t in [('全部标签', tm), ('仅稳定标签', sm)]:
            total = t['count'].sum()
            self_count = t.loc[t.from_cluster == t.to_cluster, 'count'].sum()
            text.append(f'{title}：相邻日对 {total}，自转移比例 {self_count/total:.2%}。' if total else f'{title}：无有效日对。')
        null = pd.read_csv(run/f'k{k}'/'temporal_permutation.csv')
        self_null = null[null.metric == 'self_transition_fraction']
        text.append(f'{int((self_null.observed > self_null.null_p975).sum())}/23测点的自转移比例高于各自打乱分布97.5%分位数。')
        text.append('随机打乱检验用于描述时间持续性；逐测点p值未经多重比较校正，不作为故障因果证据。')
        rr = pd.read_csv(run/f'k{k}'/'visible_runs.csv')
        text.append(f'连续可见段 {len(rr)}，至少一端删失 {int((rr.left_censored | rr.right_censored).sum())}；'
                    f'中位长度 {rr.days.median():.1f} 天，最长 {rr.days.max()} 天。')
    text += ['', '解释及边界：',
        '1. 类别编号在每个K内排序，不表示跨K类别一一对应；代表是实际测点日，不是拟合分布。',
        '2. 日分布等权；边界日排除中心学习，完整跨日间隔仅归左端一天。无记录日保持独立状态。',
        '3. 超1000秒间隔覆盖各日的时长另表保存，不进入聚类；不能用52维日分布独自描述长缺口持续机制。',
        '4. 初始化10次，日期块扰动20次/每次3次初始化；所有测点共享7日块划分。',
        '5. 匹配类别仅使用保留日期；一致性仅统计未抽取日期。全量基准作为比较参考，因此不是独立外部验证。',
        '6. 转移只连接相邻自然日；稳定标签分析在不稳定及证据不足日期处断开。',
        '7. 连续可见长度受日聚合及记录边界限制；删失段不是完整机制持续时间；图中不作生存率估计。',
        '8. 模式可能反映正常采样、归档、时间量化或停止记录，原因需要外部采集证据。',
        '9. 500次打乱保持各测点类别计数和未分类位置；这是交换性参考，不能排除共同时间趋势或工况影响。',
        '10. 来源核验使用文件大小/纳秒修改时间及选取表SHA256；CSV未另做全文件内容哈希。', '',
        '证据入口：k_comparison.csv、recommendation.json、main_result.json；',
        'k*/daily_labels.csv、class_summary.csv、representatives.json、class_stability.csv；',
        'k*/monthly_proportions_switches.csv、transitions_*.csv、visible_runs.csv、temporal_permutation.csv；',
        'k*/uncertain_dates.csv、uncertain_examples.csv、uncertain_distributions.svg；',
        '验证记录：verification.json、synthetic_tests.txt、source_audit.json、provenance_*.json。', '',
        '人工选择：python missing_analysis/daily_modes.py select --run '+str(run)+' --k N',
        '选择只更新主结果索引、报告和该K的转移/持续图，不重新扫描来源或拟合。']
    (run/'report.txt').write_text('\n'.join(text)+'\n', encoding='utf-8')


def select(run, k):
    assert k in CONFIG['k_values'] and (run/f'k{k}'/'evaluation.json').exists()
    save_json(run/'selection.json', dict(k=k, method='user_cli_selection', utc_time=pd.Timestamp.now(tz='UTC').isoformat()))
    save_json(run/'main_result.json', dict(k=k, status='human_confirmed', directory=f'k{k}'))
    if not (run/f'k{k}'/'transitions_all.csv').exists():
        temporal(run, k)
    main_figures(run, k, plot_setup())
    write_report(run)
    provenance(run, 'select')


def verify(run):
    frame, p = load_data(run)
    z = np.load(run/'daily_profiles.npz')
    audit = read_json(run/'source_audit.json')
    np.testing.assert_array_equal(z['counts'].sum(axis=1), frame.completed_intervals)
    assert int(z['counts'].sum()) == sum(a['positive_intervals'] for a in audit)
    np.testing.assert_allclose(p[frame.assignable].sum(axis=1), 1)
    train = np.flatnonzero(frame.train_eligible)
    distance = np.load(run/'hellinger_train.npy', mmap_mode='r')
    reproducible = []
    for k in CONFIG['k_values']:
        # Repeat the selected full-data initialization with exactly the saved seed.
        e = read_json(run/f'k{k}'/'evaluation.json')
        original = read_json(run/f'k{k}'/'initializations.json')[e['best_initialization']]
        repeated = fit_once(distance, k, original['seed'])
        np.testing.assert_array_equal(repeated['medoids'], original['medoids'])
        np.testing.assert_array_equal(repeated['labels'], original['labels'])
        np.testing.assert_allclose(repeated['loss'], original['loss'], atol=1e-12, rtol=0)
        base = np.load(run/f'k{k}'/'baseline.npz')
        assert set(base['medoid_ids']).issubset(set(train))
        # Replay one complete perturbation, verifying its omitted labels and train-only mapping.
        keep, _ = date_split(int(frame.day_index.max()+1), 0)
        retained = np.flatnonzero(keep[frame.iloc[train].day_index])
        old = read_json(run/f'k{k}'/'perturbation_00.json')
        best = min(old['initializations'], key=lambda r: r['loss'])
        fit = fit_once(np.ascontiguousarray(distance[np.ix_(retained, retained)]), k, best['seed'])
        mapping, _ = match_labels(base['labels'][train[retained]], fit['labels'], k)
        checks = np.load(run/f'k{k}'/'perturbation_00_checks.npz')
        m = train[retained[fit['medoids']]]
        predicted = np.argmin(hellinger(p[checks['sample_ids']], p[m]), axis=1)
        np.testing.assert_array_equal(mapping[predicted], checks['aligned'])
        reproducible.append(k)
    verify_sources(run)
    save_json(run/'verification.json', dict(daily_counts_match_source=True, probability_normalized=True,
              source_metadata_unchanged=True, historical_audit_matches=sum(a.get('historical_audit_match', False) for a in audit),
              exact_seed_replays_k=reproducible, replay_scope='best full initialization and first date perturbation per K',
              source_rows=sum(a['rows'] for a in audit), positive_intervals=int(z['counts'].sum())))
    provenance(run, 'verify')
    print('VERIFICATION PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'evaluate', 'report', 'select', 'verify'])
    parser.add_argument('--run', type=Path, default=ROOT/'missing_analysis'/'runs'/'jqrd_gt1_daily_v1')
    parser.add_argument('--k', type=int)
    args = parser.parse_args()
    t0 = time.perf_counter()
    try:
        if args.command == 'select':
            if args.k is None:
                parser.error('select requires --k')
            select(args.run, args.k)
        else:
            globals()[args.command](args.run)
    except Exception as exc:
        if args.run.exists():
            status = read_json(args.run/'status.json') if (args.run/'status.json').exists() else {}
            status[args.command] = dict(state='failed', error=repr(exc),
                                      utc_time=pd.Timestamp.now(tz='UTC').isoformat())
            save_json(args.run/'status.json', status)
        raise
    print(f'{args.command} complete in {time.perf_counter()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
