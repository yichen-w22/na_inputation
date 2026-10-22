"""Interval-only learning, Top3 comparison plots and callable masking.

API: learn(timestamps, values, Settings(...)); generate_mask(model, timestamps,
seed=...). CLI: python missing_analysis/interval_mask.py --help.
Settings are exploratory choices, not confirmed research thresholds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


@dataclass
class Settings:
    # Required: do not silently decide what constitutes stable high frequency.
    high_window_seconds: float
    high_p95_seconds: float
    high_cv: float
    windows_seconds: tuple
    step_seconds: float
    classes: tuple = (1, 2, 3, 4)
    min_intervals: int = 30
    min_class_windows: int = 5
    stability_ari: float = .8
    repeats: int = 5
    fit_sample_limit: int = 100000
    seed: int = 20261021

    def validate(self):
        positive = [self.high_window_seconds, self.high_p95_seconds,
                    self.step_seconds, *self.windows_seconds]
        if not self.windows_seconds or any(not np.isfinite(x) or x <= 0 for x in positive):
            raise ValueError('窗口、步长和高频间隔阈值必须为正有限数')
        if (not np.isfinite(self.high_cv) or self.high_cv < 0 or
                not 0 <= self.stability_ari <= 1):
            raise ValueError('CV或稳定性阈值无效')
        if (not self.classes or any(int(k) != k or k < 1 for k in self.classes)
                or self.min_intervals < 2 or self.min_class_windows < 2
                or self.repeats < 2 or self.fit_sample_limit < self.min_intervals):
            raise ValueError('类别数、样本量或重复次数无效')


def valid_times(timestamps, values=None):
    """Valid measurements only; reject duplicate/reversed times, never alter values."""
    t = pd.DatetimeIndex(pd.to_datetime(timestamps, format='mixed'))
    if t.tz is not None:
        t = t.tz_convert('Asia/Shanghai').tz_localize(None)
    good = ~t.isna()
    if values is not None:
        v = np.asarray(values, dtype=float)
        if len(v) != len(t):
            raise ValueError('时间戳和值长度不同')
        good &= np.isfinite(v)
    ns = t.asi8[good]
    if len(ns) < 2 or np.any(np.diff(ns) <= 0):
        raise ValueError('至少需要两条有效记录，且时间戳严格递增')
    return ns, np.flatnonzero(good)


def high_frequency(t, d, cfg):
    """Disjoint full blocks; both interval endpoints must lie inside a block.

    A block must cover its edges to within the high-frequency p95 threshold.
    Intervals touching an excluded block are excluded from learning as a whole.
    """
    width = cfg.high_window_seconds
    excluded = np.zeros(len(d), dtype=bool)
    blocks = []
    for left in np.arange(0., t[-1] - width + 1e-9, width):
        right = left + width
        a = np.searchsorted(t, left, side='left')
        b = np.searchsorted(t, right, side='right')
        x = d[a:max(a, b - 1)]
        if len(x) < cfg.min_intervals:
            continue
        if (t[a] - left > cfg.high_p95_seconds or right - t[b - 1] > cfg.high_p95_seconds):
            continue
        if np.quantile(x, .95) <= cfg.high_p95_seconds and x.std() / x.mean() <= cfg.high_cv:
            first = np.searchsorted(t[1:], left, side='right')
            last = np.searchsorted(t[:-1], right, side='left')
            excluded[first:last] = True
            if blocks and blocks[-1][1] == left:
                blocks[-1][1] = float(right)
            else:
                blocks.append([float(left), float(right)])
    return excluded, blocks


def window_features(t, d, excluded, width, cfg):
    # Common endpoint grid for meaningful cross-window ARI comparisons.
    ends = np.arange(max(cfg.windows_seconds), t[-1] + 1e-9, cfg.step_seconds)
    features, members, accepted = [], [], []
    for end in ends:
        a = np.searchsorted(t[1:], end - width, side='right')
        b = np.searchsorted(t[1:], end, side='right')
        ids = np.arange(a, b)
        ids = ids[~excluded[ids]]
        if len(ids) < cfg.min_intervals:
            continue
        x = np.log(d[ids])
        features.append([*np.quantile(x, [.1, .5, .9, .99]), x.mean(), x.std()])
        members.append(ids)
        accepted.append(float(end))
    return np.asarray(features), members, accepted


def fit_distribution(x):
    """BIC selects a candidate family; KS is diagnostic, not a calibrated test."""
    if np.ptp(x) <= max(1e-12, float(x.mean()) * 1e-10):
        return dict(family='constant', parameters=[float(x[0])], candidates=[])
    candidates = []
    for name, parameters in [('expon', 1), ('lognorm', 2), ('gamma', 2), ('weibull_min', 2)]:
        dist = getattr(stats, name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                p = dist.fit(x, floc=0)
                ll = float(np.sum(dist.logpdf(x, *p)))
                ks = float(stats.kstest(x, name, args=p).statistic)
            if np.isfinite(ll) and np.all(np.isfinite(p)):
                candidates.append(dict(family=name, parameters=list(map(float, p)),
                                       bic=float(parameters * np.log(len(x)) - 2 * ll), ks=ks))
        except (ValueError, FloatingPointError):
            continue
    if not candidates:
        raise ValueError('所有候选分布拟合失败')
    best = min(candidates, key=lambda v: v['bic'])
    return dict(**best, candidates=candidates)


def learn(timestamps, values, cfg):
    """Learn one sensor independently; returns diagnostics even if no stable model."""
    cfg.validate()
    ns, _ = valid_times(timestamps, values)
    t = (ns - ns[0]) / 1e9
    d = np.diff(t)
    excluded, blocks = high_frequency(t, d, cfg)
    candidates = []
    storage = {}
    rng = np.random.default_rng(cfg.seed)
    for width in sorted(set(cfg.windows_seconds)):
        f, members, ends = window_features(t, d, excluded, width, cfg)
        if len(f) < cfg.min_class_windows:
            continue
        z = StandardScaler().fit_transform(f)
        for k in sorted(set(cfg.classes)):
            if len(f) < k * cfg.min_class_windows or len(np.unique(z, axis=0)) < k:
                continue
            with threadpool_limits(limits=1):
                km = KMeans(k, n_init=10, random_state=cfg.seed).fit(z)
                labels = km.labels_
                aris = []
                for rep in range(cfg.repeats):
                    ids = rng.choice(len(z), max(k, int(.8 * len(z))), replace=False)
                    other = KMeans(k, n_init=5, random_state=cfg.seed + rep + 1).fit(z[ids])
                    aris.append(float(adjusted_rand_score(labels, other.predict(z))))
            counts = np.bincount(labels, minlength=k)
            row = dict(window_seconds=float(width), k=int(k), windows=len(f),
                       resample_ari_min=min(aris), cross_window_ari=None,
                       minimum_class_windows=int(counts.min()),
                       silhouette=float(silhouette_score(z, labels, sample_size=min(2000, len(z)),
                                                        random_state=cfg.seed)) if k > 1 else 0.)
            candidates.append(row)
            storage[(width, k)] = (members, ends, labels)
    # Require agreement at a neighboring tested width, for the same K.
    for row in candidates:
        peers = [v for v in candidates if v['k'] == row['k'] and v['window_seconds'] != row['window_seconds']]
        if not peers:
            continue
        peer = min(peers, key=lambda v: abs(v['window_seconds'] - row['window_seconds']))
        _, ea, la = storage[(row['window_seconds'], row['k'])]
        _, eb, lb = storage[(peer['window_seconds'], peer['k'])]
        _, ia, ib = np.intersect1d(ea, eb, return_indices=True)
        if len(ia) >= row['k'] * cfg.min_class_windows:
            row['cross_window_ari'] = float(adjusted_rand_score(la[ia], lb[ib]))
    stable = [v for v in candidates if v['resample_ari_min'] >= cfg.stability_ari
              and v['cross_window_ari'] is not None and v['cross_window_ari'] >= cfg.stability_ari
              and v['minimum_class_windows'] >= cfg.min_class_windows]
    result = dict(status='no_stable_model', config=asdict(cfg), candidates=candidates,
                  valid_records=len(ns), excluded_intervals=int(excluded.sum()),
                  excluded_blocks_seconds=blocks, origin=str(pd.Timestamp(ns[0])),
                  protocol='interval_only_v1', scope='independent_sensor',
                  caveat='Exploratory stability and family selection; no goodness-of-fit guarantee.')
    if not stable:
        return result
    # Prefer informative separation; K=1 remains a legitimate fallback.
    ranked = sorted(stable, key=lambda v: (-v['silhouette'], v['k'], v['window_seconds']))
    top = []
    for selected in ranked:
        fitted = materialize(selected, storage, t, d, cfg)
        if fitted is not None:
            top.append(fitted)
        if len(top) == 3:
            break
    result['top3'] = top
    if not top:
        result['reason'] = '去重后类内间隔不足'
        return result
    # Keep the mask-generation API using the highest-ranked eligible scheme.
    result.update(status='stable', **{key: top[0][key] for key in
                  ('selected', 'distributions', 'mode_runs')},
                  generation_policy='random-phase cyclic replay of observed mode runs; independent intervals within mode')
    return result


def materialize(selected, storage, t, d, cfg):
    rng = np.random.default_rng(cfg.seed)
    members, ends, labels = storage[(selected['window_seconds'], selected['k'])]
    # Canonical colors: C1..Ck sorted by median interval, not arbitrary KMeans IDs.
    medians = [np.median([np.median(d[ids]) for ids, lab in zip(members, labels)
                         if lab == label]) for label in range(selected['k'])]
    remap = np.empty(selected['k'], dtype=int)
    remap[np.argsort(medians, kind='stable')] = np.arange(selected['k'])
    labels = remap[labels]
    # Every interval is assigned once, using its closest accepted window endpoint.
    owner = np.full(len(d), -1, dtype=int)
    distance = np.full(len(d), np.inf)
    for ids, end, label in zip(members, ends, labels):
        dist = np.abs(t[ids + 1] - end)
        use = dist < distance[ids]
        owner[ids[use]] = label
        distance[ids[use]] = dist[use]
    distributions = []
    for label in range(selected['k']):
        x = d[owner == label]
        n = len(x)
        if n < cfg.min_intervals:
            return None
        observed_min, observed_max = float(x.min()), float(x.max())
        if n > cfg.fit_sample_limit:
            x = rng.choice(x, cfg.fit_sample_limit, replace=False)
        distributions.append(dict(**fit_distribution(x), intervals=n, fitted_intervals=len(x),
                                  observed_min_seconds=observed_min, observed_max_seconds=observed_max))
    # Replay contiguous mode runs; never join across unaccepted windows.
    runs = []
    for end, label in zip(ends, labels):
        if runs and label == runs[-1]['label'] and np.isclose(end - previous, cfg.step_seconds):
            runs[-1]['duration_seconds'] += cfg.step_seconds
        else:
            runs.append(dict(label=int(label), duration_seconds=float(cfg.step_seconds)))
        previous = end
    counts, probabilities = transition_matrix(ends, labels, cfg.step_seconds, selected['k'])
    return dict(selected=selected, distributions=distributions, mode_runs=runs,
                window_ends_seconds=ends, labels=labels.tolist(),
                transition_counts=counts.tolist(), transition_probabilities=probabilities.tolist())


def transition_matrix(ends, labels, step, k):
    """Adjacent accepted grid cells only, including self transitions; no gap bridging."""
    counts = np.zeros((k, k), dtype=int)
    for a, b, la, lb in zip(ends[:-1], ends[1:], labels[:-1], labels[1:]):
        if np.isclose(b - a, step, rtol=0, atol=1e-7):
            counts[la, lb] += 1
    total = counts.sum(axis=1, keepdims=True)
    p = np.divide(counts, total, out=np.full((k, k), np.nan), where=total > 0)
    return counts, p


def plot_top3(result, output_dir, title=''):
    """Exactly three SVGs; each compares up to three eligible schemes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    schemes = result.get('top3', [])
    if not schemes:
        raise ValueError('无合格方案，无法绘制Top3图')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'svg.fonttype': 'none'})
    max_k = max(s['selected']['k'] for s in schemes)
    colors = list(plt.get_cmap('tab10' if max_k <= 10 else 'tab20').colors)
    n = len(schemes)
    figures = [plt.subplots(1, n, figsize=(6*n, 5), squeeze=False, layout='constrained'),
               plt.subplots(1, n, figsize=(5*n, 5.8), squeeze=False, layout='constrained'),
               plt.subplots(n, 1, figsize=(15, 2.8*n+1), squeeze=False, layout='constrained')]
    # Shared density range; a log x-axis still shows density per second, not log-density.
    bounds = []
    for scheme in schemes:
        for dist in scheme['distributions']:
            if 'observed_min_seconds' in dist:
                bounds.extend([dist['observed_min_seconds']*.9, dist['observed_max_seconds']*1.1])
            elif dist['family'] == 'constant':
                value = dist['parameters'][0]
                bounds.extend([value * .8, value * 1.2])
            else:
                bounds.extend(getattr(stats, dist['family']).ppf([.001, .999], *dist['parameters']))
    bounds = np.asarray(bounds)
    bounds = bounds[np.isfinite(bounds) & (bounds > 0)]
    if not len(bounds):
        raise ValueError('概率密度显示范围无效')
    x = np.geomspace(max(bounds.min(), 1e-9), max(bounds.max(), bounds.min()*1.01), 1500)
    for rank, scheme in enumerate(schemes):
        selected = scheme['selected']
        k = selected['k']
        heading = (f"Top {rank+1} | 窗口 {pd.Timedelta(seconds=selected['window_seconds'])} | K={k}\n"
                   f"轮廓系数 {selected['silhouette']:.3f} | 重抽样ARI {selected['resample_ari_min']:.3f}"
                   f" | 跨窗ARI {selected['cross_window_ari']:.3f}")
        ax = figures[0][1][0, rank]
        peak = 0.
        for label, dist in enumerate(scheme['distributions']):
            color = colors[label % len(colors)]
            name = f"C{label+1} · {dist['family']}"
            if dist['family'] == 'constant':
                ax.axvline(dist['parameters'][0], color=color, label=name+'（点质量）')
            else:
                y = getattr(stats, dist['family']).pdf(x, *dist['parameters'])
                finite = np.isfinite(y) & (y > 0)
                if np.any(finite):
                    peak = max(peak, float(y[finite].max()))
                ax.plot(x, np.where(finite, y, np.nan), color=color, label=name)
        ax.set(xscale='log', xlabel='相邻记录间隔 / 秒（对数轴）', ylabel='概率密度（1/秒；对数轴）', title=heading)
        if peak > 0:
            ax.set_yscale('log')
            ax.set_ylim(peak*1e-6, peak*2)
        else:
            ax.set_ylabel('常数间隔：竖线表示点质量')
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
        ax = figures[1][1][0, rank]
        p = np.asarray(scheme['transition_probabilities'])
        cmap = plt.get_cmap('Blues').copy()
        cmap.set_bad('#dddddd')
        im = ax.imshow(np.ma.masked_invalid(p), vmin=0, vmax=1, cmap=cmap)
        for a in range(k):
            for b in range(k):
                ax.text(b, a, f'{p[a,b]:.2f}' if np.isfinite(p[a,b]) else '—',
                        ha='center', va='center', color='white' if p[a,b] > .5 else 'black')
        ax.set_xticks(range(k), [f'C{i+1}' for i in range(k)])
        ax.set_yticks(range(k), [f'C{i+1}' for i in range(k)])
        matrix_heading = (f"Top {rank+1} | 窗口 {selected['window_seconds']/3600:g} h | K={k}\n"
                          f"相邻窗口对 {int(np.sum(scheme['transition_counts'])):,}")
        ax.set(xlabel='下一窗口类别', ylabel='当前窗口类别', title=matrix_heading)
        figures[1][0].colorbar(im, ax=ax, shrink=.7)
        ax = figures[2][1][rank, 0]
        origin = pd.Timestamp(result['origin'])
        step = result['config']['step_seconds']
        for label in range(k):
            segments = []
            for end, lab in zip(scheme['window_ends_seconds'], scheme['labels']):
                if lab == label:
                    right = mdates.date2num(origin + pd.Timedelta(seconds=end))
                    segments.append((right-step/86400, step/86400))
            ax.broken_barh(segments, (label-.35, .7), facecolors=colors[label % len(colors)])
        # High frequency is excluded from learning, but remains visible as a distinct state.
        for left, right in result['excluded_blocks_seconds']:
            ax.axvspan(mdates.date2num(origin+pd.Timedelta(seconds=left)),
                       mdates.date2num(origin+pd.Timedelta(seconds=right)), color='#dedede', alpha=.7)
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_yticks(range(k), [f'C{i+1}' for i in range(k)])
        ax.set(ylim=(-.6, k-.4), xlabel='时间（上海时间）；色条为窗口末端前一个步长，不是完整分析窗', title=heading)
        ax.legend(handles=[Patch(color='#dedede', label='排除的高频时段'),
                           Patch(facecolor='white', edgecolor='#999999', label='未分类 / 无足够记录')],
                  loc='upper right', fontsize=8)
    names = ['representative_distributions.svg', 'transition_matrices.svg', 'class_timeline.svg']
    metadata = json.dumps(dict(result, plot_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                          ensure_ascii=False, default=str)
    for (fig, _), name, subtitle in zip(figures, names,
            ['各类间隔拟合概率密度（实际观测间隔范围；纵轴显示峰值以下6个数量级；常数为点质量）',
             '相邻窗口转移概率（按行归一化；缺窗断开；灰色表示无出向样本）',
             '不同间隔分布类别随时间的分布（类编号仅在本方案内有效）']):
        fig.suptitle(f'{title}\n{subtitle}')
        fig.savefig(output_dir/name, bbox_inches='tight', metadata={'Description': metadata})
        plt.close(fig)
    return names


def generate_mask(model, timestamps, *, seed, tolerance_seconds, max_draws=10000000):
    """Nearest timestamp within tolerance; no forced endpoints or changed timestamps.

    Returns a boolean retention mask and mapping diagnostics. Values are never read.
    Original absent timestamps are never invented. Model time and matching time
    are separate: matching errors do not accumulate into the generated schedule.
    """
    if model.get('status') != 'stable':
        raise ValueError('尚无稳定模型，拒绝生成掩码')
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError('匹配容差必须为非负有限数')
    ns, original = valid_times(timestamps)
    if len(original) != len(timestamps):
        raise ValueError('目标时间戳包含空值')
    t = (ns - ns[0]) / 1e9
    rng = np.random.default_rng(seed)
    runs = model['mode_runs']
    durations = np.array([r['duration_seconds'] for r in runs])
    cumulative = np.cumsum(durations)
    cycle = cumulative[-1]
    phase = rng.uniform(0, cycle)
    elapsed = 0.
    keep = np.zeros(len(ns), dtype=bool)
    draws = matched = 0
    while elapsed <= t[-1]:
        if draws >= max_draws:
            raise RuntimeError('超过最大采样次数；未返回不完整掩码')
        run = runs[min(np.searchsorted(cumulative, (elapsed + phase) % cycle, side='right'), len(runs)-1)]
        dist = model['distributions'][run['label']]
        if dist['family'] == 'constant':
            delta = dist['parameters'][0]
        else:
            delta = float(getattr(stats, dist['family']).rvs(*dist['parameters'], random_state=rng))
        if not np.isfinite(delta) or delta <= 0 or elapsed + delta <= elapsed:
            raise ValueError('抽样产生无效间隔')
        elapsed += delta
        draws += 1
        if elapsed > t[-1]:
            break
        pos = np.searchsorted(t, elapsed)
        choices = [i for i in (pos - 1, pos) if 0 <= i < len(t)]
        nearest = min(choices, key=lambda i: (abs(t[i] - elapsed), i))
        if abs(t[nearest] - elapsed) <= tolerance_seconds:
            keep[nearest] = True
            matched += 1
    return keep, dict(draws=draws, matched=matched, retained=int(keep.sum()),
                      duplicate_matches=matched-int(keep.sum()), seed=seed,
                      tolerance_seconds=tolerance_seconds)


def main():
    parser = argparse.ArgumentParser(description='单测点相邻记录间隔学习；仅输出Top3方案的三张SVG')
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--settings', required=True, type=Path, help='Settings字段的JSON；阈值须显式填写')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent/'figures')
    args = parser.parse_args()
    cfg = Settings(**json.loads(args.settings.read_text(encoding='utf-8-sig')))
    cfg.validate()
    frame = pd.read_csv(args.csv, usecols=['timestamp', 'value'], engine='pyarrow', parse_dates=['timestamp'])
    result = learn(frame.timestamp, frame.value, cfg)
    stat = args.csv.stat()
    result['source'] = dict(path=str(args.csv.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    result['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if result.get('top3'):
        names = plot_top3(result, args.output_dir, args.csv.stem)
        print(f"已生成 {len(names)} 张SVG，展示 {len(result['top3'])} 个方案：{args.output_dir}")
    else:
        print('无稳定且类内样本充分的方案；未生成图形。')


if __name__ == '__main__':
    main()
