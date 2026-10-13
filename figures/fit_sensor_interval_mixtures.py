"""单测点、逐窗口固定间隔点质量＋对数正态混合拟合；唯一产物为SVG。

复用现有CSV流式读取规则；每10个本地日的0/1日为留出检查，2/3日为
成分数选择，其余日为拟合。分区按间隔左端日期决定，三个集合不共享间隔。
对数间隔采用固定种子的均匀优先级蓄水抽样，分箱统计使用全部有效间隔。
点位置只由拟合日期确定，后续分区复用这些位置并重新估计权重。
图中曲线是分箱概率的保形展示，绝非概率密度；参数、统计、检查内嵌SVG。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.special import ndtr
import sklearn
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

import plot_sensor_interval_windows as base

DAY = 86400 * base.SECOND
DEFAULTS = dict(unit="jqrd_gt1", sensor="JQRD_11EKG60FF001OUT", window="2mo",
                max_components=6, sample_cap=50000, seed=20261013,
                chunk_size=350000, n_init=3, reg_covar=1e-4,
                max_atoms=8, min_atom_fraction=.02, min_atom_samples=100)


class Reservoir:
    def __init__(self, cap):
        self.cap = cap
        self.keys = np.empty(0)
        self.values = np.empty(0)

    def add(self, values, keys):
        if len(self.keys) == self.cap:
            keep = keys > self.keys.min()
            values, keys = values[keep], keys[keep]
        if not len(keys):
            return
        keys = np.r_[self.keys, keys]
        values = np.r_[self.values, values]
        if len(keys) > self.cap:
            selected = np.argpartition(keys, -self.cap)[-self.cap:]
            keys, values = keys[selected], values[selected]
        self.keys, self.values = keys, values


def collect(path, starts, stops, edges, args):
    n = len(starts)
    hist = np.zeros((n, 3, len(edges)+1), dtype=np.int64)
    samples = [[Reservoir(args.sample_cap) for _ in range(3)] for _ in range(n)]
    rng = np.random.default_rng(args.seed)
    previous = None
    audit = dict(rows=0, invalid_values=0, duplicates=0, eligible_intervals=0)
    for timestamps, values in base.chunks(path, args.chunk_size):
        audit["rows"] += len(timestamps)
        valid = np.isfinite(values)
        audit["invalid_values"] += int((~valid).sum())
        t = timestamps[valid]
        if not len(t):
            continue
        if previous is not None:
            t = np.r_[previous, t]
        left, right = t[:-1], t[1:]
        keep = right > left
        audit["duplicates"] += int((~keep).sum())
        left, right = left[keep], right[keep]
        previous = int(t[-1])
        indices = np.searchsorted(starts, left, side="right")-1
        keep = (indices >= 0) & (indices < n)
        keep &= left < stops[np.clip(indices, 0, n-1)]
        left, right, indices = left[keep], right[keep], indices[keep]
        delta = (right-left)/base.SECOND
        logs = np.log(delta)
        keys = rng.random(len(delta))
        categories = np.searchsorted(edges, delta, side="right")
        categories[delta == edges[-1]] = len(edges)-1
        audit["eligible_intervals"] += len(delta)
        for i in np.unique(indices):
            chosen = indices == i
            day = left[chosen]//DAY - starts[i]//DAY
            slot = day % 10
            # 0: train; 1: selection; 2: held-out check.
            split = np.where(slot < 2, 2, np.where(slot < 4, 1, 0))
            for part in range(3):
                hist[i, part] += np.bincount(categories[chosen][split == part],
                                            minlength=len(edges)+1)
            v, k = logs[chosen], keys[chosen]
            samples[i][0].add(v[split == 0], k[split == 0])
            samples[i][1].add(v[split != 2], k[split != 2])
            samples[i][2].add(v, k)
    return hist, samples, audit


def fit(values, k, args, seed):
    if len(values) < max(50, 10*k):
        raise ValueError("窗口分区样本不足，无法可靠拟合。")
    g = GaussianMixture(n_components=k, covariance_type="full", n_init=args.n_init,
                        max_iter=400, tol=1e-5, reg_covar=args.reg_covar,
                        random_state=seed, init_params="k-means++")
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        g.fit(values.reshape(-1, 1))
    order = np.argsort(g.means_.ravel())
    mu = g.means_.ravel()[order]
    sigma = np.sqrt(g.covariances_.ravel()[order])
    return dict(weights=g.weights_[order].tolist(), mu_log_seconds=mu.tolist(),
                sigma_log_seconds=sigma.tolist(), component_medians_seconds=np.exp(mu).tolist(),
                converged=bool(g.converged_), iterations=int(g.n_iter_),
                warnings=[str(w.message) for w in messages])


def discover_atoms(values, args):
    locations, counts = np.unique(values, return_counts=True)
    keep = (counts >= args.min_atom_samples) & (counts/len(values) >= args.min_atom_fraction)
    locations, counts = locations[keep], counts[keep]
    order = np.argsort(-counts, kind="stable")[:args.max_atoms]
    return np.sort(locations[order])


def fit_hybrid(values, k, atoms, args, seed):
    matches = values[:, None] == atoms[None, :]
    weights = matches.mean(axis=0)
    residual = values[~matches.any(axis=1)]
    continuous_weight = len(residual)/len(values)
    if not len(residual):
        if k != 0:
            raise ValueError("纯点质量样本无需连续成分。")
        model = dict(weights=[], mu_log_seconds=[], sigma_log_seconds=[],
                     component_medians_seconds=[], converged=True, iterations=0, warnings=[])
    else:
        if k < 1:
            raise ValueError("剩余样本需要至少一个连续成分。")
        model = fit(residual, k, args, seed)
        model["weights"] = (np.array(model["weights"])*continuous_weight).tolist()
    model.update(atom_log_seconds=atoms.tolist(),
                 atom_locations_seconds=(np.rint(np.exp(atoms)*base.SECOND)/base.SECOND).tolist(),
                 atom_weights=weights.tolist(), continuous_weight=continuous_weight,
                 residual_sample_count=len(residual))
    np.testing.assert_allclose(sum(model["weights"])+sum(model["atom_weights"]), 1, atol=1e-12)
    return model


def normal_interval(lo, hi):
    # 使用右尾对称式，避免两个接近1的CDF相减。
    return np.maximum(0, np.where(lo > 0, ndtr(-lo)-ndtr(-hi), ndtr(hi)-ndtr(lo)))


def mass(model, edges):
    weights = np.array(model["weights"])
    mu = np.array(model["mu_log_seconds"])
    sigma = np.array(model["sigma_log_seconds"])
    bounds = np.r_[-np.inf, np.log(edges), np.inf]
    z = (bounds[:, None]-mu)/sigma
    result = normal_interval(z[:-1], z[1:]) @ weights
    locations = np.array(model.get("atom_locations_seconds", []))
    categories = np.searchsorted(edges, locations, side="right")
    categories[locations == edges[-1]] = len(edges)-1
    np.add.at(result, categories, model.get("atom_weights", []))
    np.testing.assert_allclose(result.sum(), 1, atol=1e-12)
    return result


def metrics(counts, q):
    if not counts.sum():
        raise ValueError("验证/检查分区没有有效间隔。")
    p = counts/counts.sum()
    return dict(total_variation=float(.5*np.abs(p-q).sum()),
                binned_cdf_distance=float(np.max(np.abs(np.cumsum(p-q)))),
                cross_entropy_nats=float(-np.dot(p, np.log(np.maximum(q, 1e-300)))))


def moving_bin_mass(model, x, width):
    if model.get("atom_weights"):
        raise ValueError("含点质量时应绘制实际分箱积分概率，不将点质量平滑成密度。")
    mu = np.array(model["mu_log_seconds"])
    sigma = np.array(model["sigma_log_seconds"])
    centre = np.log(x)[:, None]
    lo, hi = (centre-width/2-mu)/sigma, (centre+width/2-mu)/sigma
    return normal_interval(lo, hi) @ np.array(model["weights"])


def raw_check(pools, model):
    # 全部池与development池共用优先级键；全部池中非development的键
    # 正好属于留出日期。development池容量相同，故不会遗漏全部池中的开发样本。
    assert len(np.unique(pools[2].keys)) == len(pools[2].keys)
    values = pools[2].values[~np.isin(pools[2].keys, pools[1].keys)]
    if not len(values):
        raise ValueError("留出原始间隔抽样为空，无法检查细粒度CDF。")
    values = np.sort(values)
    mu, sigma = np.array(model["mu_log_seconds"]), np.array(model["sigma_log_seconds"])
    atoms = np.array(model.get("atom_log_seconds", []))
    weights = np.array(model.get("atom_weights", []))
    query = np.unique(np.r_[values, atoms])
    continuous = ndtr((query[:, None]-mu)/sigma) @ np.array(model["weights"])
    cdf_right = continuous + (atoms[None, :] <= query[:, None]) @ weights
    cdf_left = continuous + (atoms[None, :] < query[:, None]) @ weights
    n = len(values)
    empirical_right = np.searchsorted(values, query, side="right")/n
    empirical_left = np.searchsorted(values, query, side="left")/n
    distance = max(float(np.max(np.abs(empirical_right-cdf_right))),
                   float(np.max(np.abs(empirical_left-cdf_left))))
    return dict(raw_cdf_distance=distance, sampled_heldout_intervals=n,
                exact_one_second_fraction=float(np.mean(values == 0)),
                learned_atom_fraction=float(np.mean(np.isin(values, atoms))))


def render(output, edges, hist, results, metadata):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "svg.fonttype": "path"})
    rows = (len(results)+1)//2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 3.8*rows), squeeze=False)
    mid = np.sqrt(edges[:-1]*edges[1:])
    for ax, result, counts in zip(axes.flat, results, hist.sum(axis=1)):
        p = counts[1:-1]/counts.sum()
        bx, by = base.smooth_probability(edges, p, "log", 2400)
        q = mass(result["full_window_model"], edges)[1:-1]
        x, fitted = base.smooth_probability(edges, q, "log", 2400)
        ax.plot(bx, by, color="#374151", lw=1.6, label="实际分箱概率")
        ax.plot(mid, p, ".", color="#374151", ms=3)
        ax.plot(x, fitted, color="#dc2626", lw=1.7, label="点质量＋对数正态混合")
        ax.plot(mid, q, ".", color="#dc2626", ms=3)
        ax.set_xscale("log")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, min(1.05, max(.05, float(max(p.max(), fitted.max()))*1.12)))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("相邻有效测量的时间间隔 / 秒")
        ax.set_ylabel("窗口内出现概率")
        ax.set_title(result["start"][:10]+" 至 "+result["end"][:10]+
                     f"   M={len(result['full_window_model']['atom_weights'])}, K={result['components']}")
        ax.grid(which="major", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)
    title = metadata["sensor"]+"  "+metadata["name"]
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=2.0)
    fig.savefig(output, format="svg", metadata={"Title": title,
                "Description": json.dumps(metadata, ensure_ascii=False, allow_nan=False)})
    plt.close(fig)


def self_check():
    rng = np.random.default_rng(120)
    labels = rng.random(12000) > .65
    values = rng.normal(np.where(labels, np.log(600), 0), .15)
    args = argparse.Namespace(n_init=3, reg_covar=1e-4,
                              max_atoms=8, min_atom_fraction=.02, min_atom_samples=100)
    edges = np.geomspace(.001, 604800, 81)
    a, b = fit(values, 1, args, 1), fit(values, 2, args, 1)
    counts = np.histogram(np.exp(values), np.r_[0, edges, np.inf])[0]
    assert metrics(counts, mass(b, edges))["total_variation"] < .04
    assert metrics(counts, mass(a, edges))["total_variation"] > .3
    np.testing.assert_allclose(b["component_medians_seconds"], [1, 600], rtol=.03)
    keys = rng.random(1000)
    first, second = Reservoir(100), Reservoir(100)
    first.add(np.arange(1000), keys)
    for i in range(0, 1000, 37):
        second.add(np.arange(1000)[i:i+37], keys[i:i+37])
    np.testing.assert_array_equal(np.sort(first.values), np.sort(second.values))
    combined = np.r_[np.zeros(12000), rng.normal(np.log(10), .2, 6000)]
    atoms = discover_atoms(combined, args)
    np.testing.assert_array_equal(atoms, [0.])
    hybrid = fit_hybrid(combined, 1, atoms, args, 1)
    np.testing.assert_allclose(hybrid["atom_weights"], [2/3])
    histogram = np.histogram(np.exp(combined), np.r_[0, edges, np.inf])[0]
    assert metrics(histogram, mass(hybrid, edges))["total_variation"] < .02
    empty, pool = Reservoir(100), Reservoir(100)
    pool.keys, pool.values = np.arange(100, dtype=float), np.zeros(100)
    pure = fit_hybrid(pool.values, 0, np.array([0.]), args, 1)
    assert raw_check([empty, empty, pool], pure)["raw_cdf_distance"] == 0
    assert raw_check([empty, empty, pool], dict(weights=[1.], mu_log_seconds=[0.],
                       sigma_log_seconds=[.1]))["raw_cdf_distance"] == .5
    edge_atom = fit_hybrid(np.full(100, np.log(10)), 0, np.array([np.log(10)]), args, 1)
    np.testing.assert_array_equal(mass(edge_atom, np.array([1., 10.])), [0., 1., 0.])
    print("SELF_CHECK_PASSED: continuous/atomic mixture, CDF jumps, boundary bin, reservoir")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("unit", "sensor", "window"):
        p.add_argument("--"+key, default=DEFAULTS[key])
    for key in ("max_components", "sample_cap", "seed", "chunk_size", "n_init", "max_atoms", "min_atom_samples"):
        p.add_argument("--"+key.replace("_", "-"), type=int, default=DEFAULTS[key])
    p.add_argument("--reg-covar", type=float, default=DEFAULTS["reg_covar"])
    p.add_argument("--min-atom-fraction", type=float, default=DEFAULTS["min_atom_fraction"])
    p.add_argument("--output", type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        with threadpool_limits(limits=1):
            self_check()
        return
    if not 1 <= args.max_components <= 12 or min(args.sample_cap, args.chunk_size, args.n_init) < 1 or args.reg_covar <= 0:
        raise ValueError("成分数须1至12，抽样/分块/初始化次数和正则须为正。")
    if args.max_atoms < 0 or args.min_atom_samples < 1 or not 0 < args.min_atom_fraction <= 1:
        raise ValueError("点质量数量须非负，次数须为正，比例须在(0,1]。")
    source = base.ROOT/"raw_data"/args.unit/"csv"/(args.sensor+".csv")
    reference = base.ROOT/"figures"/"interval_scattering"/(args.sensor+"_interval_probability.svg")
    old = json.loads(ET.parse(reference).getroot().find(".//{http://purl.org/dc/elements/1.1/}description").text)
    if old["parameters"]["window"] != args.window or old["parameters"]["step"] != args.window or old["parameters"]["state_mode"] != "all":
        raise ValueError("先生成同一测点、全时段、相同窗口及步长的原始概率SVG，供独立计数核对。")
    output = (args.output or reference.with_name(args.sensor+"_lognormal_mixture.svg")).resolve()
    raw = (base.ROOT/"raw_data").resolve()
    if output.suffix.lower() != ".svg" or raw == output or raw in output.parents or output == reference.resolve():
        raise ValueError("输出须为raw_data之外的独立SVG，不覆盖原始分布图。")
    if output.exists() and not args.overwrite:
        raise FileExistsError("输出已存在；使用--overwrite。")
    previous_continuous = None
    if output.exists():
        previous = json.loads(ET.parse(output).getroot().find(".//{http://purl.org/dc/elements/1.1/}description").text)
        previous_continuous = (previous.get("previous_continuous_model")
                               if previous.get("model_type") == "point_mass_plus_lognormal_mixture" else previous)
    first, last = base.source_extent(source)
    starts, stops = base.make_windows(first, last+1, args.window, args.window, 100)
    edges = np.array(old["bin_edges_seconds"])
    before = base.metadata(source)
    print(f"扫描完整CSV：{args.sensor}；{len(starts)}窗；每窗尝试1至{args.max_components}个成分。", flush=True)
    hist, samples, audit = collect(source, starts, stops, edges, args)
    assert before == base.metadata(source)
    assert before == old["source_files"][0]
    expected = np.column_stack([old["underflow_counts"], old["all_window_bin_counts"], old["overflow_counts"]])
    np.testing.assert_array_equal(hist.sum(axis=1), expected)
    assert audit["eligible_intervals"] == int(hist.sum())
    results = []
    with threadpool_limits(limits=1):
        for i, pools in enumerate(samples):
            candidates = []
            atoms = discover_atoms(pools[0].values, args)
            residual_n = int((~np.isin(pools[0].values, atoms)).sum())
            components = range(1, args.max_components+1) if residual_n else [0]
            for k in components:
                model = fit_hybrid(pools[0].values, k, atoms, args, args.seed+100*i+k)
                score = metrics(hist[i, 1], mass(model, edges))
                candidates.append(dict(components=k, model=model, selection_metrics=score))
            valid_candidates = [c for c in candidates if c["model"]["converged"]]
            if not valid_candidates:
                raise RuntimeError("所有候选拟合均未收敛。")
            chosen = min(valid_candidates, key=lambda c: c["selection_metrics"]["cross_entropy_nats"])
            k = chosen["components"]
            development = fit_hybrid(pools[1].values, k, atoms, args, args.seed+100*i+50)
            baseline = fit(pools[1].values, 1, args, args.seed+100*i+50)
            full = fit_hybrid(pools[2].values, k, atoms, args, args.seed+100*i+60)
            if not development["converged"] or not full["converged"]:
                raise RuntimeError("最终模型未收敛，不能将此窗口记作拟合完成。")
            result = dict(start=str(pd.Timestamp(starts[i])), end=str(pd.Timestamp(stops[i])),
                          components=k, partition_counts=hist[i].sum(axis=1).tolist(),
                          sample_counts=[len(pool.values) for pool in pools], candidates=candidates,
                          full_window_model=full, development_model=development,
                          full_window_metrics=metrics(hist[i].sum(axis=0), mass(full, edges)),
                          heldout_metrics=metrics(hist[i, 2], mass(development, edges)),
                          heldout_raw_check=raw_check(pools, development),
                          single_lognormal_heldout_metrics=metrics(hist[i, 2], mass(baseline, edges)))
            results.append(result)
            print(json.dumps(dict(window=i+1, K=k, medians=full["component_medians_seconds"],
                  atoms=full["atom_locations_seconds"], atom_weights=full["atom_weights"],
                  full_TV=result["full_window_metrics"]["total_variation"],
                  heldout_TV=result["heldout_metrics"]["total_variation"],
                  heldout_raw_CDF=result["heldout_raw_check"]["raw_cdf_distance"],
                  baseline_TV=result["single_lognormal_heldout_metrics"]["total_variation"])), flush=True)
    metadata = dict(sensor=args.sensor, name=old["name"], source=before, source_audit=audit,
                    model_type="point_mass_plus_lognormal_mixture",
                    previous_continuous_model=previous_continuous,
                    window=args.window, bin_edges_seconds=edges.tolist(), partition_histograms=hist.tolist(),
                    results=results, parameters={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    reference_svg=base.metadata(reference), reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    base_script_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                    environment=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
                    definition="Positive adjacent finite-valued records; left timestamp anchors window. Full-source bin probabilities include underflow/overflow in denominator.",
                    fit_rule="P(dt)=sum(atom_weights*Dirac(atom_locations)) + sum(unconditional_continuous_weights*LogNormal(mu,sigma)). Atom locations discovered ONLY on training-day reservoir: exact repeats exceeding min_atom_fraction and min_atom_samples; at most max_atoms. Locations frozen for development/full refits; weights re-estimated in each respective reservoir. Residual Gaussian mixture MLE on natural-log seconds. Sample cap applies separately to train/development/full pools; reg_covar sets minimum log variance. Frequent exact intervals are candidate periods, not confirmed acquisition/fault mechanisms.",
                    split_rule="Non-overlapping local calendar days within each window: day_offset % 10 in [0,1] held-out check, [2,3] selection, [4..9] training. K chosen on selection only; development=training+selection for held-out scores. Final full-window model uses all periods solely for descriptive plotting.",
                    plot_rule="Both observed and fitted curves use identical PCHIP display of original bin probabilities. Fitted bins integrate the continuous mixture and add atom masses using the original bin endpoint convention. Markers identify original bin centres. Interpolated display does not replace the mathematical atoms with a continuous density.",
                    limits="Same exploratory held-out days have already informed the change from a continuous to an atomic model; comparisons are diagnostic, not untouched independent test evidence. Events remain temporally dependent. Raw sampled CDF checks compare BOTH left/right limits at observation and atom locations, without IID p-values. Rare tails, serial missingness and fault causes are not established; exact repeats may reflect timestamp quantization. Thresholds and component cap have not been tuned on held-out scores.")
    output.parent.mkdir(parents=True, exist_ok=True)
    render(output, edges, hist, results, metadata)
    print(f"完成：{output}；仅生成这张SVG，参数与检查结果内嵌。")


if __name__ == "__main__":
    main()
