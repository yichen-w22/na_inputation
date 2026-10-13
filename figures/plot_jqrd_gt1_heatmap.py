"""A single PNG heatmap: operating-conditioned coverage, quantity, correlation.

python figures/plot_jqrd_gt1_heatmap.py --compute-only
python figures/plot_jqrd_gt1_heatmap.py --render-only --sort-mode selection
Default row order follows the first sheet of the selected measurement workbook.
Use --sort-mode grouped to keep highly correlated groups adjacent; groups and
their members are ranked by unobserved counts. No original data are modified.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "jqrd_gt1_heatmap"
OLD = ROOT / "figures" / "jqrd_gt1_missingness_20261012_v2"
spec = importlib.util.spec_from_file_location("jqrd_helpers", ROOT / "figures" / "plot_jqrd_gt1_missingness.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def compute():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    previous = previous.get("historical_before_image_replacement", previous)
    paths = sorted(helpers.SOURCE.glob("*.csv"))
    before = [helpers.metadata(p) for p in paths]
    if before != previous["source_files"]:
        raise ValueError("Raw source metadata changed; audit source version before reusing labels")
    pt, pv, spans, _ = helpers.load_power(1, 5)
    start = int(pt[0] // helpers.DAY * helpers.DAY)
    end = int((pt[-1] // helpers.DAY + 1) * helpers.DAY)
    n = (end - start) // helpers.SECOND
    nd = n // 86400
    support = np.zeros(n, dtype=bool)
    for a, b in zip(*spans[1]):
        lo = max(0, int((a - start + helpers.SECOND - 1) // helpers.SECOND))
        hi = min(n, int((b - start) // helpers.SECOND) + 1)
        support[lo:hi] = True
    day_support = support.reshape(nd, 86400).sum(axis=1)
    valid_day = day_support >= 1800
    support &= np.repeat(valid_day, 86400)
    day_support = support.reshape(nd, 86400).sum(axis=1)
    hour_support = support.reshape(nd * 24, 3600).sum(axis=1)
    summary = pd.read_csv(OLD / "statistics" / "sensor_summary.csv").set_index("tag")

    def scan(path):
        observed = np.zeros(n, dtype=bool)
        rows = invalid = 0
        for chunk in pd.read_csv(path, chunksize=350000):
            t = helpers.ns_times(chunk["timestamp"])
            v = pd.to_numeric(chunk["value"], errors="coerce").to_numpy()
            finite = np.isfinite(v)
            invalid += int((~finite).sum())
            idx = (t + helpers.SECOND // 2 - start) // helpers.SECOND
            actual_run = helpers.containing_span(t, *spans[1]) >= 0
            keep = finite & actual_run & (idx >= 0) & (idx < n)
            pos = idx[keep].astype(np.int64)
            pos = pos[support[pos]]
            observed[pos] = True  # multiple events in the same second count once
            rows += len(chunk)
        if rows != int(summary.loc[path.stem, "rows"]):
            raise ValueError("Source row count changed")
        observed_day = observed.reshape(nd, 86400).sum(axis=1)
        observed_hour = observed.reshape(nd * 24, 3600).sum(axis=1)
        assert np.all(observed_day <= day_support)
        return {"tag": path.stem, "name": str(summary.loc[path.stem, "label"]),
                "rows": rows, "invalid": invalid, "day": observed_day, "hour": observed_hour}

    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(scan, p): p for p in paths}
        for f in as_completed(pending):
            r = f.result(); results.append(r)
            print(f"COVERAGE {len(results)}/35 {r['tag']}", flush=True)
    results.sort(key=lambda r: r["tag"])
    if [helpers.metadata(p) for p in paths] != before:
        raise ValueError("Raw data changed during scan")
    np.savez_compressed(OUT / "coverage_cache.npz", day_support=day_support, hour_support=hour_support,
                        observed_day=np.array([r["day"] for r in results]),
                        observed_hour=np.array([r["hour"] for r in results]))
    info = {"source_files": before, "source_records": sum(r["rows"] for r in results),
            "grid_start_ns": start, "grid_end_ns": end, "timezone": "Asia/Shanghai",
            "sensors": [{k: r[k] for k in ("tag", "name", "rows", "invalid")} for r in results],
            "reference_grid_seconds": 1, "mapping": "nearest nominal second, half-second ties upward; multiple events counted once; both actual event time and query second require running support",
            "state_rule": "power >1 MW; connect same-state power endpoints only when gap <=5 s",
            "day_rule": "at least 1800 supported grid seconds per day; remaining days excluded",
            "hour_correlation_rule": "at least 1800 supported grid seconds per hour, identical eligible hours for all sensors",
            "quantity_definition": "unobserved positions on a common supported-running 1-second grid; includes ordinary low-frequency sampling, not verified acquisition failures",
            "source_integrity": "path/bytes/mtime unchanged, plus exact source row-count match; not full raw content hashing",
            "state_csv_sha256": previous["state_csv_sha256"]}
    (OUT / "coverage_manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COVERAGE_COMPUTATION_COMPLETE", flush=True)


def display_names(info):
    replacements = {
        "H2": "氢气组分", "N2": "氮气组分", "CO2": "二氧化碳组分", "CO": "一氧化碳组分",
        "CH4": "甲烷组分", "C2H6": "乙烷组分", "C3H8": "丙烷组分",
        "iC4H10": "异丁烷组分", "nC4H10": "正丁烷组分",
        "iC5H12": "异戊烷组分", "nC5H12": "正戊烷组分",
        "燃料氧组分通道（GARD O2 / 其他厂O2+Ar）": "燃料氧组分通道（口径待核）",
        "重烃组分通道（JXRD原名乙烷+）": "重烃组分通道（口径待核）",
    }
    names = [replacements.get(r["name"], r["name"]) for r in info["sensors"]]
    # Human-readable disambiguation for repeated names; codes remain in tables only.
    for name in set(names):
        positions = [i for i, n in enumerate(names) if n == name]
        if len(positions) > 1:
            for number, i in enumerate(positions, 1):
                names[i] = f"{name}（测点{number}）"
    return names


def get_order(missing, corr, mode):
    distance = 1 - np.nan_to_num(corr, nan=0.0)
    distance = np.clip((distance + distance.T) / 2, 0, 2)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance), method="complete", optimal_ordering=True)
    leaves = leaves_list(tree)
    leaf_rank = {int(i): k for k, i in enumerate(leaves)}
    groups = fcluster(tree, t=.30, criterion="distance")
    if mode == "quantity":
        order = sorted(range(len(missing)), key=lambda i: (-int(missing[i]), leaf_rank[i]))
        assert np.all(np.diff(missing[order]) <= 0)
    else:
        group_ids = sorted(set(groups), key=lambda g: -float(missing[groups == g].mean()))
        order = []
        for group in group_ids:
            members = np.flatnonzero(groups == group).tolist()
            members.sort(key=lambda i: (-int(missing[i]), leaf_rank[i]))
            order.extend(members)
    return np.array(order), groups


def selection_order(info):
    workbook = ROOT / "raw_data" / "jqrd_gt1" / "jqrd_gt1_选取点表.xlsx"
    wb = helpers.load_workbook(workbook, read_only=True, data_only=True)
    try:
        sheet = wb.worksheets[0]
        sheet_name = sheet.title
        tags = [str(row[0]).strip() for row in sheet.iter_rows(min_row=2, values_only=True) if row[0]]
    finally:
        wb.close()
    positions = {sensor["tag"]: i for i, sensor in enumerate(info["sensors"])}
    if len(tags) != len(set(tags)) or set(tags) != set(positions):
        raise ValueError("Selected measurement workbook must match the 35 cached sensors exactly")
    provenance = {"path": str(workbook.relative_to(ROOT)), "sheet": sheet_name,
                  "column": "A", "first_data_row": 2,
                  "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest()}
    return np.array([positions[tag] for tag in tags]), provenance


def render(mode):
    helpers.setup_style()
    info = json.loads((OUT / "coverage_manifest.json").read_text(encoding="utf-8"))
    if [helpers.metadata(p) for p in sorted(helpers.SOURCE.glob("*.csv"))] != info["source_files"]:
        raise ValueError("Raw source version changed")
    with np.load(OUT / "coverage_cache.npz") as cache:
        day_support = cache["day_support"]
        hour_support = cache["hour_support"]
        observed_day = cache["observed_day"]
        observed_hour = cache["observed_hour"]
    total = int(day_support.sum())
    missing = total - observed_day.sum(axis=1)
    fraction = np.divide(day_support - observed_day, day_support,
                         out=np.full(observed_day.shape, np.nan), where=day_support > 0)
    valid_hours = hour_support >= 1800
    hourly_fraction = (hour_support[valid_hours] - observed_hour[:, valid_hours]) / hour_support[valid_hours]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(hourly_fraction)
    np.fill_diagonal(corr, 1)
    order, groups = get_order(missing, corr, "quantity" if mode == "selection" else mode)
    ordering_source = None
    if mode == "selection":
        order, ordering_source = selection_order(info)
    names = display_names(info)
    labels = [f"{k+1:02d}  {names[i]}" for k, i in enumerate(order)]
    data = fraction[order]
    ordered_corr = corr[np.ix_(order, order)]
    dates = pd.to_datetime(info["grid_start_ns"] + np.arange(len(day_support)) * helpers.DAY)
    fig = plt.figure(figsize=(26, 14), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[4.2, .58, 1.7], left=.245, right=.985,
                          bottom=.18, top=.865, wspace=.065)
    ax = fig.add_subplot(gs[0, 0]); quantity = fig.add_subplot(gs[0, 1]); relation = fig.add_subplot(gs[0, 2])
    extent = [mdates.date2num(dates[0]), mdates.date2num(dates[-1] + pd.Timedelta(days=1)), 34.5, -.5]
    boundaries = [0, .10, .25, .50, .75, .90, .95, .98, .99, .995, .999, 1.00000001]
    cmap = plt.get_cmap("YlOrRd", len(boundaries)-1).copy(); cmap.set_bad("#d3dae2")
    im = ax.imshow(np.ma.masked_invalid(data), extent=extent, aspect="auto", interpolation="nearest",
                   cmap=cmap, norm=BoundaryNorm(boundaries, cmap.N))
    ax.set_yticks(np.arange(35), labels, fontsize=10)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.tick_params(axis="x", rotation=25)
    ax.set_xlabel("日期（日聚合）", labelpad=8)
    ax.set_title("A  有运行证据时段的未观测比例", fontsize=14, pad=15)
    quantity.imshow((missing[order] / total)[:, None], cmap="Oranges", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    quantity.set_xticks([]); quantity.set_yticks([])
    quantity.set_title("未观测秒数\n（百万）", fontsize=11, pad=11)
    for k, i in enumerate(order):
        quantity.text(0, k, f"{missing[i]/1e6:.3f}", ha="center", va="center", fontsize=9)
    corr_cmap = plt.get_cmap("RdBu_r").copy(); corr_cmap.set_bad("#d3dae2")
    ri = relation.imshow(np.ma.masked_invalid(ordered_corr), vmin=-1, vmax=1, cmap=corr_cmap,
                         aspect="auto", interpolation="nearest")
    relation.set_yticks([]); relation.set_xticks(np.arange(35), [str(i+1) for i in range(35)], fontsize=6, rotation=90)
    relation.set_xlabel("测点行号（与左图一致）", fontsize=10)
    relation.set_title("B  每小时未观测模式相关性", fontsize=12, pad=15)
    if mode == "grouped":
        group_order = groups[order]
        for y in np.flatnonzero(np.diff(group_order) != 0) + .5:
            for target in (ax, quantity, relation):
                target.axhline(y, color="#34495e", lw=.75)
    colour_ax = fig.add_axes([.245, .103, .435, .018])
    cb = fig.colorbar(im, cax=colour_ax, orientation="horizontal", spacing="uniform")
    ticks = boundaries[:-1] + [1]
    cb.set_ticks(ticks, labels=["0", "10", "25", "50", "75", "90", "95", "98", "99", "99.5", "99.9", "100"])
    cb.ax.tick_params(labelsize=8); cb.set_label("未观测比例 / %（分段色阶，突出高缺失区间）", fontsize=10)
    corr_ax = fig.add_axes([.77, .103, .215, .018])
    cb2 = fig.colorbar(ri, cax=corr_ax, orientation="horizontal", ticks=[-1, -.5, 0, .5, 1])
    cb2.set_label("Pearson 相关系数（关联，不代表因果）", fontsize=10)
    sort_text = {"selection": "按选取点表原始顺序排列",
                 "quantity": "数量严格降序；数量相同时按相关性排序",
                 "grouped": "相关性先分组；组间按平均未观测数、组内按未观测数降序"}[mode]
    fig.suptitle("JQRD GT1 | 运行期间测量值未观测热力图", fontsize=22, y=.962)
    fig.text(.245, .91, sort_text + f"  ·  35 个测点  ·  共同运行支持网格 {total:,} 个秒点", fontsize=11)
    fig.text(.245, .035, "灰色：每日运行支持不足30分钟，不填为停机或缺失。1秒网格未观测包括正常低频采样，不能直接解释为采集故障。\n"
             "运行支持：相邻出力记录均>1 MW且间隔≤5秒；原始记录按最近秒映射（容差≤0.5秒），同秒多条只计一次。相关性使用共同有效小时（支持≥30分钟）。", fontsize=10)
    image_path = OUT / "jqrd_gt1_missingness_heatmap.png"
    fig.savefig(image_path, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    rows = []
    for k, i in enumerate(order):
        rows.append({"rank": k+1, "tag": info["sensors"][i]["tag"], "name": names[i], "correlation_group": int(groups[i]),
                     "common_running_seconds": total, "observed_seconds": int(total-missing[i]),
                     "unobserved_seconds": int(missing[i]), "unobserved_fraction": float(missing[i]/total)})
    pd.DataFrame(rows).to_csv(OUT / "row_order.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ordered_corr, index=[names[i] for i in order], columns=[names[i] for i in order]).to_csv(OUT / "missingness_correlation.csv", encoding="utf-8-sig")
    daily = pd.DataFrame(data.T, index=dates, columns=[names[i] for i in order])
    daily.index.name = "date"; daily.to_csv(OUT / "daily_unobserved_fraction.csv", encoding="utf-8-sig")
    manifest = dict(info, sort_mode=mode, ordering_description=sort_text, output_png=str(image_path.relative_to(ROOT)),
                    grouping="complete linkage of 1-Pearson correlation, cut at 0.30; all within-group finite pairwise correlations >=0.70",
                    valid_correlation_hours=int(valid_hours.sum()), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    ordering_source=ordering_source, helper_script_sha256=hashlib.sha256(Path(helpers.__file__).read_bytes()).hexdigest())
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(image_path), "sort_mode": mode, "common_running_seconds": total,
                      "correlation_hours": int(valid_hours.sum()), "groups": len(set(groups))}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--sort-mode", choices=["selection", "quantity", "grouped"], default="selection")
    args = parser.parse_args()
    if args.compute_only and args.render_only:
        raise ValueError("Choose only one stage")
    if not args.render_only:
        compute()
    if not args.compute_only:
        render(args.sort_mode)


if __name__ == "__main__":
    main()
