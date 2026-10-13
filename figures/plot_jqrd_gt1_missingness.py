"""Historical numerical diagnostics and helpers for JQRD GT1.

The command-line entry now redirects to the heatmap-only script:
  python figures/plot_jqrd_gt1_heatmap.py --render-only --sort-mode quantity

Original CSVs and the existing operating-state CSV are read only. Cadences and
record-gap events are exploratory proxies, not verified historian outage labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties, fontManager
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "raw_data" / "jqrd_gt1" / "csv"
POWER_TAG = "JQRD_11CJA01CE901XQ01"
SECOND = 1_000_000_000
DAY = 86400 * SECOND
HOUR = 3600 * SECOND
EDGES = np.geomspace(1e-4, 90 * 86400, 260)
EDGES = np.r_[0.0, EDGES, np.inf]
CENTRES = np.sqrt(np.maximum(EDGES[:-1], 1e-5) * np.minimum(EDGES[1:], 180 * 86400))
CHUNK = 350_000


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metadata(path):
    s = path.stat()
    return {"path": str(path.relative_to(ROOT)), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def ns_times(column):
    t = pd.to_datetime(column, format="mixed", errors="coerce")
    if t.isna().any():
        raise ValueError("Unparseable timestamp; cannot silently assign an operating state")
    return t.astype("int64").to_numpy()


def supported_spans(t, v, threshold, max_gap):
    """Interpolate state only between close, same-state power endpoints."""
    state = v > threshold
    dt = np.diff(t) / SECOND
    valid = (dt > 0) & (dt <= max_gap) & (state[:-1] == state[1:])
    spans = {}
    for label in (0, 1):
        keep = valid & (state[:-1] == bool(label))
        starts = np.flatnonzero(keep & ~np.r_[False, keep[:-1]])
        ends = np.flatnonzero(keep & ~np.r_[keep[1:], False]) + 1
        spans[label] = (t[starts], t[ends])
    return spans, valid


def containing_span(t, starts, ends):
    if not len(starts):
        return np.full(len(t), -1, dtype=np.int64)
    idx = np.searchsorted(starts, t, side="right") - 1
    clipped = np.clip(idx, 0, len(ends) - 1)
    return np.where((idx >= 0) & (t <= ends[clipped]), idx, -1)


def exposure_bins(starts, ends, origin, width, n):
    """Exact interval overlap duration; boundary intervals are split, not reset."""
    out = np.zeros(n, dtype=float)
    for a, b in zip(starts, ends):
        first = max(0, int((a - origin) // width))
        last = min(n - 1, int((b - 1 - origin) // width))
        if last < first:
            continue
        if first == last:
            out[first] += (b - a) / SECOND
        else:
            out[first] += (origin + (first + 1) * width - a) / SECOND
            out[last] += (b - (origin + last * width)) / SECOND
            if last > first + 1:
                out[first + 1:last] += width / SECOND
    return out


def check_logic():
    t = np.array([0, 1, 2, 20, 21], dtype=np.int64) * SECOND
    v = np.array([100, 110, 100, 0, 0], dtype=float)
    spans, _ = supported_spans(t, v, 1, 5)
    assert spans[1][0].tolist() == [0] and spans[1][1].tolist() == [2 * SECOND]
    q = np.array([1, 10, 20.5]) * SECOND
    assert containing_span(q, *spans[1]).tolist() == [0, -1, -1]
    assert containing_span(q, *spans[0]).tolist() == [-1, -1, 0]
    assert exposure_bins(np.array([0]), np.array([2 * HOUR]), 0, HOUR, 3).tolist() == [3600, 3600, 0]
    assert 3 - 1 == 2  # candidate hidden duration for dt=3 s, inferred cadence=1 s
    starts = np.array([1, 4]) * SECOND
    ends = np.array([3, 6]) * SECOND
    wait = waiting_times(starts, ends, np.array([0, 0]), None, None)
    assert wait.tolist() == [1.0]  # one retained slot between consecutive gap events
    assert waiting_times(starts[1:], ends[1:], np.array([0]), 0, 3*SECOND).tolist() == [1.0]
    assert waiting_times(starts[1:], ends[1:], np.array([1]), 0, 3*SECOND).size == 0


def waiting_times(starts, ends, keys, previous_key, previous_end):
    same = keys[1:] == keys[:-1]
    durations = (starts[1:][same] - ends[:-1][same]) / SECOND
    if len(keys) and previous_key == keys[0]:
        durations = np.r_[(starts[0] - previous_end) / SECOND, durations]
    if np.any(durations <= 0):
        raise ValueError("Candidate gaps overlap; waiting-time definition is inconsistent")
    return durations


def histogram_quantile(hist, quantile):
    total = hist.sum(axis=-1)
    cdf = np.cumsum(hist, axis=-1)
    idx = np.argmax(cdf >= total[..., None] * quantile, axis=-1)
    return np.where(total > 0, CENTRES[idx], np.nan)


def load_power(threshold, max_gap):
    ts, values = [], []
    for c in pd.read_csv(SOURCE / (POWER_TAG + ".csv"), chunksize=CHUNK):
        t = ns_times(c["timestamp"])
        v = pd.to_numeric(c["value"], errors="coerce").to_numpy()
        if not np.isfinite(v).all():
            raise ValueError("Non-finite power values must remain unknown")
        ts.append(t); values.append(v)
    t = np.concatenate(ts); v = np.concatenate(values)
    if np.any(np.diff(t) <= 0):
        raise ValueError("Power timestamps are not strictly increasing")
    spans, valid = supported_spans(t, v, threshold, max_gap)
    return t, v, spans, valid


def scan_sensor(path, ctx):
    nd, nh = ctx["nd"], ctx["nh"]
    day_counts = np.zeros(nd, dtype=np.int64)
    run_counts = np.zeros(nd, dtype=np.int64)
    hour_counts = np.zeros(nh, dtype=np.int64)
    all_hist = np.zeros((nd, len(EDGES) - 1), dtype=np.int64)
    run_hist = np.zeros_like(all_hist)
    modes = [dict() for _ in range(nd)]
    n = dup = reverse = same = finite = nonfinite = 0
    last_t = last_v = None; first_t = None; max_gap = 0.0; max_pair = None
    run_starts, run_ends = ctx["spans"][1]
    for c in pd.read_csv(path, chunksize=CHUNK):
        t = ns_times(c["timestamp"])
        v = pd.to_numeric(c["value"], errors="coerce").to_numpy()
        nonfinite += int((~np.isfinite(v)).sum())
        day = ((t - ctx["d0"]) // DAY).astype(int)
        day_counts += np.bincount(day, minlength=nd)
        sid = containing_span(t, run_starts, run_ends)
        inside = sid >= 0
        run_counts += np.bincount(day[inside], minlength=nd)
        hour_counts += np.bincount(((t[inside] - ctx["h0"]) // HOUR).astype(int), minlength=nh)
        a = np.r_[last_t if last_t is not None else t[0], t[:-1]]
        dt = (t - a) / SECOND
        prev_sid = containing_span(a, run_starts, run_ends)
        good = dt > 0
        if first_t is None:
            first_t = int(t[0]); good[0] = False
        else:
            dup += int(dt[0] == 0)
        dup += int((np.diff(t) == 0).sum()); reverse += int((dt < 0).sum())
        full_run = good & inside & (prev_sid == sid)
        ids = np.searchsorted(EDGES, dt, side="right") - 1
        np.add.at(all_hist, (day[good], ids[good]), 1)
        np.add.at(run_hist, (day[full_run], ids[full_run]), 1)
        local = full_run & (((a - ctx["d0"]) // DAY) == day) & (dt >= .05) & (dt <= 3600)
        keys = np.rint(dt[local] * 10).astype(np.int64)
        packed = day[local].astype(np.int64) * 100000 + keys
        uniques, counts = np.unique(packed, return_counts=True)
        for key, count in zip(uniques, counts):
            d, k = divmod(int(key), 100000)
            modes[d][k] = modes[d].get(k, 0) + int(count)
        if good.any():
            m = int(np.argmax(np.where(good, dt, -1)))
            if dt[m] > max_gap:
                max_gap = float(dt[m]); max_pair = [int(a[m]), int(t[m])]
        pv = np.r_[last_v if last_v is not None else np.nan, v[:-1]]
        fv = np.isfinite(v) & np.isfinite(pv)
        same += int((fv & (v == pv)).sum()); finite += int(fv.sum())
        n += len(t); last_t = int(t[-1]); last_v = float(v[-1])
    if dup or reverse:
        raise ValueError(f"{path.name}: duplicates={dup}, reverse={reverse}; do not silently sort/drop")
    tau = np.full(nd, np.nan); dominance = np.full(nd, np.nan); mode_n = np.zeros(nd, dtype=int)
    for d, counts in enumerate(modes):
        total = sum(counts.values())
        mode_n[d] = total
        if total:
            peak = max(counts, key=counts.get)
            dominance[d] = counts[peak] / total
            if total >= 30 and dominance[d] >= .20:
                tau[d] = peak / 10

    gap_hist = np.zeros(len(EDGES) - 1, dtype=np.int64)
    normal_hist = np.zeros_like(gap_hist)
    gap_daily = np.zeros(nd, dtype=np.int64); gap_seconds_daily = np.zeros(nd)
    load_starts = np.zeros(5, dtype=np.int64)
    complete_normals = boundary_normals = candidate_n = comparable_n = 0
    expected_missing = 0; top = []
    prev_t = None; last_gap_key = None; last_gap_end = None
    comparison_groups = set(); gap_groups = set()
    # Second pass: cadence inferred in pass 1 is fixed. Only full, same-day,
    # supported-running intervals contribute to candidate gap events.
    for c in pd.read_csv(path, usecols=["timestamp"], chunksize=CHUNK):
        t = ns_times(c["timestamp"])
        a = np.r_[prev_t if prev_t is not None else t[0], t[:-1]]
        dt = (t - a) / SECOND
        day = ((t - ctx["d0"]) // DAY).astype(int)
        sa = containing_span(a, run_starts, run_ends)
        st = containing_span(t, run_starts, run_ends)
        local_tau = tau[day]
        eligible = (dt > 0) & (sa >= 0) & (sa == st) & (((a - ctx["d0"]) // DAY) == day) & np.isfinite(local_tau)
        keys = st.astype(np.int64) * nd + day
        comparison_groups.update(np.unique(keys[eligible]).tolist())
        gaps = eligible & (dt > 1.5 * local_tau)
        comparable_n += int(eligible.sum()); candidate_n += int(gaps.sum())
        durations = dt[gaps] - local_tau[gaps]
        gap_hist += np.histogram(durations, EDGES)[0]
        gap_daily += np.bincount(day[gaps], minlength=nd)
        gap_seconds_daily += np.bincount(day[gaps], weights=durations, minlength=nd)
        expected_missing += int(np.maximum(0, np.rint(dt[gaps] / local_tau[gaps]).astype(int) - 1).sum())
        if gaps.any():
            start = a[gaps] + np.rint(local_tau[gaps] * SECOND).astype(np.int64)
            gap_keys = keys[gaps]
            gap_groups.update(np.unique(gap_keys).tolist())
            wait = waiting_times(start, t[gaps], gap_keys, last_gap_key, last_gap_end)
            normal_hist += np.histogram(wait, EDGES)[0]
            complete_normals += len(wait)
            last_gap_key = int(gap_keys[-1]); last_gap_end = int(t[gaps][-1])
            pidx = np.searchsorted(ctx["pt"], start, side="right") - 1
            load = np.clip(np.searchsorted(ctx["load_edges"], ctx["pv"][np.maximum(pidx, 0)], side="right") - 1, 0, 4)
            load_starts += np.bincount(load, minlength=5)
            order = np.argsort(durations)[-10:]
            for k in order:
                top.append({"start_ns": int(start[k]), "end_ns": int(t[gaps][k]), "duration_s": float(durations[k]),
                            "inferred_cadence_s": float(local_tau[gaps][k]), "left_record_ns": int(a[gaps][k])})
            top = sorted(top, key=lambda z: z["duration_s"], reverse=True)[:30]
        prev_t = int(t[-1])
    # Left/right normal spans are boundary-truncated; no-gap comparison groups
    # have one truncated span. Their durations are not silently treated as full.
    boundary_normals = 2 * len(gap_groups) + len(comparison_groups - gap_groups)
    return {"tag": path.stem, "rows": n, "first_ns": first_t, "last_ns": last_t,
            "max_all_interval_s": max_gap, "max_all_interval_endpoints_ns": max_pair,
            "adjacent_equal_value_fraction": same / finite if finite else None,
            "nonfinite_value_records": nonfinite,
            "day_counts": day_counts, "run_counts": run_counts, "hour_counts": hour_counts,
            "all_hist": all_hist, "run_hist": run_hist, "cadence": tau, "dominance": dominance,
            "cadence_comparable_counts": mode_n, "gap_hist": gap_hist, "normal_hist": normal_hist,
            "gap_daily": gap_daily, "gap_seconds_daily": gap_seconds_daily, "load_starts": load_starts,
            "candidate_gap_events": candidate_n, "comparable_intervals": comparable_n,
            "candidate_missing_slots": expected_missing, "complete_normal_segments": complete_normals,
            "boundary_normal_segments_excluded": boundary_normals, "top_gaps": top}


def setup_style():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        fontManager.addfont(str(font))
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "figure.facecolor": "white", "axes.facecolor": "#f5f7fa",
                         "axes.spines.top": False, "axes.spines.right": False, "grid.alpha": .22,
                         "savefig.facecolor": "white"})


def save(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=145, bbox_inches="tight")
    plt.close(fig)


def time_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=25)


def hist_and_cdf(ax, hist, label, colour, cdf=False, survival=False):
    if hist.sum() == 0:
        return False
    y = np.cumsum(hist) / hist.sum() if cdf else hist / hist.sum()
    if survival:
        y = (hist.sum() - np.cumsum(hist) + hist) / hist.sum()
    good = hist > 0
    if cdf or survival:
        # Empty bins outside the observed support otherwise flatten short-gap
        # curves across many decades and hide their actual shape.
        occupied = np.flatnonzero(hist)
        good = np.zeros(len(hist), dtype=bool)
        good[occupied[0]:occupied[-1]+1] = True
    ax.plot(CENTRES[good], y[good], lw=1.6, label=label, color=colour,
            marker="o" if good.sum() == 1 else None, ms=5)
    if int(hist.sum()) < 10:
        ax.text(.5, .93, f"N={int(hist.sum())}；事件过少，不拟合分布", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
    ax.set_xscale("log")
    if survival:
        ax.set_yscale("log")
    ax.grid(True, which="both")
    return True


def autocorrelation(rate, max_lag=48):
    result = []
    bad = np.r_[0, np.cumsum(~np.isfinite(rate))]
    for lag in range(1, max_lag + 1):
        a, b = rate[:-lag], rate[lag:]
        good = (bad[lag + 1:] - bad[:-lag - 1]) == 0
        if good.sum() >= 24 and np.std(a[good]) > 0 and np.std(b[good]) > 0:
            result.append(float(np.corrcoef(a[good], b[good])[0, 1]))
        else:
            result.append(np.nan)
    return np.arange(1, max_lag + 1), np.array(result)


def comparable_load_exposure(r, ctx):
    """Same cohort as candidate events: supported state, estimable cadence,
    one day, and within this sensor's actual record coverage."""
    if "pair_run" not in ctx:
        t, v = ctx["pt"], ctx["pv"]
        dt = np.diff(t) / SECOND
        day = ((t[:-1] - ctx["d0"]) // DAY).astype(int)
        day_next = ((t[1:] - ctx["d0"]) // DAY).astype(int)
        ctx["pair_run"] = (dt > 0) & (dt <= ctx["max_gap"]) & (v[:-1] > 1) & (v[1:] > 1) & (day == day_next)
        ctx["pair_day"] = day; ctx["pair_dt"] = dt
        ctx["pair_load"] = np.clip(np.searchsorted(ctx["load_edges"], v[:-1], side="right") - 1, 0, 4)
    keep = ctx["pair_run"] & np.isfinite(r["cadence"][ctx["pair_day"]])
    keep &= (ctx["pt"][:-1] >= r["first_ns"]) & (ctx["pt"][1:] <= r["last_ns"])
    return np.bincount(ctx["pair_load"][keep], weights=ctx["pair_dt"][keep], minlength=5)


def sensor_figure(r, ctx, labels, out):
    fig, axs = plt.subplots(4, 2, figsize=(17, 19), layout="constrained")
    title = labels.get(r["tag"], r["tag"])
    fig.suptitle(f"JQRD GT1 | {title}\n{r['tag']}", fontsize=17)
    days = pd.to_datetime(ctx["d0"] + np.arange(ctx["nd"]) * DAY)
    axs = axs.ravel()
    full_med = histogram_quantile(r["all_hist"], .5)
    run_med = histogram_quantile(r["run_hist"], .5)
    run_p95 = histogram_quantile(r["run_hist"], .95)
    axs[0].plot(days, full_med, color="#9ba7b5", label="全部记录间隔 P50")
    axs[0].plot(days, run_med, color="#176b87", label="运行支持内 P50")
    axs[0].plot(days, run_p95, color="#df7a37", alpha=.8, label="运行支持内 P95")
    axs[0].set_yscale("log"); axs[0].set_ylabel("相邻记录间隔 / s")
    axs[0].set_title("① 间隔随日期变化（分位数由全量直方图近似）"); time_axis(axs[0]); axs[0].legend(fontsize=9)
    run_h = ctx["run_day"] / 3600
    density = np.divide(r["run_counts"], run_h, out=np.full(ctx["nd"], np.nan), where=run_h >= .5)
    axs[1].plot(days, density, color="#176b87")
    axs[1].set_yscale("log"); axs[1].set_ylabel("记录数 / 有运行证据的小时")
    axs[1].set_title("② 运行期间记录密度（每日支持至少 0.5 h）"); time_axis(axs[1])
    hist_and_cdf(axs[2], r["all_hist"].sum(axis=0), "全部记录", "#9ba7b5")
    hist_and_cdf(axs[2], r["run_hist"].sum(axis=0), "运行支持内", "#176b87")
    axs[2].set_title("③ 记录间隔分布（对数等宽分箱，按事件计数）")
    axs[2].set_xlabel("间隔 / s"); axs[2].set_ylabel("每箱比例"); axs[2].legend()
    hist_and_cdf(axs[3], r["run_hist"].sum(axis=0), "运行记录间隔", "#176b87", cdf=True)
    axs[3].set_title("④ 运行记录间隔 ECDF（全量分箱近似）")
    axs[3].set_xlabel("间隔 / s"); axs[3].set_ylabel("累计比例"); axs[3].set_ylim(0, 1.02)
    if not hist_and_cdf(axs[4], r["gap_hist"], "候选额外空档", "#bf4b45", survival=True):
        axs[4].text(.5, .5, "没有可评分的候选空档\n不代表没有缺失", transform=axs[4].transAxes, ha="center")
    axs[4].set_title("⑤ 候选空档生存曲线（dt > 1.5τ，时长 dt−τ）")
    axs[4].set_xlabel("候选空档 / s"); axs[4].set_ylabel("达到或超过该长度的比例")
    if not hist_and_cdf(axs[5], r["normal_hist"], "完整正常记录段", "#287b58", cdf=True):
        axs[5].text(.5, .5, "完整正常段不足\n不拟合持续时间分布", transform=axs[5].transAxes, ha="center")
    axs[5].set_title("⑥ 同日候选空档间正常段 ECDF（单个保留点计入τ；边界段排除）")
    axs[5].set_xlabel("正常段 / s"); axs[5].set_ylabel("累计比例")
    hourly = np.divide(r["hour_counts"], ctx["run_hour"] / 3600, out=np.full(ctx["nh"], np.nan), where=ctx["run_hour"] >= 1800)
    lags, acf = autocorrelation(hourly)
    axs[6].plot(lags, acf, color="#7865a8", marker=".", ms=3)
    axs[6].axhline(0, color="#9ba7b5", lw=1)
    axs[6].set_ylim(-1.05, 1.05); axs[6].set_xlabel("时滞 / h"); axs[6].set_ylabel("相关系数")
    axs[6].set_title("⑦ 小时记录密度自相关（不跨未知段；不是缺失掩码 ACF）")
    eligible_exposure = comparable_load_exposure(r, ctx)
    rate = np.divide(r["load_starts"], eligible_exposure / 3600, out=np.full(5, np.nan), where=eligible_exposure >= 1800)
    axs[7].bar(np.arange(5), rate, color="#df7a37", alpha=.85)
    axs[7].set_xticks(np.arange(5), ["1–50", "50–100", "100–150", "150–200", "≥200"])
    axs[7].set_xlabel("空档起始前可观测出力 / MW"); axs[7].set_ylabel("候选空档起始数 / 运行支持小时")
    axs[7].set_title("⑧ 负荷分层（仅按该测点周期可比区段曝光归一）")
    if r["candidate_gap_events"] == 0:
        axs[7].set_ylim(0, 1)
        axs[7].text(.5, .5, "当前可比区段内候选起始数为 0\n不代表全部时期无缺失", transform=axs[7].transAxes, ha="center", fontsize=9)
    for ax in axs:
        ax.grid(True, alpha=.2)
    fig.get_layout_engine().set(rect=(0, .045, 1, .96))
    fig.text(.015, .012, f"状态支持：相邻出力记录同属 >1 MW 且间隔≤{ctx['max_gap']:g} s。τ 为每日 0.1 s 分箱众数，N≥30、占比≥20%。\n"
             "候选空档仅限同一连续运行支持段、同一日且τ可估区段；跨边界空档不截短纳入。采集周期/压缩机制未核验，不能解释为故障率。", fontsize=9)
    save(fig, out / "sensors", r["tag"])


def overview(results, ctx, out, pt, pv, sensitivity):
    nd = ctx["nd"]; dates = pd.to_datetime(ctx["d0"] + np.arange(nd) * DAY)
    fig, ax = plt.subplots(3, 1, figsize=(17, 10), sharex=True, layout="constrained")
    hour = ((pt - ctx["h0"]) // HOUR).astype(int)
    counts = np.bincount(hour, minlength=ctx["nh"])
    mean = np.divide(np.bincount(hour, weights=pv, minlength=ctx["nh"]), counts,
                     out=np.full(ctx["nh"], np.nan), where=counts > 0)
    hours = pd.to_datetime(ctx["h0"] + np.arange(ctx["nh"]) * HOUR)
    ax[0].plot(hours, mean, color="#176b87", lw=.8)
    ax[0].axhline(1, color="#bf4b45", ls="--", label="运行代理阈值 1 MW")
    ax[0].set_ylabel("有功功率 / MW"); ax[0].legend(); ax[0].set_title("出力小时记录均值（未插值；均值按记录计数）")
    run = ctx["run_day"] / 3600; stop = ctx["stop_day"] / 3600
    unknown = np.maximum(0, 24 - run - stop)
    ax[1].stackplot(dates, run, stop, unknown, labels=["有运行证据", "有未发电证据", "状态未知"],
                    colors=["#287b58", "#df7a37", "#d2d9e1"], step="mid")
    ax[1].set_ylim(0, 24); ax[1].set_ylabel("每天时长 / h"); ax[1].legend(loc="upper left", ncol=3)
    ax[1].set_title(f"状态覆盖：相邻功率记录同状态且间隔≤{ctx['max_gap']:g} s 的区间才计入支持")
    pday = np.bincount(((pt - ctx["d0"]) // DAY).astype(int), minlength=nd)
    ax[2].plot(dates, pday, color="#7865a8")
    ax[2].set_yscale("log"); ax[2].set_ylabel("有功功率记录数 / 日"); ax[2].set_title("出力记录本身的密度变化")
    for a in ax: a.grid(True, alpha=.2)
    time_axis(ax[2]); fig.suptitle("JQRD GT1 | 出力、状态支持与记录覆盖", fontsize=18)
    save(fig, out, "01_power_and_state")

    density = np.array([np.divide(r["run_counts"], run, out=np.full(nd, np.nan), where=run >= .5) for r in results])
    all_counts = np.array([r["day_counts"] for r in results], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(19, 21), layout="constrained")
    extent = [mdates.date2num(dates[0]), mdates.date2num(dates[-1] + pd.Timedelta(days=1)), len(results)-.5, -.5]
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#d2d9e1")
    for a, data, title in zip(axes, [all_counts, density], ["全部时间：每日记录数", "仅运行支持：每运行小时记录数（灰色为运行支持不足；紫色可表示零记录）"]):
        data = np.ma.masked_invalid(np.log10(data + 1))
        im = a.imshow(data, aspect="auto", extent=extent, interpolation="nearest", cmap=cmap, vmin=0, vmax=max(1, float(data.max())))
        a.set_yticks(np.arange(len(results)), [r["tag"].replace("JQRD_", "") for r in results], fontsize=8)
        a.xaxis_date(); time_axis(a); a.set_title(title); fig.colorbar(im, ax=a, pad=.01, shrink=.8, label="log10(记录数 + 1)；0 与未知分开")
    fig.suptitle("JQRD GT1 | 全部 35 测点记录密度", fontsize=18)
    save(fig, out, "02_all_sensor_density")

    med = np.array([histogram_quantile(r["run_hist"], .5) for r in results])
    fig, a = plt.subplots(figsize=(19, 12), layout="constrained")
    masked = np.ma.masked_invalid(med)
    im = a.imshow(masked, aspect="auto", extent=extent, interpolation="nearest", cmap=cmap, norm=LogNorm(vmin=.1, vmax=max(1, float(masked.max()))))
    a.set_yticks(np.arange(len(results)), [r["tag"].replace("JQRD_", "") for r in results], fontsize=8)
    a.xaxis_date(); time_axis(a); fig.colorbar(im, ax=a, label="每日运行支持内记录间隔 P50 / s", pad=.01)
    a.set_title("JQRD GT1 | 记录间隔随时间变化（全量分箱近似；灰色无可比区间）", fontsize=16)
    save(fig, out, "03_all_sensor_interval")

    fig, a = plt.subplots(figsize=(10, 6), layout="constrained")
    x = [s["max_gap_s"] for s in sensitivity]
    a.plot(x, [s["running_hours"] for s in sensitivity], marker="o", color="#287b58", label="有运行证据时长")
    a.plot(x, [s["non_generating_hours"] for s in sensitivity], marker="o", color="#df7a37", label="有未发电证据时长")
    a.set_xscale("log"); a.set_xticks(x, [str(z) for z in x]); a.set_ylabel("时长 / h")
    a.set_xlabel("允许连接相邻同状态功率记录的最大间隔 / s")
    a.set_title("运行状态支持的间隔容限敏感性（不等于真实运行时长）"); a.legend(); a.grid(True)
    save(fig, out, "04_state_support_sensitivity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/jqrd_gt1_missingness_20261012")
    parser.add_argument("--state-max-gap", type=float, default=5)
    parser.add_argument("--power-threshold", type=float, default=1)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.power_threshold != 1:
        raise ValueError("The previously requested status CSV uses 1 MW; use a separate explicit protocol to change it")
    check_logic(); setup_style()
    out = (ROOT / args.output).resolve()
    if ROOT / "figures" not in out.parents:
        raise ValueError("Output must be a new subdirectory of figures")
    if out.exists():
        # Reuse only an entirely empty output scaffold after an interrupted scan.
        if any(p.is_file() for p in out.rglob("*")):
            raise FileExistsError("Use a new output directory; existing artifacts are never overwritten")
    out.mkdir(parents=True, exist_ok=True); (out / "sensors").mkdir(exist_ok=True); (out / "statistics").mkdir(exist_ok=True)
    paths = sorted(SOURCE.glob("*.csv")); before = [metadata(p) for p in paths]
    workbook = ROOT / "raw_data" / "jqrd_gt1" / "jqrd_gt1_选取点表.xlsx"
    wb = load_workbook(workbook, read_only=True, data_only=True)
    labels = {str(row[0]): str(row[1]) for row in list(wb.worksheets[0].values)[1:] if row[0]}
    wb.close()
    pt, pv, spans, valid = load_power(args.power_threshold, args.state_max_gap)
    # This source range was independently inspected before plotting. Expand
    # conservatively from all tag first/last rows without loading entire files.
    firsts = []; lasts = []
    for p in paths:
        with p.open("rb") as f:
            f.readline(); firsts.append(pd.Timestamp(f.readline().decode().split(",")[0]).value)
            f.seek(max(0, p.stat().st_size - 4096)); tail = f.read().decode().strip().splitlines()
            lasts.append(pd.Timestamp(tail[-1].split(",")[0]).value)
    d0 = min(firsts) // DAY * DAY; d1 = (max(lasts) // DAY + 1) * DAY
    nd = int((d1-d0)//DAY); nh=nd*24
    ctx = {"d0": d0, "h0": d0, "nd": nd, "nh": nh, "spans": spans, "pt": pt, "pv": pv,
           "max_gap": args.state_max_gap, "load_edges": np.array([1, 50, 100, 150, 200, np.inf])}
    ctx["run_day"] = exposure_bins(*spans[1], d0, DAY, nd)
    ctx["stop_day"] = exposure_bins(*spans[0], d0, DAY, nd)
    ctx["run_hour"] = exposure_bins(*spans[1], d0, HOUR, nh)
    dt = np.diff(pt) / SECOND
    load = np.clip(np.searchsorted(ctx["load_edges"], pv[:-1], side="right") - 1, 0, 4)
    running_pair = valid & (pv[:-1] > args.power_threshold)
    ctx["load_exposure"] = np.bincount(load[running_pair], weights=dt[running_pair], minlength=5)
    sensitivity=[]
    for cap in (2, 5, 10, 60):
        s, _ = supported_spans(pt, pv, args.power_threshold, cap)
        sensitivity.append({"max_gap_s": cap, "running_hours": float((s[1][1]-s[1][0]).sum()/SECOND/3600),
                            "non_generating_hours": float((s[0][1]-s[0][0]).sum()/SECOND/3600)})
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(scan_sensor, p, ctx): p for p in paths}
        for f in as_completed(pending):
            r=f.result(); results.append(r)
            arrays={k:v for k,v in r.items() if isinstance(v,np.ndarray)}
            scalars={k:v for k,v in r.items() if not isinstance(v,np.ndarray)}
            np.savez_compressed(out/"statistics"/(r["tag"]+"_cache.npz"),**arrays)
            (out/"statistics"/(r["tag"]+"_cache.json")).write_text(json.dumps(scalars,ensure_ascii=False),encoding="utf-8")
            print(f"SCANNED {len(results)}/35 {r['tag']} rows={r['rows']} candidate_gaps={r['candidate_gap_events']}", flush=True)
    results.sort(key=lambda r:r["tag"])
    overview(results, ctx, out, pt, pv, sensitivity)
    summary=[]; daily=[]; top=[]; histrows=[]; load_rows=[]
    for i,r in enumerate(results):
        sensor_figure(r,ctx,labels,out)
        row={k:r[k] for k in ("tag","rows","max_all_interval_s","adjacent_equal_value_fraction","nonfinite_value_records","candidate_gap_events","comparable_intervals","candidate_missing_slots","complete_normal_segments","boundary_normal_segments_excluded")}
        row.update(label=labels.get(r["tag"],r["tag"]), first=str(pd.Timestamp(r["first_ns"])),last=str(pd.Timestamp(r["last_ns"])),
                   running_supported_record_count=int(r["run_counts"].sum()),cadence_estimable_days=int(np.isfinite(r["cadence"]).sum()))
        summary.append(row)
        exposure = comparable_load_exposure(r, ctx)
        for b in range(5):
            load_rows.append({"tag":r["tag"],"load_lower_MW":ctx["load_edges"][b],"load_upper_MW":ctx["load_edges"][b+1],
                              "cadence_comparable_exposure_hours":exposure[b]/3600,"candidate_gap_starts":int(r["load_starts"][b])})
        for d in range(nd):
            daily.append({"tag":r["tag"],"date":str(pd.Timestamp(d0+d*DAY).date()),"all_records":int(r["day_counts"][d]),
                          "running_supported_records":int(r["run_counts"][d]),"running_supported_hours":ctx["run_day"][d]/3600,
                          "candidate_cadence_s":r["cadence"][d],"cadence_mode_fraction":r["dominance"][d],
                          "cadence_comparable_intervals":int(r["cadence_comparable_counts"][d]),"candidate_gap_events":int(r["gap_daily"][d]),
                          "candidate_gap_seconds":float(r["gap_seconds_daily"][d])})
        for e in r["top_gaps"]:
            top.append({"tag":r["tag"],"candidate_start":str(pd.Timestamp(e["start_ns"])),"right_record":str(pd.Timestamp(e["end_ns"])),
                        "left_record":str(pd.Timestamp(e["left_record_ns"])),"duration_s":e["duration_s"],"candidate_cadence_s":e["inferred_cadence_s"]})
        for j in range(len(CENTRES)):
            histrows.append({"tag":r["tag"],"lower_s":EDGES[j],"upper_s":EDGES[j+1],"plot_centre_s":CENTRES[j],
                             "all_interval_count":int(r["all_hist"].sum(axis=0)[j]),"running_interval_count":int(r["run_hist"].sum(axis=0)[j]),
                             "candidate_gap_count":int(r["gap_hist"][j]),"complete_normal_segment_count":int(r["normal_hist"][j])})
        print(f"PLOTTED {i+1}/35 {r['tag']}",flush=True)
    pd.DataFrame(summary).to_csv(out/"statistics"/"sensor_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(daily).to_csv(out/"statistics"/"daily_statistics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(top).to_csv(out/"statistics"/"longest_candidate_gaps.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(histrows).to_csv(out/"statistics"/"distribution_histograms.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(load_rows).to_csv(out/"statistics"/"load_exposure_by_sensor.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(out/"statistics"/"state_support_sensitivity.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"date":pd.to_datetime(d0+np.arange(nd)*DAY),"running_supported_hours":ctx["run_day"]/3600,
                  "non_generating_supported_hours":ctx["stop_day"]/3600,"unknown_hours":24-(ctx["run_day"]+ctx["stop_day"])/3600}).to_csv(out/"statistics"/"state_exposure_daily.csv",index=False)
    after=[metadata(p) for p in paths]
    if before!=after: raise ValueError("Source metadata changed during analysis")
    status=ROOT/"raw_data"/"jqrd_gt1"/"jqrd_gt1_running_status.csv"
    manifest={"created_on":datetime.now().isoformat(),"timezone":"Asia/Shanghai","source_scope":"full timestamp and adjacent-value scan of all 35 current JQRD GT1 CSVs; no old project method reuse",
              "source_files":before,"source_integrity_check":"path/bytes/mtime preserved; not full source-content hash",
              "script_sha256":sha256(Path(__file__)),"state_csv_sha256":sha256(status),"power_tag":POWER_TAG,"power_threshold_MW":1,
              "state_support_max_gap_s":args.state_max_gap,"state_support_assumption":"state constant only between close same-state recorded power endpoints; no support before first power record or through longer gaps",
              "running_supported_hours":float(ctx["run_day"].sum()/3600),"non_generating_supported_hours":float(ctx["stop_day"].sum()/3600),
              "known_state_coverage_bias":"joint recording outages can remove power support and be excluded; these figures do not estimate all outage frequencies",
              "candidate_cadence":"daily 0.1-s-rounded interval mode from full supported-running, same-day intervals, 0.05..3600 s; at least 30 intervals and modal fraction >=0.20",
              "candidate_gap":"dt > 1.5*tau; duration dt-tau; interval must lie completely in one supported-running span and one day with estimated tau",
              "normal_segments":"next candidate-gap start minus previous candidate-gap end, same supported span/day; one retained point contributes tau; boundary stretches counted and excluded, not treated as complete durations",
              "unverified":"nominal scanning/archiving cadence, instantaneous/aggregate/compressed/held semantics, true fault cause, physical startup state",
              "acf":"hourly record density, exposure >=0.5 h; pairs cannot cross an unsupported hour; not a verified missing-indicator ACF",
              "outputs":{"png":39,"sensors":35},"total_records":sum(r["rows"] for r in results),"python_libraries":{"pandas":pd.__version__,"numpy":np.__version__,"matplotlib":matplotlib.__version__},
              "self_checks":"synthetic support/gap/boundary/exposure checks passed"}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    sections=[]
    for name,title in [("01_power_and_state","出力与状态支持"),("02_all_sensor_density","全部测点记录密度"),("03_all_sensor_interval","全部测点记录间隔"),("04_state_support_sensitivity","状态支持容限敏感性")]:
        sections.append(f'<section><h2>{title}</h2><a href="{name}.png"><img src="{name}.png" loading="lazy"></a></section>')
    for r in results:
        tag=html.escape(r["tag"]); label=html.escape(labels.get(r["tag"],r["tag"]))
        sections.append(f'<section id="{tag}"><h2>{label} · {tag}</h2><a href="sensors/{tag}.png"><img src="sensors/{tag}.png" loading="lazy"></a></section>')
    nav=" ".join(f'<a href="#{html.escape(r["tag"])}">{html.escape(labels.get(r["tag"],r["tag"]))}</a>' for r in results)
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JQRD GT1 记录规律图册</title><style>body{font:16px/1.65 system-ui;margin:0;background:#edf2f6;color:#172c3b}main{max-width:1450px;margin:auto;padding:28px}section{background:white;margin:24px 0;padding:20px;border-radius:12px}img{width:100%;height:auto}a{color:#176b87;margin-right:14px}h1{font-size:30px}h2{font-size:21px}nav a{display:inline-block;margin:5px}</style><main><h1>JQRD GT1：运行条件下的记录规律</h1><p>35 测点全量扫描。图中“运行”是出力代理且仅在相邻同状态功率记录间隔≤5秒的区间得到支持。未知时段不填为停机。</p><p>记录间隔是观测事实；候选额外空档依赖每日候选周期，尚未核实采集/归档/压缩规则。不能据此宣称真实故障率或最佳缺失分布。点击图片查看原尺寸。</p><p><a href="statistics/sensor_summary.csv">测点汇总</a><a href="statistics/daily_statistics.csv">每日统计</a><a href="manifest.json">方法与来源</a></p><nav>'+nav+'</nav>'+"".join(sections)+'</main></html>'
    (out/"index.html").write_text(page,encoding="utf-8")
    print(json.dumps({"output":str(out),"rows":manifest["total_records"],"running_supported_hours":manifest["running_supported_hours"],"png":len(list(out.rglob("*.png")))},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    import runpy
    runpy.run_path(str(ROOT / "figures" / "plot_jqrd_gt1_heatmap.py"), run_name="__main__")
