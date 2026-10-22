"""七台燃机任务一。分块统计，系统临时目录缓存，目录内只输出SVG。

依赖：numpy、pandas、pyarrow、openpyxl、matplotlib。
运行：python gt_analysis/run.py
合成检查：python gt_analysis/run.py check
"""
import csv
import hashlib
import inspect
import json
import pickle
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as arrow_csv

import config

SECOND = 1_000_000_000
EXCLUDED = {"H2", "N2", "CO2", "CO", "C2H6", "C3H8", "iC4H10",
            "nC4H10", "iC5H12", "nC5H12", "oxygen_channel", "C6plus"}
CACHE = Path(tempfile.gettempdir()) / "gt_analysis_task1"


def file_info(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def timestamp(value):
    time = pd.Timestamp(value)
    if time.tzinfo is not None:
        time = time.tz_convert("Asia/Shanghai").tz_localize(None)
    return time.value


def points(unit):
    folder = config.ROOT / "raw_data" / unit
    workbooks = list(folder.glob("*选取点表.xlsx"))
    if len(workbooks) != 1:
        raise ValueError(f"点表数量不是1：{folder}")
    frame = pd.read_excel(workbooks[0], sheet_name=0)
    selected = []
    for row in frame.itertuples(index=False, name=None):
        code, name, semantic = row[:3]
        if pd.isna(code) or semantic in EXCLUDED:
            continue
        if config.SENSORS is not None and code not in config.SENSORS:
            continue
        selected.append({"code": code, "name": name, "semantic": semantic,
                         "path": folder / "csv" / f"{code}.csv"})
    if not selected or len({p["code"] for p in selected}) != len(selected):
        raise ValueError(f"点表为空或编码重复：{unit}")
    power = frame[frame.iloc[:, 2] == "active_power"]
    if len(power) != 1:
        raise ValueError(f"出力测点不唯一：{unit}")
    power_path = folder / "csv" / f"{power.iloc[0, 0]}.csv"
    return selected, workbooks[0], power_path


def source_extent(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        first = next(reader)
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        lines = stream.read().decode("utf-8-sig").splitlines()
    last = next(csv.reader([next(line for line in reversed(lines) if line.strip())]))
    return timestamp(first["timestamp"]), timestamp(last[reader.fieldnames.index("timestamp")])


def window_grid(start, end, window, step):
    width = pd.Timedelta(window).value
    stride = pd.Timedelta(step).value
    if width <= 0 or stride <= 0 or stride % pd.Timedelta("1h").value:
        raise ValueError("窗口必须为正，滑动步长必须为正整数小时")
    # 所有末端对齐到本地整点，随后按配置步长推进。
    first = ((start + width + SECOND * 3600 - 1) // (SECOND * 3600)) * SECOND * 3600
    ends = np.arange(first, end + 1, stride, dtype=np.int64)
    if len(ends) == 0:
        raise ValueError("所选范围不足一个完整窗口")
    return ends, width


def window_sums(times, values, ends, width):
    left = np.searchsorted(times, ends - width, side="right")
    right = np.searchsorted(times, ends, side="right")
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return cumulative[right] - cumulative[left], right - left


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals, key=lambda pair: (pair[0], pair[1])):
        start, end = int(start), int(end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def stop_candidates(sensor_intervals, sensor_count):
    events = {}
    for intervals in sensor_intervals:
        for start, end in merge_intervals(intervals):
            events[start] = events.get(start, 0) + 1
            events[end] = events.get(end, 0) - 1
    count = 0
    start = None
    result = []
    threshold = sensor_count // 2 + 1
    for time, change in sorted(events.items()):
        previous = count
        count += change
        if previous < threshold <= count:
            start = time
        if previous >= threshold > count:
            result.append([start, time])
    return result


def display_indices(times, values, bucket):
    """每个绘图桶保留首、末、最小、最大值；不参与任何统计。"""
    starts = np.r_[0, np.flatnonzero(np.diff(times // bucket)) + 1]
    ends = np.r_[starts[1:], len(times)]
    selected = []
    for left, right in zip(starts, ends):
        part = values[left:right]
        selected.extend([left, right - 1, left + int(part.argmin()), left + int(part.argmax())])
    return np.unique(selected)


def scan_source(path, ends, width, power_ends, power_width, is_power=False):
    before = file_info(path)
    interval_sum = np.zeros(len(ends))
    interval_count = np.zeros(len(ends), dtype=np.int64)
    power_sum = np.zeros(len(power_ends))
    power_count = np.zeros(len(power_ends), dtype=np.int64)
    stop_intervals = []
    power_gaps = []
    display_times, display_values = [], []
    sample_indices = np.unique([0, len(ends) // 2, len(ends) - 1])
    sample_ends = ends[sample_indices]
    direct_sum = np.zeros(len(sample_indices))
    direct_count = np.zeros(len(sample_indices), dtype=np.int64)
    power_samples = np.unique([0, len(power_ends) // 2, len(power_ends) - 1])
    direct_power_sum = np.zeros(len(power_samples))
    direct_power_count = np.zeros(len(power_samples), dtype=np.int64)
    threshold = config.STOP_MINUTES * 60 * SECOND
    power_threshold = config.POWER_GAP_MINUTES * 60 * SECOND
    bucket = pd.Timedelta(config.DISPLAY_BUCKET).value
    if threshold <= 0 or power_threshold <= 0 or bucket <= 0:
        raise ValueError("停机阈值、断线阈值及绘图桶长度必须为正")

    previous_time = None
    previous_value = None
    run_start = None
    first_time = None
    row_count = 0
    reader = arrow_csv.open_csv(
        path,
        read_options=arrow_csv.ReadOptions(block_size=config.CSV_BLOCK_BYTES, use_threads=False),
        convert_options=arrow_csv.ConvertOptions(
            include_columns=["timestamp", "value"],
            column_types={"timestamp": pa.timestamp("ns"), "value": pa.float64()},
        ),
    )
    try:
        for batch in reader:
            if batch.column(0).null_count or batch.column(1).null_count:
                nulls = np.asarray(batch.column(0).is_null()) | np.asarray(batch.column(1).is_null())
                bad = int(np.flatnonzero(nulls)[0])
                raise ValueError(f"无效空值：{path}，数据行{row_count + bad + 1}")
            times = batch.column(0).cast(pa.int64()).to_numpy()
            values = batch.column(1).to_numpy()
            if len(times) == 0:
                continue
            invalid = np.flatnonzero(~np.isfinite(values))
            if len(invalid):
                raise ValueError(f"非有限数值：{path}，数据行{row_count + int(invalid[0]) + 1}")
            invalid = np.flatnonzero(np.diff(times) <= 0)
            if len(invalid) or (previous_time is not None and times[0] <= previous_time):
                bad = int(invalid[0]) + 2 if len(invalid) else 1
                raise ValueError(f"重复或逆序时间戳：{path}，数据行{row_count + bad}")

            raw_times, raw_values = times, values
            if previous_time is None:
                first_time = int(times[0])
                run_start = first_time
            else:
                times = np.r_[previous_time, times]
                values = np.r_[previous_value, values]

            deltas = np.diff(times)
            sums, counts = window_sums(times[1:], deltas / SECOND, ends, width)
            interval_sum += sums
            interval_count += counts
            for i, end in enumerate(sample_ends):
                mask = (times[1:] > end - width) & (times[1:] <= end)
                direct_sum[i] += np.sum(deltas[mask] / SECOND)
                direct_count[i] += int(mask.sum())

            gaps = np.flatnonzero(deltas >= threshold)
            stop_intervals.extend(zip(times[gaps].tolist(), times[gaps + 1].tolist()))
            changes = np.flatnonzero(values[1:] != values[:-1]) + 1
            if len(changes):
                starts = np.r_[run_start, times[changes[:-1]]]
                stops = times[changes - 1]
                long_runs = stops - starts >= threshold
                stop_intervals.extend(zip(starts[long_runs].tolist(), stops[long_runs].tolist()))
                run_start = int(times[changes[-1]])

            if is_power:
                sums, counts = window_sums(raw_times, raw_values, power_ends, power_width)
                power_sum += sums
                power_count += counts
                for i, index in enumerate(power_samples):
                    end = power_ends[index]
                    mask = (raw_times > end - power_width) & (raw_times <= end)
                    direct_power_sum[i] += raw_values[mask].sum()
                    direct_power_count[i] += int(mask.sum())
                gaps = np.flatnonzero(deltas > power_threshold)
                power_gaps.extend(zip(times[gaps].tolist(), times[gaps + 1].tolist()))
                indices = display_indices(raw_times, raw_values, bucket)
                display_times.append(raw_times[indices])
                display_values.append(raw_values[indices])

            previous_time = int(times[-1])
            previous_value = float(values[-1])
            row_count += len(raw_times)
    except pa.ArrowInvalid as error:
        raise ValueError(f"CSV解析失败：{path}，当前块从数据行{row_count + 1}开始：{error}") from error
    finally:
        reader.close()

    if previous_time is None:
        raise ValueError(f"空数据文件：{path}")
    if previous_time - run_start >= threshold:
        stop_intervals.append([run_start, previous_time])
    if before != file_info(path):
        raise RuntimeError(f"计算期间源文件改变：{path}")
    np.testing.assert_array_equal(interval_count[sample_indices], direct_count)
    np.testing.assert_allclose(interval_sum[sample_indices], direct_sum, rtol=1e-9, atol=1e-6)
    if is_power:
        np.testing.assert_array_equal(power_count[power_samples], direct_power_count)
        np.testing.assert_allclose(power_sum[power_samples], direct_power_sum, rtol=1e-8, atol=1e-4)

    interval_mean = np.full(len(ends), np.nan)
    np.divide(interval_sum, interval_count, out=interval_mean, where=interval_count > 0)
    power_mean = np.full(len(power_ends), np.nan)
    np.divide(power_sum, power_count, out=power_mean, where=power_count > 0)
    if is_power:
        display_times = np.concatenate(display_times)
        display_values = np.concatenate(display_values)
        indices = display_indices(display_times, display_values, bucket)
        display_times, display_values = display_times[indices], display_values[indices]
    return {"source": before, "rows": row_count, "first": first_time, "last": previous_time,
            "interval_mean": interval_mean, "interval_count": interval_count,
            "stops": merge_intervals(stop_intervals), "power_mean": power_mean,
            "power_count": power_count, "power_gaps": power_gaps,
            "display_times": display_times, "display_values": display_values,
            "window_checks": len(sample_indices),
            "power_window_checks": len(power_samples) if is_power else 0}


def power_speed(ends, mean, width, gaps):
    valid = np.isfinite(mean)
    for start, end in gaps:
        # 无记录开区间与窗口内部重叠，则不能用该均值计算变化率。
        valid &= ~((ends > start) & (ends - width < end))
    speed = np.full(len(ends), np.nan)
    minutes = np.diff(ends) / (60 * SECOND)
    adjacent_valid = valid[1:] & valid[:-1]
    speed[1:] = np.where(adjacent_valid, np.abs(np.diff(mean)) / minutes, np.nan)
    return speed


def stopped_mask(times, stops):
    mask = np.zeros(len(times), dtype=bool)
    for start, end in stops:
        mask |= (times >= start) & (times < end)
    return mask


def average_speed(ends, speed, stops=()):
    """停机数据不参与平均，停机时刻不输出平均曲线；不以零代替。"""
    width = pd.Timedelta(config.SPEED_WINDOW).value
    if width <= 0:
        raise ValueError("变化率平均窗口必须为正")
    stopped = stopped_mask(ends, stops)
    valid = np.isfinite(speed) & ~stopped
    sums, counts = window_sums(ends[valid], speed[valid], ends, width)
    complete = (ends - width >= ends[0]) & (counts > 0) & ~stopped
    averaged = np.full(len(ends), np.nan)
    np.divide(sums, counts, out=averaged, where=complete)
    return averaged


def running_power_mean(path, ends, width, stops):
    """从原始记录求非停机样本的均值，分母仅计非停机记录。"""
    signature = json.dumps([file_info(path), ends.tolist(), width, stops])
    code = inspect.getsource(running_power_mean) + inspect.getsource(stopped_mask)
    cache = CACHE / ("running_power_" + hashlib.sha256((signature + code).encode()).hexdigest() + ".pkl")
    if cache.exists():
        with cache.open("rb") as stream:
            return pickle.load(stream)
    before = file_info(path)
    sums = np.zeros(len(ends))
    counts = np.zeros(len(ends), dtype=np.int64)
    with arrow_csv.open_csv(path,
            read_options=arrow_csv.ReadOptions(block_size=config.CSV_BLOCK_BYTES),
            convert_options=arrow_csv.ConvertOptions(column_types={
                "timestamp": pa.timestamp("ns"), "value": pa.float64()})) as reader:
        for batch in reader:
            times = batch.column(0).cast(pa.int64()).to_numpy()
            values = batch.column(1).to_numpy()
            keep = ~stopped_mask(times, stops)
            chunk_sums, chunk_counts = window_sums(times[keep], values[keep], ends, width)
            sums += chunk_sums
            counts += chunk_counts
    if before != file_info(path):
        raise RuntimeError(f"运行出力分析期间源文件改变：{path}")
    mean = np.full(len(ends), np.nan)
    np.divide(sums, counts, out=mean, where=counts > 0)
    mean[stopped_mask(ends, stops)] = np.nan
    with cache.open("wb") as stream:
        pickle.dump((mean, counts), stream)
    return mean, counts


def low_power_stops(path):
    """相邻记录之间按前值判断低出力；超过长缺口阈值不延续状态。"""
    source = file_info(path)
    signature = json.dumps([source, config.STOP_POWER_MW, config.POWER_GAP_MINUTES])
    key = hashlib.sha256((signature + inspect.getsource(low_power_stops)).encode()).hexdigest()
    cache = CACHE / f"low_power_{key}.pkl"
    if cache.exists():
        with cache.open("rb") as stream:
            return pickle.load(stream)
    intervals = []
    previous_time = None
    previous_value = None
    with arrow_csv.open_csv(path,
            read_options=arrow_csv.ReadOptions(block_size=config.CSV_BLOCK_BYTES),
            convert_options=arrow_csv.ConvertOptions(column_types={
                "timestamp": pa.timestamp("ns"), "value": pa.float64()})) as reader:
        for batch in reader:
            times = batch.column(0).cast(pa.int64()).to_numpy()
            values = batch.column(1).to_numpy()
            if previous_time is not None:
                times = np.r_[previous_time, times]
                values = np.r_[previous_value, values]
            low = (values[:-1] < config.STOP_POWER_MW)
            low &= np.diff(times) <= config.POWER_GAP_MINUTES * 60 * SECOND
            # 先合并块内连续段，避免逐秒区间占用大量内存。
            changes = np.diff(np.r_[False, low, False].astype(np.int8))
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1)
            intervals.extend(zip(times[starts].tolist(), times[ends].tolist()))
            previous_time, previous_value = int(times[-1]), float(values[-1])
    if file_info(path) != source:
        raise RuntimeError(f"低出力分析期间源文件改变：{path}")
    intervals = merge_intervals(intervals)
    with cache.open("wb") as stream:
        pickle.dump(intervals, stream)
    return intervals


def cached_scan(path, ends, width, power_ends, power_width, is_power):
    functions = [window_sums, merge_intervals, display_indices, scan_source]
    fingerprint = "".join(inspect.getsource(function) for function in functions)
    settings = [file_info(path), ends.tolist(), width, power_ends.tolist(), power_width,
                is_power, config.STOP_MINUTES, config.POWER_GAP_MINUTES, config.DISPLAY_BUCKET]
    key = hashlib.sha256((json.dumps(settings) + fingerprint).encode()).hexdigest()
    path_cache = CACHE / f"{key}.pkl"
    if path_cache.exists():
        with path_cache.open("rb") as stream:
            result = pickle.load(stream)
        print(f"  缓存 {path.name} | {result['rows']:,}行", flush=True)
        return result
    result = scan_source(path, ends, width, power_ends, power_width, is_power)
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = path_cache.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path_cache)
    print(f"  完成 {path.name} | {result['rows']:,}行 | 抽查通过", flush=True)
    return result


def unit_statistics(unit):
    selected, workbook, power_path = points(unit)
    sources = [point["path"] for point in selected]
    if power_path not in sources:
        sources.append(power_path)
    extents = [source_extent(path) for path in sources]
    start = min(pair[0] for pair in extents)
    end = max(pair[1] for pair in extents)
    if config.START is not None:
        start = max(start, timestamp(config.START))
    if config.END is not None:
        end = min(end, timestamp(config.END))
    ends, width = window_grid(start, end, config.INTERVAL_WINDOW, config.INTERVAL_STEP)
    power_ends, power_width = window_grid(start, end, config.POWER_WINDOW, config.POWER_STEP)
    results = {}
    print(f"{unit} | {len(selected)}测点 | {len(ends)}窗口", flush=True)
    for path in sources:
        results[path] = cached_scan(path, ends, width, power_ends, power_width, path == power_path)
    rows = [results[point["path"]] for point in selected]
    stops = stop_candidates([row["stops"] for row in rows], len(rows))
    stops = merge_intervals(stops + low_power_stops(power_path))
    power = dict(results[power_path])
    power["power_mean"], power["power_count"] = running_power_mean(
        power_path, power_ends, power_width, stops)
    stops = [[max(a, start), min(b, end)] for a, b in stops if a < end and b > start]
    return {"unit": unit, "points": selected, "workbook": file_info(workbook),
            "workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
            "start": start, "end": end, "ends": ends, "width": width,
            "power_ends": power_ends, "power_width": power_width,
            "rows": rows, "power": power, "stops": stops}


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def metadata(result, figure_type, color_limits):
    settings = {key: value for key, value in vars(config).items() if key.isupper()}
    rows = [{key: row[key] for key in ["source", "rows", "first", "last",
             "window_checks", "power_window_checks"]} for row in result["rows"]]
    info = {"figure": figure_type, "unit": result["unit"], "timezone": "Asia/Shanghai",
            "created": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "config": settings, "sources": rows, "workbook": result["workbook"],
            "workbook_sha256": result["workbook_sha256"],
            "power_source": result["power"]["source"],
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(Path(config.__file__).read_bytes()).hexdigest(),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                         "pyarrow": pa.__version__, "matplotlib": matplotlib.__version__},
            "rules": {"window": "(t-window,t], full windows only",
                      "interval": "right endpoint assignment, predecessor may be outside window",
                      "power_mean": "equal weight per non-stop original record; stop samples excluded from sums/counts; no mean plotted inside stop intervals",
                      "speed": "absolute difference of adjacent non-stop power window means per minute, then trailing SPEED_WINDOW valid-value mean excluding stop times; no average plotted in stops",
                      "stop": "strict-majority >=60min absent/constant OR power below STOP_POWER_MW; low-power state held until next record only within POWER_GAP_MINUTES",
                      "missing": "null means no computable statistic, not zero"},
            "stop_candidates_ns": result["stops"], "color_limits_seconds": color_limits,
            "display_limits": PLOT_LIMITS,
            "display_rule": "pooled seven-unit display extrema and window means; quantile outliers masked only for plotting; interval colors saturate at pooled quantiles",
            "window_ends_ns": result["ends"],
            "sensor_order": [{k: p[k] for k in ["code", "name", "semantic"]} for p in result["points"]]}
    if figure_type == "interval_heatmap":
        info["mean_interval_seconds"] = [row["interval_mean"] for row in result["rows"]]
        info["interval_counts"] = [row["interval_count"] for row in result["rows"]]
    else:
        info["power_window_ends_ns"] = result["power_ends"]
        info["power_mean"] = result["power"]["power_mean"]
        info["power_counts"] = result["power"]["power_count"]
        info["speed_before_average"] = power_speed(result["power_ends"], result["power"]["power_mean"],
                                                   result["power_width"], result["power"]["power_gaps"])
        info["speed"] = average_speed(result["power_ends"], info["speed_before_average"], result["stops"])
    return {"Title": f"{result['unit']} {figure_type}",
            "Description": json.dumps(json_value(info), ensure_ascii=False, allow_nan=False)}


def dates(times):
    return mdates.date2num(np.asarray(times, dtype="datetime64[ns]"))


def average_plot_data(times, values, stops):
    # 停机段可能短于1小时，两个非停机采样点之间也不能连线穿过红区。
    crossings = np.zeros(max(0, len(times) - 1), dtype=bool)
    for start, end in stops:
        crossings |= (times[:-1] < end) & (times[1:] > start)
    positions = np.flatnonzero(crossings) + 1
    x = dates(times)
    return np.insert(x, positions, x[positions]), np.insert(values, positions, np.nan)


def format_axis(axis, result):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.set_xlim(dates([result["start"], result["end"]]))
    axis.set_xlabel("时间（Asia/Shanghai）")


def save_figure(figure, path, result, kind, color_limits):
    figure.savefig(path, metadata=metadata(result, kind, color_limits), dpi=config.DPI)
    # 同一画布渲染到临时PNG供检查，不作为交付物。
    preview = CACHE / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    figure.savefig(preview / f"{path.stem}.png", dpi=100)
    plt.close(figure)
    print(f"  图片 {path.name}", flush=True)


def plot_unit(result, color_limits):
    unit = result["unit"]
    heat = np.array([row["interval_mean"] for row in result["rows"]])
    figure, axis = plt.subplots(figsize=(19, 0.33 * len(heat) + 2.8))
    cmap = LinearSegmentedColormap.from_list("yellow_blue", ["#fff239", "#36b6c5", "#061a67"])
    cmap.set_bad("#dddddd")
    x = dates(result["ends"])
    half_step = pd.Timedelta(config.INTERVAL_STEP).value / SECOND / 86400 / 2
    edges = np.r_[x - half_step, x[-1] + half_step]
    mesh = axis.pcolormesh(edges, np.arange(len(heat) + 1), np.ma.masked_invalid(heat),
                           cmap=cmap, norm=LogNorm(*color_limits, clip=True), rasterized=True)
    axis.set_yticks(np.arange(len(heat)) + 0.5)
    axis.set_yticklabels([f"{p['name']}  |  {p['code']}" for p in result["points"]], fontsize=8)
    axis.invert_yaxis()
    for start, end in result["stops"]:
        axis.axvspan(*dates([start, end]), color="#ed2424", alpha=config.STOP_ALPHA, linewidth=0)
    axis.set_title(f"{unit.upper()}  测点平均记录间隔\n窗口 {config.INTERVAL_WINDOW} · 步长 {config.INTERVAL_STEP}", pad=12)
    format_axis(axis, result)
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.015, extend="both")
    q_low, q_high = config.INTERVAL_COLOR_QUANTILES
    colorbar.set_label(f"平均记录间隔（秒，对数色阶；{q_low:.0%}—{q_high:.0%}分位范围）")
    axis.legend(handles=[Patch(facecolor="#dddddd", label="无法计算"),
                         Patch(facecolor="#ed2424", alpha=config.STOP_ALPHA, label="停机候选")],
                loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=8)
    figure.subplots_adjust(left=0.32, right=0.94, top=0.87, bottom=0.09)
    save_figure(figure, config.OUTPUT_DIR / f"{unit}_interval_heatmap.svg", result, "interval_heatmap", color_limits)

    power = result["power"]
    times = power["display_times"]
    values = power["display_values"]
    keep = (times >= result["start"]) & (times <= result["end"])
    times, values = times[keep], values[keep]
    # 在原始长缺口之后插入NaN；不修改观测值，也不补零。
    breaks = []
    for gap_start, gap_end in power["power_gaps"]:
        position = int(np.searchsorted(times, gap_end))
        if 0 < position < len(times) and times[position - 1] <= gap_start:
            breaks.append(position)
    breaks = np.unique(breaks).astype(int)
    display_x = np.insert(dates(times), breaks, np.nan)
    display_y = np.insert(values, breaks, np.nan)
    power_low, power_high = PLOT_LIMITS["power"]
    display_y[(display_y < power_low) | (display_y > power_high)] = np.nan
    mean = power["power_mean"].copy()
    mean[(mean < power_low) | (mean > power_high)] = np.nan
    figure, axes = plt.subplots(2, 1, figsize=(17, 8), sharex=True)
    axes[0].plot(display_x, display_y, color="#91aebf", linewidth=0.5, label="原始出力（仅绘图压缩）")
    power_x = dates(result["power_ends"])
    mean_x, mean_y = average_plot_data(result["power_ends"], mean, result["stops"])
    axes[0].plot(mean_x, mean_y, color="#135c85", linewidth=1.0, label=f"{config.POWER_WINDOW}滑窗均值")
    axes[0].set_ylabel("出力（MW）")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{unit.upper()}  出力与滑窗均值绝对变化率\n窗口 {config.POWER_WINDOW} · 步长 {config.POWER_STEP}")
    speed = power_speed(result["power_ends"], power["power_mean"], result["power_width"], power["power_gaps"])
    averaged_speed = average_speed(result["power_ends"], speed, result["stops"])
    speed_low, speed_high = PLOT_LIMITS["speed"]
    speed[(speed < speed_low) | (speed > speed_high)] = np.nan
    averaged_speed[(averaged_speed < speed_low) | (averaged_speed > speed_high)] = np.nan
    axes[1].plot(power_x, speed, color="#dfba87", linewidth=0.55, alpha=0.7, label="绝对变化率（滑窗平均前）")
    average_x, average_y = average_plot_data(result["power_ends"], averaged_speed, result["stops"])
    axes[1].plot(average_x, average_y, color="#af5c08", linewidth=1.2, label=f"绝对变化率的{config.SPEED_WINDOW}滑窗均值")
    axes[1].set_ylabel("|ΔP/Δt|（MW/分钟）")
    axes[1].legend(loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.2)
        for start, end in result["stops"]:
            axis.axvspan(*dates([start, end]), color="#ed2424", alpha=config.STOP_ALPHA, linewidth=0)
    power_margin = (power_high - power_low) * 0.04
    axes[0].set_ylim(power_low - power_margin, power_high + power_margin)
    axes[1].set_ylim(0, speed_high * 1.04)
    axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + [
        Patch(facecolor="#ed2424", alpha=config.STOP_ALPHA, label="停机候选")], loc="upper right")
    figure.text(0.5, 0.005, "七台统一纵轴；停机样本不参与平均，停机段不绘制均值；分位范围外曲线点不绘制", ha="center", fontsize=9)
    format_axis(axes[1], result)
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    save_figure(figure, config.OUTPUT_DIR / f"{unit}_power.svg", result, "power", color_limits)


def check():
    """临时合成CSV，核对边界、分块、停机规则及直接滑窗统计。"""
    ends = np.array([3600, 7200, 10800], dtype=np.int64) * SECOND
    width = 3600 * SECOND
    times = np.array([0, 1800, 3600, 7200, 7201, 10800], dtype=np.int64) * SECOND
    values = np.array([1., 1., 1., 2., 2., 2.])
    sums, counts = window_sums(times[1:], np.diff(times) / SECOND, ends, width)
    np.testing.assert_array_equal(counts, [2, 1, 2])
    np.testing.assert_array_equal(sums, [3600, 3600, 3600])
    np.testing.assert_array_equal(window_sums(times, values, ends, width)[1], [2, 1, 2])
    assert stop_candidates([[[0, 10]], []], 2) == []
    assert stop_candidates([[[0, 10]], [[5, 15]], []], 3) == [[5, 10]]
    assert stop_candidates([[[0, 10], [2, 8]], []], 2) == []
    saved_block = config.CSV_BLOCK_BYTES
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "synthetic.csv"
        pd.DataFrame({"timestamp": pd.to_datetime(times), "value": values}).to_csv(path, index=False)
        config.CSV_BLOCK_BYTES = 90
        small = scan_source(path, ends, width, ends, width, True)
        config.CSV_BLOCK_BYTES = 4096
        large = scan_source(path, ends, width, ends, width, True)
        for key in ["interval_mean", "interval_count", "power_mean", "power_count", "display_times", "display_values"]:
            np.testing.assert_allclose(small[key], large[key], equal_nan=True)
        assert small["stops"] == large["stops"] == [[0, 10800 * SECOND]]
        np.testing.assert_allclose(small["interval_mean"], [1800, 3600, 1800])
        np.testing.assert_allclose(small["power_mean"], [1, 2, 2])
        speed = power_speed(ends, np.array([1., 2., 4.]), width, [])
        np.testing.assert_allclose(speed[1:], [1 / 60, 2 / 60])
        assert np.isnan(power_speed(ends, np.array([1., 2., 4.]), width, [[3600 * SECOND, 7201 * SECOND]])).all()
        for bad in [pd.DataFrame({"timestamp": pd.to_datetime([0, 0]), "value": [1, 2]}),
                    pd.DataFrame({"timestamp": pd.to_datetime([0, SECOND]), "value": [1, np.inf]})]:
            bad.to_csv(path, index=False)
            try:
                scan_source(path, ends, width, ends, width, True)
            except ValueError:
                pass
            else:
                raise AssertionError("无效数据未报错")
    config.CSV_BLOCK_BYTES = saved_block
    print("合成检查通过：窗口边界、右端归属、跨块、不变/缺口60分钟、严格多数、变化率和错误处理", flush=True)


def main():
    global PLOT_LIMITS
    check()
    if sys.argv[1:] == ["check"]:
        return
    if sys.argv[1:]:
        raise ValueError("仅支持默认运行或check")
    plt.rcParams.update({"font.family": config.FONT, "axes.unicode_minus": False,
                         "svg.fonttype": "path", "font.size": 10})
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [unit_statistics(unit) for unit in config.UNITS]
    finite = np.concatenate([row["interval_mean"][np.isfinite(row["interval_mean"])]
                             for result in results for row in result["rows"]])
    if len(finite) == 0 or finite.min() <= 0:
        raise ValueError("没有正的平均间隔，无法使用对数色阶")
    low, high = np.quantile(finite, config.INTERVAL_COLOR_QUANTILES).tolist()
    if high == low:
        high = low * 1.01
    power_values = np.concatenate([np.r_[r["power"]["display_values"], r["power"]["power_mean"]] for r in results])
    speed_series = [power_speed(r["power_ends"], r["power"]["power_mean"],
                               r["power_width"], r["power"]["power_gaps"]) for r in results]
    speed_values = np.concatenate(speed_series + [average_speed(r["power_ends"], speed, r["stops"])
                         for r, speed in zip(results, speed_series)])
    PLOT_LIMITS = {
        "power": np.quantile(power_values[np.isfinite(power_values)], config.POWER_DISPLAY_QUANTILES).tolist(),
        "speed": np.quantile(speed_values[np.isfinite(speed_values)], config.SPEED_DISPLAY_QUANTILES).tolist(),
    }
    if PLOT_LIMITS["power"][1] <= PLOT_LIMITS["power"][0] or PLOT_LIMITS["speed"][1] <= 0:
        raise ValueError("显示分位范围退化，请调整显示参数")
    print(f"显示范围：间隔色阶{[low, high]}；曲线{PLOT_LIMITS}", flush=True)
    for result in results:
        plot_unit(result, [low, high])
    print(f"完成：{len(results) * 2}张SVG，缓存位于{CACHE}", flush=True)


if __name__ == "__main__":
    main()
