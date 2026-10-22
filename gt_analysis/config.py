"""任务一参数：修改后运行 python gt_analysis/run.py。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
UNITS = ["gard_gt1", "gard_gt2", "jqrd_gt1", "jqrd_gt2",
         "jxrd_gt1", "jxrd_gt2", "jxrd_gt3"]

# None表示使用全部入选测点；也可填写测点编码列表，顺序仍按点表。
SENSORS = None
START = None
END = None

INTERVAL_WINDOW = "7d"
INTERVAL_STEP = "1h"
POWER_WINDOW = "7d"
POWER_STEP = "1h"
SPEED_WINDOW = "7d"
STOP_MINUTES = 60
STOP_POWER_MW = 10
POWER_GAP_MINUTES = 60
DISPLAY_BUCKET = "15min"

CSV_BLOCK_BYTES = 32 * 1024 * 1024
DPI = 150
FONT = "Microsoft YaHei"

# 只控制显示，不删除记录或改变统计。七台共用同一套范围。
INTERVAL_COLOR_QUANTILES = (0.02, 0.98)
POWER_DISPLAY_QUANTILES = (0.005, 0.995)
SPEED_DISPLAY_QUANTILES = (0.0, 0.995)
STOP_ALPHA = 0.18
