"""集中设置参数。"""
from pathlib import Path

# 填写CSV路径；相对路径以运行命令时的工作目录为基准。
DATA_PATH = "data.csv"
FEATURE_COLUMNS = ["x1", "x2", "x3"]
TARGET_COLUMN = "y"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SEED = 17
DEVICE = "cpu"  # 例如 "cuda:0"；不可用时直接报错
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 40
HIDDEN_SIZE = 32
CHECKPOINT_PATH = Path(__file__).resolve().parent / "best_model.pt"
