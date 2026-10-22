"""数据模块：本示例采用随机划分；时序任务须在这里改为时间划分。"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import config


def prepare_data():
    path = Path(config.DATA_PATH)
    frame = pd.read_csv(path)

    selected_columns = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]

    selected_frame = frame[selected_columns]
    values = selected_frame.to_numpy(dtype=np.float32)

    sample_count = len(values)
    rng = np.random.default_rng(config.SEED)
    shuffled_indices = rng.permutation(sample_count)
    shuffled_values = values[shuffled_indices]

    train_count = int(sample_count * config.TRAIN_RATIO)
    val_count = int(sample_count * config.VAL_RATIO)
    train_end = train_count
    val_end = train_count + val_count

    train_values = shuffled_values[:train_end]
    val_values = shuffled_values[train_end:val_end]
    test_values = shuffled_values[val_end:]

    train_features = train_values[:, :-1]
    train_targets = train_values[:, -1:]
    val_features = val_values[:, :-1]
    val_targets = val_values[:, -1:]
    test_features = test_values[:, :-1]
    test_targets = test_values[:, -1:]

    mean = train_features.mean(axis=0, dtype=np.float64)
    std = train_features.std(axis=0, dtype=np.float64)

    train_features = ((train_features - mean) / std).astype(np.float32)
    val_features = ((val_features - mean) / std).astype(np.float32)
    test_features = ((test_features - mean) / std).astype(np.float32)

    train_features_tensor = torch.from_numpy(train_features)
    train_targets_tensor = torch.from_numpy(train_targets.copy())
    val_features_tensor = torch.from_numpy(val_features)
    val_targets_tensor = torch.from_numpy(val_targets.copy())
    test_features_tensor = torch.from_numpy(test_features)
    test_targets_tensor = torch.from_numpy(test_targets.copy())

    # TensorDataset把每条输入和对应目标配成一对。
    train_dataset = TensorDataset(train_features_tensor, train_targets_tensor)
    val_dataset = TensorDataset(val_features_tensor, val_targets_tensor)
    test_dataset = TensorDataset(test_features_tensor, test_targets_tensor)

    # 8. DataLoader每次取出一个批次。只有训练集在每轮打乱顺序。
    train_generator = torch.Generator().manual_seed(config.SEED)
    val_generator = torch.Generator().manual_seed(config.SEED)
    test_generator = torch.Generator().manual_seed(config.SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        generator=val_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        generator=test_generator,
    )

    # 9. 返回三个加载器，以及将来处理新数据时要复用的标准化参数。
    preprocessing = {"mean": mean, "std": std}
    return train_loader, val_loader, test_loader, preprocessing
