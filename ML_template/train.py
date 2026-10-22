"""训练模型，并恢复验证损失最低的权重。"""
import torch
from torch import nn

import config


def train_model(model, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_loss = float("inf")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0

        for features, targets in train_loader:
            features = features.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(targets)
            train_sample_count += len(targets)

        train_loss = train_loss_sum / train_sample_count

        model.eval()
        val_loss_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(config.DEVICE)
                targets = targets.to(config.DEVICE)

                predictions = model(features)
                if predictions.shape != targets.shape:
                    raise ValueError("验证阶段：预测与目标形状不一致")

                loss = criterion(predictions, targets)
                if not torch.isfinite(loss):
                    raise ValueError("验证阶段：损失不是有限值")

                val_loss_sum += loss.item() * len(targets)
                val_sample_count += len(targets)

        val_loss = val_loss_sum / val_sample_count

        print(f"轮次 {epoch + 1:03d} | 训练损失 {train_loss:.6f} | 验证损失 {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)

    best_weights = torch.load(
        config.CHECKPOINT_PATH,
        map_location=config.DEVICE,
        weights_only=True,
    )
    model.load_state_dict(best_weights)
    return model
