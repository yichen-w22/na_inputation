"""预测并计算误差。"""
import torch

import config


def predict(model, loader):
    model.eval()
    prediction_batches = []
    target_batches = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(config.DEVICE)
            predictions = model(features)

            prediction_batches.append(predictions.cpu())
            target_batches.append(targets.cpu())

    predictions = torch.cat(prediction_batches)
    targets = torch.cat(target_batches)
    return predictions, targets


def evaluate(predictions, targets):

    errors = predictions.double() - targets.double()
    mae = errors.abs().mean().item()
    rmse = errors.square().mean().sqrt().item()

    metrics = {"MAE": mae, "RMSE": rmse}
    print(f"测试集 | MAE {mae:.6f} | RMSE {rmse:.6f}")
    return metrics
