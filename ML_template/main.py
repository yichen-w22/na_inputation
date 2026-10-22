"""运行：python dl_template/main.py；依赖：torch、numpy、pandas。"""
from data import prepare_data
from model import build_model
from train import train_model
from evaluate import predict, evaluate


def main():
    train_loader, val_loader, test_loader, preprocessing = prepare_data()
    model = build_model()
    model = train_model(model, train_loader, val_loader)
    predictions, targets = predict(model, test_loader)
    evaluate(predictions, targets)


if __name__ == "__main__":
    main()
