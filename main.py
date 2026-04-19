from itertools import product
import os
from utils.helper import get_dataloaders, plot_history, save_results ,organize_data
from models.model import CNNClassifier
import torch
import torch.optim as optim
from train.train import train_model, save_model, evaluate_model


EXPERIMENTS = list(product(
    [5, 10, 20],
    ["SGD", "Adam"],
    [0.01, 0.001],
    [32, 64]
))


def run_experiment(epochs, opt_name, lr, batch_size, device):
    print("\n" + "=" * 60)
    print(f"Experiment: epochs={epochs}, optimizer={opt_name}, lr={lr}, batch={batch_size}")
    print("=" * 60)

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = CNNClassifier(num_classes=10)

    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        epochs=epochs,
        optimizer=optimizer,
        device=device
    )

    test_acc, test_loss = evaluate_model(model, test_loader, device)

    exp_name = f"{opt_name}_ep{epochs}_lr{lr}_bs{batch_size}"
    plot_history(history, save_dir=os.path.join("plots", exp_name))

    result = {
        "epochs": epochs,
        "optimizer": opt_name,
        "lr": lr,
        "batch_size": batch_size,
        "train_acc": round(history["train_acc"][-1], 2),
        "test_acc": round(test_acc, 2),
        "test_loss": round(test_loss, 4)
    }

    return result, model


def main():

    organize_data();
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_results = []
    best_acc = -1.0
    best_result = None

    for (epochs, opt_name, lr, batch_size) in EXPERIMENTS:
        result, model = run_experiment(epochs, opt_name, lr, batch_size, device)
        all_results.append(result)

        if result["test_acc"] > best_acc:
            best_acc = result["test_acc"]
            best_result = result
            save_model(model, file_name="best_model.pth")

        print("\nResults so far:")
        print(f"{'Epochs':>8} {'Opt':>6} {'LR':>8} {'BS':>4} {'TrainAcc':>10} {'TestAcc':>9} {'TestLoss':>10}")
        for r in all_results:
            print(
                f"{r['epochs']:>8} {r['optimizer']:>6} {r['lr']:>8} {r['batch_size']:>4} "
                f"{r['train_acc']:>10.2f} {r['test_acc']:>9.2f} {r['test_loss']:>10.4f}"
            )

    save_results(all_results)

    print("\nAll experiments complete")
    print(f"Best test accuracy: {best_result['test_acc']:.2f}%")
    print("Best configuration:")
    print(best_result)
    print("Best model saved at results/best_model.pth")


if __name__ == "__main__":
    main()