import os
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from dataset.custom_dataset import CIFAR10Dataset, TRAIN_TRANSFORM, TEST_TRANSFORM
from torch.utils.data import DataLoader

def organize_data():
    base_dir = "data/unstructure"
    train_src=os.path.join(base_dir, "train")
    label_file=os.path.join(base_dir, "trainLabels.csv")
    train_dst="data/train"
    test_dst="data/test"
    
    df = pd.read_csv(label_file)
    print("Labels loaded! Total images:", len(df))
    print(df.head())

    df=df.sample(frac=1 ,random_state=42).reset_index(drop=True)

    # split data into train and test
    split_index=int(0.8*len(df))
    train_df=df[:split_index]
    test_df=df[split_index:]

    print(f"Train: {len(train_df)} images")
    print(f"Test:  {len(test_df)} images")

    for index,row in train_df.iterrows():
        img_name=str(row['id']) + ".png"
        class_name=row['label']
        src_path=os.path.join(train_src,img_name)
        dst_dir=os.path.join(train_dst,class_name)
        dst_path=os.path.join(dst_dir,img_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src_path, dst_path)

    print("Data organized into train directory.")

    for index,row in test_df.iterrows():
        img_name=str(row['id']) + ".png"
        class_name=row['label']
        src_path=os.path.join(train_src,img_name)
        dst_dir=os.path.join(test_dst,class_name)
        dst_path=os.path.join(dst_dir,img_name)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src_path, dst_path)
    
    print("Test data organized into test directory.")


def get_dataloaders(batch_size=64):

    train_dataset = CIFAR10Dataset(
        root_dir="data/train",
        transform=TRAIN_TRANSFORM
    )

    test_dataset = CIFAR10Dataset(
        root_dir="data/test",
        transform=TEST_TRANSFORM
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, test_loader


def plot_history(history, save_dir="plots"):

    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot 1 - Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", color="steelblue", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epochs vs Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()
    print(f"Loss plot saved -> {os.path.join(save_dir, 'loss.png')}")

    # Plot 2 - Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", color="darkorange", label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Epochs vs Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy.png"))
    plt.close()
    print(f"Accuracy plot saved -> {os.path.join(save_dir, 'accuracy.png')}")


def save_results(results_list):

    os.makedirs("results", exist_ok=True)

    csv_path = os.path.join("results", "results.csv")

    with open(csv_path, "w") as f:
        f.write("epochs,optimizer,lr,batch_size,train_acc,test_acc,test_loss\n")
        for r in results_list:
            f.write(f"{r['epochs']},{r['optimizer']},{r['lr']},"
                    f"{r['batch_size']},{r['train_acc']},"
                    f"{r['test_acc']},{r['test_loss']}\n")

    print("Results saved -> results/results.csv")