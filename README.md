# Image Classifier (CIFAR-10) using PyTorch

This project implements a complete image classification pipeline for CIFAR-10 using a custom dataset class, CNN model, training/evaluation functions, and hyperparameter tuning.

## Project Structure

```text
image-classifier/
	dataset/
		custom_dataset.py
	models/
		model.py
	train/
		train.py
	utils/
		helper.py
	plots/
	results/
	data/
		train/
		test/
		Unstructure/
	main.py
	requirements.txt
	README.md
```

## Features

- Custom `CIFAR10Dataset` class with `__len__` and `__getitem__`
- Train and test `DataLoader` creation
- CNN with 4 convolutional blocks + fully connected classifier
- Training with `CrossEntropyLoss`
- Optimizer experiments with SGD and Adam
- Hyperparameter tuning across 24 combinations
- Automatic best-model saving (`results/best_model.pth`)
- Plot generation per experiment (`plots/<experiment_name>/`)
- CSV export of all experiment results (`results/results.csv`)

## Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

### Option 1: Use already organized data (recommended)

Place images in:

- `data/train/<class_name>/...`
- `data/test/<class_name>/...`

Then run `main.py` directly.

### Option 2: Organize from Kaggle-style unstructured data

If using raw files (`trainLabels.csv` + flat image folder), place them in:

- `data/unstructure/trainLabels.csv`
- `data/unstructure/train/`

Then uncomment the `organize_data()` call in `main.py` and run once.

Note: On Linux/Colab, paths are case-sensitive. Ensure folder name is exactly `data/unstructure` if you use organizer mode.

## Run Experiments

```bash
python main.py
```

The script runs all combinations of:

- Epochs: `5, 10, 20`
- Optimizer: `SGD, Adam`
- Learning rate: `0.01, 0.001`
- Batch size: `32, 64`

Total experiments: `3 x 2 x 2 x 2 = 24`

## Outputs

- `results/results.csv`: all experiment metrics
- `results/best_model.pth`: best model by highest test accuracy in current run
- `plots/<experiment_name>/loss.png`
- `plots/<experiment_name>/accuracy.png`

## Best Observed Configuration

From completed experiments, the best configuration was:

- Optimizer: `SGD`
- Epochs: `20`
- Learning Rate: `0.01`
- Batch Size: `64`
- Train Accuracy: `90.88%`
- Test Accuracy: `83.27%`

## Colab Quick Notes

- Create a new notebook and enable GPU runtime.
- Clone this repository.
- Install dependencies with `pip install -r requirements.txt`.
- Mount Google Drive and unzip dataset into `data/train` and `data/test`.
- Run `python main.py`.

