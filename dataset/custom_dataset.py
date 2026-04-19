import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CIFAR10Dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.samples   = []

        self.classes = sorted(os.listdir(root_dir))

        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, file_name)
                label    = self.classes.index(class_name)
                self.samples.append((img_path, label))

        print(f"Dataset loaded: {len(self.samples)} images, {len(self.classes)} classes")
        print(f"Classes: {self.classes}")


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label








TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])