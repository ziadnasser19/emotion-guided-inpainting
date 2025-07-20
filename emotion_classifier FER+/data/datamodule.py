from torchvision import transforms
from .dataset import FERPlusDataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch

class FERDataModule:
    def __init__(self, csv_path, batch_size=32, image_size=128, criterion=None, train_transform=None, val_transform=None):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.criterion = criterion
        self.image_size = (image_size, image_size)
        self.emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                            4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

        # Define transforms
        self.train_transform = train_transform or transforms.Compose([
            transforms.Resize((int(self.image_size[0] * 1.1), int(self.image_size[1] * 1.1))),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])

        self.val_transform = val_transform or transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def setup(self):
        self.train_dataset = FERPlusDataset(self.csv_path, usage='Training',
                                            transform=self.train_transform, emotion_map=self.emotion_map)
        self.val_dataset = FERPlusDataset(self.csv_path, usage='PrivateTest',
                                          transform=self.val_transform, emotion_map=self.emotion_map)
        self.test_dataset = FERPlusDataset(self.csv_path, usage='PublicTest',
                                           transform=self.val_transform, emotion_map=self.emotion_map)

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self, num_classes=7):
        if self.criterion == 'mixup':
            mixup_fn = v2.RandomChoice([
                v2.MixUp(alpha=0.4, num_classes=num_classes),
                v2.CutMix(alpha=0.4, num_classes=num_classes)
            ])
            def collate_fn(batch):
                return mixup_fn(*torch.utils.data.default_collate(batch))
        else:
            collate_fn = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def get_class_names(self):
        return [self.emotion_map[i] for i in sorted(self.emotion_map.keys())]

    def get_class_weights(self):
        labels = [label for _, label in self.train_dataset]
        labels_tensor = torch.tensor(labels)
        counts = torch.bincount(labels_tensor, minlength=7)
        total = len(labels_tensor)
        weights = total / (counts.float() * 7)
        return weights