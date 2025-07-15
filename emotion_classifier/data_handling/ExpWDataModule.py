from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms

class ExpWDataModule:
    def __init__(self, label_file, img_dir, batch_size=32, val_split=0.15, test_split=0.15, random_state = rand_state):
        self.label_file = label_file
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split

        # Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def setup(self):
        base_dataset = ExpWDataset(
            label_file=self.label_file,
            img_dir=self.img_dir,
            transform=self.eval_transform  # set default, override per split
        )

        # Split dataset indices
        total_size = len(base_dataset)
        indices = list(range(total_size))

        val_size = int(self.val_split * total_size)
        test_size = int(self.test_split * total_size)

        train_idx, tmp_idx = train_test_split(indices, test_size=val_size + test_size, random_state=rand_state)
        val_idx, test_idx = train_test_split(tmp_idx, test_size=test_size, random_state=rand_state)

        # Create subsets
        self.train_dataset = Subset(base_dataset, train_idx)
        self.val_dataset = Subset(base_dataset, val_idx)
        self.test_dataset = Subset(base_dataset, test_idx)

        # Assign transforms
        self.train_dataset.dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.eval_transform
        self.test_dataset.dataset.transform = self.eval_transform

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
