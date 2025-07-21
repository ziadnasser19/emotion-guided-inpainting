import os
import random
import cv2
from PIL import Image

from torch.utils.data import Dataset


class TripletVGGFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        """
        Args:
            root_dir (str): Path to 'train/' or 'val/' directory.
            transform: torchvision transforms to apply.
            max_samples (int): Length of the dataset (default: 10000).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.max_samples = max_samples or 10000

        self.class_dirs = [d for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))]

        self.image_paths = {
            cls: [os.path.join(root_dir, cls, f)
                  for f in os.listdir(os.path.join(root_dir, cls))
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for cls in self.class_dirs
        }

        # Keep only classes with >=2 images (needed for triplets)
        self.class_dirs = [cls for cls in self.class_dirs if len(self.image_paths[cls]) >= 2]

    def __len__(self):
        return self.max_samples

    def _load_gray_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_pil = Image.fromarray(gray, mode='L')
        return gray_pil

    def __getitem__(self, idx):
        while True:
            try:
                anchor_cls = random.choice(self.class_dirs)
                anchor_img_path, positive_img_path = random.sample(self.image_paths[anchor_cls], 2)

                negative_cls = random.choice(self.class_dirs)
                while negative_cls == anchor_cls:
                    negative_cls = random.choice(self.class_dirs)
                negative_img_path = random.choice(self.image_paths[negative_cls])

                anchor = self._load_gray_image(anchor_img_path)
                positive = self._load_gray_image(positive_img_path)
                negative = self._load_gray_image(negative_img_path)

                if self.transform:
                    anchor = self.transform(anchor)
                    positive = self.transform(positive)
                    negative = self.transform(negative)

                return anchor, positive, negative
            except Exception as e:
                print(f"⚠️ Skipping triplet: {str(e)}")

# Example Usage

# from torchvision import transforms
# from torch.utils.data import DataLoader
#
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])
#
# train_dataset = TripletVGGFaceDataset(
#     root_dir='/kaggle/input/vggface2/train',
#     transform=transform
# )