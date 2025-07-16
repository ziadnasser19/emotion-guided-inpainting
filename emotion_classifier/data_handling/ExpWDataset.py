import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ExpWDataset(Dataset):
    class_names = ['Happy', 'Sad', 'Angry', 'Surprised', 'Fear', 'Disgust', 'Neutral'] 

    def __init__(self, label_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 8:
                    continue  # skip malformed lines
                img_name, _, top, left, right, bottom, _, label = parts
                bbox = (int(left), int(top), int(right), int(bottom))
                self.samples.append((img_name, bbox, int(label)))

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [label for _, _, label in self.samples]
    

    def __getitem__(self, idx):
        img_name, bbox, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
            face = image.crop(bbox)  # Crop face using bounding box
        except Exception as e:
            print(f"Error loading {img_path} with bbox {bbox}: {e}")
            return torch.zeros(3, 224, 224), -1


        if self.transform:
            face = self.transform(face)

        return face, label
