import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader


class FERPlusDataset(Dataset):
    def __init__(self, csv_path, usage='Training', transform=None, emotion_map=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['Usage'] == usage].reset_index(drop=True)
        self.transform = transform
        self.emotion_map = emotion_map or {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                                           4: 'Sad', 5: 'Surprise', 6: 'Neutral', 7: 'Contempt'}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get label and convert to integer
        label = int(self.data.iloc[idx]['emotion'])
        
        # Convert pixel string to numpy array then to PIL image
        pixel_str = self.data.iloc[idx]['pixels']
        pixels = np.array(list(map(int, pixel_str.split())), dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(pixels)  # single channel image

        if self.transform:
            image = self.transform(image)

        return image, label