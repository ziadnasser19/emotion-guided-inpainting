import pandas as pd
import numpy as np
from PIL.Image import Image
from torch.utils.data import Dataset


class FERPlusDataset(Dataset):
    def __init__(self, csv_path, usage='Training', transform=None, emotion_map=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['Usage'] == usage].reset_index(drop=True)

        # Drop 'Angry' (0) and 'Disgust' (1)
        self.data = self.data[~self.data['emotion'].isin([0, 1])].reset_index(drop=True)

        self.transform = transform

        # Remap remaining labels to 0...n-1
        original_to_new = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
        self.data['emotion'] = self.data['emotion'].map(original_to_new)

        self.emotion_map = {
            0: 'Fear', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Neutral'
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = int(self.data.iloc[idx]['emotion'])

        pixel_str = self.data.iloc[idx]['pixels']
        pixels = np.array(list(map(int, pixel_str.split())), dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(pixels)

        if self.transform:
            image = self.transform(image)

        return image, label