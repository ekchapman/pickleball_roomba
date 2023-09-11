import os

from PIL import Image
import cv2 as cv
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.io import read_image, ImageReadMode

class PballTennisDataset(Dataset):
    def __init__(self, dir, transform=None, target_transform=None):

        self.paths = []
        for root, dirs, files in os.walk(dir):
            for filename in files:
                self.paths.append(os.path.join(root, filename))

        self.labels = [0 if "pickleball" in path else 1 for path in self.paths]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # image = read_image(self.paths[idx], ImageReadMode.RGB)
        # image = cv.imread(self.paths[idx])
        # image = Image.open(self.paths[idx]).convert("RGB")
        # image = np.array(image)
        # assert image.shape[2] == 3 or image.shape[2] == 1
        # if image.shape[2] == 1:
        #     image = cv.merge((image, image, image))
        # assert image.shape[2] == 3
        # print(image.shape)
        image = read_image(self.paths[idx], ImageReadMode.RGB).type(torch.FloatTensor) / 255
        if image.shape[0] == 1:
            image = image.expand(3, *image.shape[1:])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
