import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import random

import matplotlib.pyplot as plt


class IndoorSceneRecognition(Dataset):
    def __init__(self, root, train_test='train', resize=(256, 256)):
        self.root = root
        self.train_test = train_test
        self.transform = MaskImage()
        self.target_transform = transforms.ToTensor()
        self.resize = transforms.Resize(resize)

        if self.train_test == 'train':
            self.ref_file = os.path.join(self.root, 'TrainImages.txt')
        else:
            self.ref_file = os.path.join(self.root, 'TestImages.txt')
        self.images_path = os.path.join(self.root, 'Images')
        self.dataset = []
        with open(self.ref_file, 'r') as f:
            for line in f:
                img_path = os.path.join(self.images_path, line.strip())
                self.dataset.append(img_path)
                
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        target = Image.open(img_path).convert('RGB')
        target = self.resize(target)
        data = self.transform(target)
        target = self.target_transform(target)
        return data, target


class MaskImage(object):
    def __init__(self, percentage=0.2):
        self.to_tensor = transforms.ToTensor()
        self.percentage = percentage

    def __call__(self, sample):
        img_tensor = self.to_tensor(sample)
        h, w = img_tensor.shape[1:]
        h_to_affect = int(h*self.percentage)
        w_to_affect = int(w*self.percentage)
        starting_h = random.randint(0, h-h_to_affect-1)
        starting_w = random.randint(0, w-w_to_affect-1)
        img_tensor[:, starting_h:starting_h+h_to_affect, starting_w:starting_w+w_to_affect] = 0
        
        return img_tensor