import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import random


class Flowers102(Dataset):
    def __init__(self, root, split='train', resize=(256, 256)):
        self.root = root
        self.split = split
        self.transform = MaskImage()
        self.target_transform = transforms.ToTensor()
        if resize:
            self.resize = transforms.Resize(resize)
        else:
            self.resize = None
        imgs = os.listdir(self.root)
        num_imgs = len(imgs)
        if split == 'train':
            split_len = (0, int(num_imgs*0.7))
        else:
            split_len = (int(num_imgs*0.7), int(num_imgs*0.3))
        split_imgs = imgs[split_len[0]:split_len[1]]
        self.dataset = []
        for i in range(len(split_imgs)):
            self.dataset.append(os.path.join(self.root, split_imgs[i]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        target = Image.open(img_path).convert('RGB')
        if self.resize:
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