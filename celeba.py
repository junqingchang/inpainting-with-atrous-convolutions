import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random


class CelebA(Dataset):
    def __init__(self, root, train_test='train', resize=(256, 256)):
        self.dataset = torchvision.datasets.CelebA(root, split=train_test, target_type='attr')
        self.transform = MaskImage()
        self.target_transform = transforms.ToTensor()
        if resize:
            self.resize = transforms.Resize(resize)
        else:
            self.resize = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        target, _ = self.dataset[idx]
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