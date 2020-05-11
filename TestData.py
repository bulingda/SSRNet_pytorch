from torch.utils.data import Dataset
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
transfromers = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


class Datasets(Dataset):
    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2
        self.datasets = []
        self.datasets.extend(open(os.path.join(path1, r'test_label.txt')).readlines())

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        strs1 = self.datasets[index].split()
        # print(strs1[0])
        img = Image.open(os.path.join(self.path2, "{0}".format(strs1[0]))).convert('RGB')
        labels = torch.tensor(int(strs1[1]))
        img = transfromers(img)
        return img, labels


if __name__ == '__main__':
    data = Datasets(r'D:\E\document\datas\megaage_asian\list', r'D:\E\document\datas\megaage_asian\test')
    print(data[0][0].shape)
