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


# path2 = r'D:\E\document\datas\face_age_dataset\test'
path1 = r'D:\E\document\datas\face_age\train'


class Datasets(Dataset):

    def __init__(self, path1):
        self.images = []
        for agelist in os.listdir(path1):
            for image in os.listdir(os.path.join(path1, agelist)):
                self.images.append(os.path.join(path1, agelist, image))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # print(self.images)
        strs1 = self.images[index].split('\\')[-2]
        # strs2 = self.images[index].split('\\')[-2]
        # print(strs1)
        img = Image.open(os.path.join(self.images[index])).convert('RGB')
        labels = torch.tensor(int(strs1))
        img = transfromers(img)
        return img, labels


if __name__ == '__main__':
    data = Datasets(path1)
    print(data[0])
