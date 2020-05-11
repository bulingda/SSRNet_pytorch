import torch
import numpy as np
import time
import os
from SSRNET.Mydata import Datasets
from torch.utils.data import DataLoader
from SSRNET.Net2 import SSRNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_IMAGE_PATH = r"D:\E\document\datas\megaage_asian\example2"
LABELPATH = r'D:\E\document\datas\megaage_asian\list'


class Detector:
    def __init__(self, net, label, image=TEST_IMAGE_PATH, net_param=r"SSRNet/params/ssrnet0.pt", isCuda=True):
        self.net = net
        if os.path.exists(net_param):
            self.net.load_state_dict(torch.load(net_param))
            self.net.eval()
        else:
            print("没找到模型！！！")
        self.label_path = label
        self.image_path = image
        self.isCuda = isCuda
        if self.isCuda:
            self.net.to(device)

    def detect(self):
        start_time = time.time()
        datasets1 = Datasets(self.label_path, self.image_path, train=False)
        dataloader = DataLoader(datasets1, batch_size=len(datasets1)-1, shuffle=True, num_workers=0, drop_last=False)
        age = []
        for i, (inputs, labels) in enumerate(dataloader):
            if self.isCuda:
                inputs = inputs.to(device)
                # print(inputs.size())
                label = labels.to(device).float()
            outputs = self.net(inputs)
            age.extend(outputs.cpu().detach().numpy())
        end_time = time.time()
        usingtime = end_time - start_time
        print("time of per picture in {:.4f}:".format(usingtime))
        return age


def SSRnetMain(label):
    net = SSRNet()
    detects = Detector(net, label, net_param=r"SSRNet/params/ssrnet27.pt")
    age = detects.detect()
    # print(age)
    x = np.median(age)
    print("age：{}".format(int(x)))


if __name__ == '__main__':
    SSRnetMain(LABELPATH)
