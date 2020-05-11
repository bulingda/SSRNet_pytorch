from torchvision import transforms
from PIL import Image
# from SSRNET.Net2 import *
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
transfromers = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
#
#
# class Detector:
#     def __init__(self):
#         self.net = SSRNet().to(device)
#         if os.path.exists(r'params/ssrnet76.pt'):
#             self.net.load_state_dict(torch.load(r'params/ssrnet76.pt'))
#             self.net.eval()
#         else:
#             print("没找到模型！！！")
#
#     def detect(self, img):  # 将图片放入网络模型，得到3个特征图，然后对3个特征图进行解析
#         img = transfromers(img)
#         age = self.net(img.to(device))
#         # print(age)
#         return age
#
#
# if __name__ == '__main__':
#     path = r"D:\E\document\datas\megaage_asian\test"
#     for j in range(1, len(os.listdir(path))):
#         img = os.path.join(path, '{}.jpg'.format(j))
#         img = Image.open(img).convert('RGB')
#         test = Detector()
#         print(test.detect(img))
#     path = config.path
#     img = Image.open(path).convert('RGBA')
#     test = Detector()
#     print(test.detect(img))
import numpy as np
import torch
import time
import os
from SSRNET.Mydata import Datasets
from torch.utils.data import DataLoader
from SSRNET.Net2 import SSRNet
import configuration as config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 1

# TEST_IMAGE_PATH = r"D:\E\document\datas\megaage_asian\example6"
# TEST_IMAGE_PATH = config.path
# LABELPATH = r'D:\E\document\datas\megaage_asian\list'


class Detector:
    def __init__(self, net, image, net_param=r"./params/ssrnet0.pt", isCuda=True):
        self.net = net
        if os.path.exists(net_param):
            self.net.load_state_dict(torch.load(net_param))
            self.net.eval()
        else:
            print("没找到模型！！！")
        self.image = Image.open(image).convert("RGB")
        self.isCuda = isCuda
        if self.isCuda:
            self.net.to(device)

    def detect(self):
        start_time = time.time()
        inputs = transfromers(self.image)
        inputs = inputs[np.newaxis, :, :, :]
        if self.isCuda:
            inputs = inputs.to(device)
        outputs = self.net(inputs)
        end_time = time.time()
        usingtime = end_time - start_time
        # print("time of per picture in {:.4f}:".format(usingtime))
        return outputs


def SSRnetMain(imgpath):
    net = SSRNet()
    detects = Detector(net, imgpath, net_param=r"SSRNet/params/ssrnet27.pt")
    x = detects.detect()
    print("age：{}".format(int(x.item())))


if __name__ == '__main__':
    path = r"D:\E\document\datas\megaage_asian\example2\1.jpg"
    SSRnetMain(path)