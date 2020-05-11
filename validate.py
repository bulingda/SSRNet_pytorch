import torch
import time
import os
from SSRNET.Mydata import Datasets
from torch.utils.data import DataLoader
from SSRNET.Net2 import SSRNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 10

TEST_IMAGE_PATH = r"D:\E\document\datas\megaage_asian\test"
LABELPATH = r'D:\E\document\datas\megaage_asian\list'


class Detector:
    def __init__(self, net, label=LABELPATH, image=TEST_IMAGE_PATH, net_param=r"./params/ssrnet0.pt", isCuda=True):
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
        dataloader = DataLoader(datasets1, batch_size=BATCHSIZE, shuffle=True, num_workers=0, drop_last=False)
        running_corrects_3 = 0
        running_corrects_5 = 0
        TP = 0
        best_CA3 = 0.
        best_CA5 = 0.
        best_p = 0.

        for i, (inputs, labels) in enumerate(dataloader):
            if self.isCuda:
                inputs = inputs.to(device)
                label = labels.to(device).float()
            outputs = self.net(inputs)
            TP += sum((outputs - label) ** 2 < 1.0)
            running_corrects_3 += torch.sum(torch.abs(outputs - label) < 3)  # CA 3
            running_corrects_5 += torch.sum(torch.abs(outputs - label) < 5)

        CA_3 = float(running_corrects_3) / len(dataloader.dataset)
        CA_5 = float(running_corrects_5) / len(dataloader.dataset)

        P = float(TP) / float(len(dataloader.dataset))

        running_corrects_3 = 0
        running_corrects_5 = 0
        TP = 0

        end_time = time.time()
        usingtime = end_time - start_time
        usingtime = usingtime / len(dataloader.dataset)
        # print("time of per picture in {:.4f}:".format(usingtime))
        return P, CA_3, CA_5


if __name__ == '__main__':
    # max_value = 0
    # max_index = 0
    # score = []
    net = SSRNet()
    for i in range(0, 150):
        detects = Detector(net, net_param=r"./params/ssrnet{}.pt".format(i))

        P, CA_3, CA_5 = detects.detect()
        score1 = 0.3 * P + 0.3 * CA_3 + 0.4 * CA_5
    #     score.append(score1)
    # # print(score)
    # for index, value in score:
    #     if value > max_value:
    #         max_value = value
    #         max_index = index
    #     elif value == max_value:
    #         max_index = max(max_index, index)
    # print('所属参数：{} Precision:{:.4f} CA_3: {:.4f}, CA_5: {:.4f}'.format(i, P, CA_3, CA_5))
        print("所属轮次：{} 分数：{:.4f} Precision:{:.4f} CA_3: {:.4f}, CA_5: {:.4f}".format(i, score1, P, CA_3, CA_5))