import os
import torch
import time
from SSRNET.Net1 import SSRNet
import torch.optim as optim
import torch.nn as nn
from SSRNET.Facedata import Datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVEPATH = r'D:\E\SomeProjects\params\ssrnet\ssrnet'
IMAGEPATH = r'D:\E\document\datas\megaage_asian\train'
LABELPATH = r'D:\E\document\datas\megaage_asian\list'
BATCHSIZE = 32
WEIGHTDECAY = 1e-4
learning_rate = 0.002  # 改成0.002
path1 = r'D:\E\document\datas\face_age\train'


class Trainer:
    def __init__(self, net, save_path, image_path, isCuda=True):  # , label_path, image_path
        self.net = net
        self.save_path = save_path
        # self.label_path = label_path
        self.image_path = image_path
        self.isCuda = isCuda

        self.loss_fn = nn.L1Loss()

        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=WEIGHTDECAY, lr=learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        # if os.path.exists(self.save_path):
        #     self.net.load_state_dict(torch.load(self.save_path))
        #     self.net.eval()

        if self.isCuda:
            self.net.to(device)

    def train(self):
        datasets1 = Datasets(self.image_path)  # self.label_path, self.image_path, train=True
        dataloader = DataLoader(datasets1, batch_size=BATCHSIZE, shuffle=True, num_workers=0, drop_last=False)
        running_loss = 0.0
        running_corrects_3 = 0
        running_corrects_5 = 0
        TP = 0
        start_time = time.time()
        for epoch in range(150):
            self.net.train()
            for i, (inputs, labels) in enumerate(dataloader):
                if self.isCuda:
                    inputs = inputs.to(device)
                    label = labels.to(device).float()
                # inputs = inputs.view(inputs.size(0), 3, inputs.size(2), inputs.size(2))
                outputs = self.net(inputs)
                # flop, param = profile(self.net, inputs=(inputs,))
                # flops, params = clever_format([flop, param], "%.3f")
                # print(outputs.shape)
                # print(label.shape)
                loss = self.loss_fn(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                # print(outputs)
                # print(labels)
                TP += sum((outputs - label)**2 < 1.0)  #  .squeeze(1)
                running_corrects_3 += torch.sum(torch.abs(outputs - label) < 3)  # CA 3
                running_corrects_5 += torch.sum(torch.abs(outputs - label) < 5)
                # print(TP)
                # running_corrects_3 += torch.sum(torch.abs(outputs - labels) < 3)  # CA 3
                # running_corrects_5 += torch.sum(torch.abs(outputs - labels) < 5)
                if i % 10 == 9:
                    # print(outputs)
                    print("epoch:", epoch, "batch:", i+1, "loss:", loss.cpu().data.numpy(), 'generator parameters:', sum(param.numel() for param in self.net.parameters()))  # , "flops:", flops, "params:", params

                writer.add_scalar('loss', loss.item(), global_step=1)
            self.lr_scheduler.step(epoch)
            writer.close()
            torch.save(self.net.state_dict(), os.path.join(self.save_path, 'ssrnet{}.pt'.format(epoch)))
            epoch_loss = running_loss / len(dataloader.dataset)
            P = float(TP) / float(len(dataloader.dataset))
            CA_3 = float(running_corrects_3) / len(dataloader.dataset)
            CA_5 = float(running_corrects_5) / len(dataloader.dataset)
            print('Precision:{:.4f}, CA_3: {:.4f}, CA_5: {:.4f}, 这轮损失为Loss:{:.4f}, 第{}轮训练完了'.format(P, CA_3, CA_5, epoch_loss, epoch))
            running_corrects_3 = 0
            running_corrects_5 = 0
            TP = 0
            running_loss = 0.0
            # CA_3 = running_corrects_3.double() / len(dataloader)
            # CA_5 = running_corrects_5.double() / len(dataloader)
            # self.lr_scheduler.step(epoch)
        end_time = time.time()
        usingtime = end_time - start_time
        usingtime = usingtime / len(dataloader.dataset)
        print("time of per picture in {:.4f}:".format(usingtime))


if __name__ == '__main__':
    net = SSRNet()  # stage_num, lambda_local, lambda_d
    trainer = Trainer(net, SAVEPATH, path1)  # , LABELPATH, IMAGEPATH
    trainer.train()









