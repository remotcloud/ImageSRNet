import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# from ImageSRNet.image_functions import default_functions
# from ImageSRNet.sr_objs import ImageCGP
from image_functions import default_functions
from sr_objs import ImageCGP
import json
import _pickle as pickle

from Minist_imageCGP import ConvNet
class resultDeal(object):
    def __init__(self,bestIndividual):
        self.bestIndividual = bestIndividual
        self.dataSet = self.__getDataPickle()
    def __getDataPickle(self):
        fileDir = f'data/mnistData.pkl'
        with open(fileDir,'rb') as f:
            nnMnistData = pickle.load(f)
        return nnMnistData
    def __saveDataPickle(self,batch_size = 640):
        train_dataset = dsets.MNIST(root='./data',  # 文件存放路径
                                    train=True,  # 提取训练集
                                    # 将图像转化为 Tensor，在加载數据时，就可以对图像做预处理
                                    transform=transforms.ToTensor(),
                                    download=True)  # 当找不到文件的时候，自动下載
        # 加载测试数据集
        test_dataset = dsets.MNIST(root='./data',
                                   train=False,
                                   transform=transforms.ToTensor())
        # 训练数据集的加载器，自动将数据切分成批，顺序随机打乱
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        model = ConvNet()
        fname = 'ministCnn'
        model.load_state_dict(torch.load(f"ModelSave/{fname}.pth"))

        if torch.cuda.is_available():
            model = model.cuda()


        for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
            if torch.cuda.is_available():
                data = data.cuda()
                nndata = model(data)
                break
            else:
                nndata = model(data)
                break
        fn = f'data/mnistData.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(nndata,f)
        return nndata
    def getResultImageComp(self,idx):
        input = self.dataSet[2][idx].unsqueeze(0)
        out = indiv(input)
        fig = plt.figure(figsize=(10, 7))
        fig.suptitle('input')
        for i in range(input.shape[1]):
            plt.subplot(2, math.ceil( input.shape[1]/ 2), i + 1)
            plt.imshow(input[0][i, ...].cpu().data.numpy())

        fig = plt.figure(figsize=(10, 7))
        fig.suptitle('out')
        for i in range(out.shape[1]):
            plt.subplot(2, math.ceil(out.shape[1]/2), i + 1)
            plt.imshow(out[0][i, ...].cpu().data.numpy())

        # plt.colorbar(shrink=0.5)

        # plt.show()
        fig = plt.figure(figsize=(10, 7))
        fig.suptitle('origin out')
        for i in range(out.shape[1]):
            plt.subplot(2, math.ceil(out.shape[1]/2), i + 1)
            plt.imshow(self.dataSet[3][idx][i, ...].cpu().data.numpy())
        plt.show()


    def getResultImageComp2(self,number):
        """
        画图，得到卷积层2预测输出和实际输出的关系
        :param number: input的数量
        :return:
        """
        fig = plt.figure(figsize=(10, 7))
        fig.suptitle('input')
        row = math.ceil(math.sqrt(number))
        col = math.ceil(number / row)
        for imageId in range(number):
            for i in range(col):
                plt.subplot(row, col,imageId+1 )
                plt.imshow(self.dataSet[2][imageId].unsqueeze(0)[0][0, ...].cpu().data.numpy())

        plt.show()
    def getResultJson(self):
        bestExpression = self.bestIndividual.get_expressions()
        express_dict = {}
        for i, exp in enumerate(bestExpression):
            express_dict[f'exp({i})'] = str(exp)
        log_dict = {
            "Expression": express_dict,
            "fitness": str(indiv.fitness)
        }
        with open(f'../result/log/result.json', 'w') as f:
            json.dump(log_dict, f, indent=4)
        print("Save json ok!")
if __name__ == "__main__":
    fn = f'CGPIndiva/3bestIndiv.pkl'
    with open(fn,'rb') as f:
        indiv = pickle.load(f)
    result = resultDeal(indiv)

    result.getResultImageComp2(81)
    result.getResultJson()


