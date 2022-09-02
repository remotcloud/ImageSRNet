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
class resultDeal(object):
    def __init__(self,bestIndividual):
        self.bestIndividual = bestIndividual
        self.dataSet = self.__getDataSet()
    def __getDataSet(self):
        train_dataset = dsets.MNIST(root='./data',  # 文件存放路径
                                    train=True,  # 提取训练集
                                    # 将图像转化为 Tensor，在加载數据时，就可以对图像做预处理
                                    transform=transforms.ToTensor(),
                                    download=True)  # 当找不到文件的时候，自动下載
        # 加载测试数据集
        test_dataset = dsets.MNIST(root='./data',
                                   train=False,
                                   transform=transforms.ToTensor())

        return [train_dataset,test_dataset]
    def getResultImageComp(self,idx):
        input = self.dataSet[1][idx][0].unsqueeze(0)
        out = indiv(input)
        fig = plt.figure(figsize=(15, 7))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.imshow(out[0][i, ...].data.numpy())
        plt.colorbar(shrink=0.5)
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
    fn = f'CGPIndiva/0bestIndiv.pkl'
    with open(fn,'rb') as f:
        indiv = pickle.load(f)
    result = resultDeal(indiv)
    result.getResultImageComp(70)
    result.getResultJson()


