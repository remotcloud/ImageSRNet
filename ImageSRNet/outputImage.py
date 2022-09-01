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

if __name__ == "__main__":
    fn = f'CGPIndiva/0bestIndiv.pkl'
    with open(fn,'rb') as f:
        indiv = pickle.load(f)
    batch_size = 60
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

    '''                                         
    将测试数据分成两部分，一部分作为校验数据，一部分作为测试数据。
    校验数据用于检测模型是否过拟合并调整参数，测试数据检验整个模型的工作
    '''
    # 首先，定义下标数组 indices，它相当于对所有test_dataset 中数据的编码
    # 然后，定义下标 indices_val 表示校验集数据的下标，indices_test 表示测试集的下标

    indices = range(len(test_dataset))
    indices_val = indices[: 64]
    indices_test = indices[64:128]
    # 根据下标构造两个数据集的SubsetRandomSampler 来样器，它会对下标进行来样
    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
    sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)
    # 根据两个采样器定义加载器
    # 注意将sampler_val 和sampler_test 分别賦值给了 validation_loader 和 test_loader
    validation_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    sampler=sampler_val)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              sampler=sampler_test)
    idx = 70
    input = test_dataset[idx][0].unsqueeze(0)
    out = indiv(input)

    fig = plt.figure(figsize=(15, 7))
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(out[0][ i, ...].data.numpy())
    plt.colorbar(shrink = 0.5)
    plt.show()
    bestExpression = indiv.get_expressions()
    express_dict ={}
    for i,exp in enumerate(bestExpression):
        express_dict[f'exp({i})'] = str(exp)
    log_dict = {
       "Expression": express_dict,
        "fitness":str(indiv.fitness)
    }
    with open(f'../result/log/result.json', 'w') as f:
        json.dump(log_dict, f, indent=4)


