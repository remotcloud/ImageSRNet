import math
import os
import random

from utils import SRNetDataset, save_checkpoint
from minis_sr_utils import evolutionNAddLamdaCommon, get_data, DataSetManager, parallel_save_result
from sr_nets import *
import torch
import torch.nn as nn
from Minist_imageCGP import ConvNet  # loss函数希望最后能够通用化
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter

class Minist_Data(Dataset):
    def __init__(self, data_dir, data_list):
        super(Minist_Data, self).__init__()
        self.data_dir = data_dir
        self.train = data_list
        self.train_loader = data_list[0]
        self.validation_loader = data_list[1]
        self.test_loader = data_list[2]


class Minist_SR(nn.Module):
    def __init__(self, args, image_cgp_net=None, cgp_net=None):
        super(Minist_SR, self).__init__()
        self.args = args
        # 自定义训练参数
        self.image_cgp_net = image_cgp_net
        self.cgpnet = cgp_net
        self.channels_list = args.channels_list
        self.input_sizes = args.input_sizes
        self.output_sizes = args.output_sizes

        self.fitness = None

        self.params = args.params
        if self.image_cgp_net is None:
            self.image_cgp_net = ImageCGPNet(self.channels_list, self.input_sizes, self.output_sizes, self.params)

    def forward(self, input):
        x = self.image_cgp_net(input)
        return x[-1]

    def mutate(self, prob):
        self.image_cgp_net = self.image_cgp_net.mutate(prob)
        return self
    def get_expressions(self,var_names=None, active_pool_name=None):
        return self.image_cgp_net.__repr__(var_names, active_pool_name)


class MnistSRWithWeight(nn.Module):
    def __init__(self, mnist_sr: Minist_SR):
        super(MnistSRWithWeight, self).__init__()
        weight = torch.Tensor(1)
        self.weight = torch.nn.Parameter(data = weight, requires_grad=True)
        self.weight.data.uniform_(-1, 1)
        self.register_parameter("weight", self.weight)
        self.mnist_sr = mnist_sr

    def forward(self, input):
        x = self.weight * self.mnist_sr(input)
        return x


class Args(object):
    def __init__(self):
        super(Args, self).__init__()


if __name__ == '__main__':
    run_num = 1    # number of run the program
    run_times = 1  # times of run the program
    print(SRNetDataset)
    sr_data = Minist_Data("./data", get_data(60))

    model = ConvNet()
    fname = 'ministCnn'
    model.load_state_dict(torch.load(f"ModelSave/{fname}.pth"))

    if torch.cuda.is_available():
        model = model.cuda()
    image_size = 28  # 图像的总尺寸为 28x28
    num_classes = 10  # 标签的种类数
    num_epochs = 20  # 训练的总猜环周期
    layer = 'cnn_hidden1' #训练的层
    indiv_dir = f'CGPIndiva/{layer}/'
    params = {
        'prob': 0.4,
        'verbose': 10,
        'stop_fitness': 1e-6,
        'n_row': 2,
        'n_col': 2,
        'levels_back': None,
        'n_eph': 1,
        'function_set': default_functions,
        'optim_interval': 50
    }
    itemNum = 1
    # 最大的代数
    batch_size = 60
    batch_num = 0

    train_loader = get_data(batch_size)

    # 参数初始化
    args = Args()
    args.channels_list = [1, 8]  # , 16]
    args.input_sizes = [(28, 28)]  # , (12, 12)]
    args.output_sizes = [(14, 14)]  # , (8, 8)]
    args.mlp_input = 16 * 4 * 4
    args.mlp_hiddens = [120, 84]
    args.mlp_output = 10
    args.params = params
    args.batch_size = 200
    args.populationSize = 500
    args.genNum = 120
    args.lamda = 0.8
    args.mutate_prob = 0.4

    # 第一步跑遗传规划得到相对较好的结构
    # mnist_sr = Minist_SR(args)
    bestIndividual = None
    dataset_manager = DataSetManager()

    network_input = dataset_manager.dataSet[0].cuda()

    for batch_idx, (data, target) in enumerate(sr_data.train_loader):  # 针对容器中的每一个批进行循环
        # 暂时不使用整体数据集
        # if torch.cuda.is_available():
        #     data = data.cuda()
        #     network_data = model(data)
        #     input = network_data[0]
        #     target = network_data[3][:]
        # else:
        #     network_data = model(data)
        #     input = network_data[0]
        #     target = network_data[3][:]

        if torch.cuda.is_available():
            network_input = network_input.cuda()
            network_data = model(network_input)
            net_input = network_data[0]
            cnn_hidden1 = network_data[1, :]
            cnn_hidden2 = network_data[2, :]
            output = network_data[3, :]
        else:
            network_data = model(network_input)
            net_input = network_data[0]
            cnn_hidden1 = network_data[1, :]
            cnn_hidden2 = network_data[2, :]
            output = network_data[3, :]

        # 训练数据和目标
        train_input = network_input
        train_target = cnn_hidden1

        bestIndividual = evolutionNAddLamdaCommon(args, train_input, train_target, indiv_dir, run_num, run_times, Minist_SR)
        # 保存此次种群演化最优的个体的表达式
        # 保存此次种群演化最优的个体的相关信息(表达式)
        parallel_save_result(indiv_dir,bestIndividual, args.genNum, run_num, run_times)
        break

    # 在模型上添加参数
    if bestIndividual is None:
        raise ValueError("Individual is none")
    mnist_sr_weight = MnistSRWithWeight(bestIndividual)
    print(mnist_sr_weight.named_parameters())
    if torch.cuda.is_available():
        mnist_sr_weight = mnist_sr_weight.cuda()
    criterion = nn.L1Loss() # Loss 函数的定义，交叉熵
    optimizer = torch.optim.SGD(mnist_sr_weight.parameters(), lr=0.001, momentum=0.9)  # 定义优化器，普通的随机梯度下降算法
    for batch_idx, (data, target) in enumerate(sr_data.train_loader):
        if torch.cuda.is_available():
            network_input = network_input.cuda()
            network_data = model(network_input)
            net_input = network_data[0]
            cnn_hidden1 = network_data[1, :]
            cnn_hidden2 = network_data[2, :]
            output = network_data[3, :]
        else:
            network_data = model(network_input)
            net_input = network_data[0]
            cnn_hidden1 = network_data[1, :]
            cnn_hidden2 = network_data[2, :]
            output = network_data[3, :]

        # 训练数据和目标
        train_input = network_input
        train_target = cnn_hidden1

        predict = mnist_sr_weight(train_input)
        # print(predict)

        loss = criterion(predict, train_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % 50 == 0:
            print(loss.data.cpu())
        # print(batch_idx)
