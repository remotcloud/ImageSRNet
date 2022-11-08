import copy
import math
import os
import traceback
from multiprocessing import Process

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# from ImageSRNet.image_functions import default_functions
# from ImageSRNet.sr_objs import ImageCGP
from ImageSRNet.image_functions import default_functions
from ImageSRNet.sr_objs import ImageCGP

import _pickle as pickle
from joblib import Parallel, delayed
import json

params = {
    'n_population': 100,
    'n_generation': 5000,
    'prob': 0.4,
    'verbose': 10,
    'stop_fitness': 1e-6,
    'n_row': 10,
    'n_col': 10,
    'levels_back': None,
    'n_eph': 1,
    'function_set': default_functions
}
image_size = 28  # 图像的总尺寸为 28x28
num_classes = 10  # 标签的种类数
num_epochs = 20  # 训练的总猜环周期

results_dir = '../result/CGP/AllImage'

depth = [4, 8]


class ConvNet(nn.Module):
    def __init__(self):
        # 该函数在创建一个ConvNet对象即调用语句net=ConvNet()时就会被调用
        # 首先调用父类相应的构造函数
        super(ConvNet, self).__init__()

        # 其次构造ConvNet需要用到的各个神经模块
        # 注意，定义组件并不是卖正搭建组件，只是把基本建筑砖块先找好
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)  # 定义一个卷积层，输入通道为1，输出通道为4，窗口大小为5，padding为2
        self.pool = nn.MaxPool2d(2, 2)  # 定义一个池化层，一个窗口为2x2的池化运箅
        # 第二层卷积，输入通道为depth[o]，输出通道为depth[2]，窗口为 5，padding 为2
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)  # 输出通道为depth[1]，窗口为5，padding为2
        # 一个线性连接层，输入尺寸为最后一层立方体的线性平铺，输出层 512个节点
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)
        # nn.Conv2d
        self.fc2 = nn.Linear(512, num_classes)  # 最后一层线性分类单元，输入为 512，输出为要做分类的类别数

    def forward(self, x):  # 该函数完成神经网络真正的前向运算，在这里把各个组件进行实际的拼装
        # x的尺寸：(batch_size, image_channels, image_width, image_height)
        layer1CnnInput = x

        x = self.conv1(x)  # 第一层卷积
        layer1CnnOut = F.relu(x)  # 激活函数用ReLU，防止过拟合
        # x的尺寸：(batch_size, num_filters, image_width, image_height)

        layer2input = self.pool(layer1CnnOut)  # 第二层池化，将图片变小
        # x的尺寸：(batch_size, depth[0], image_width/ 2， image_height/2)

        x = self.conv2(layer2input)  # 第三层又是卷积，窗口为5，输入输出通道分列为depth[o]=4,depth[1]=8
        layer2output = F.relu(x)  # 非线性函数
        # x的尺寸：(batch_size, depth[1], image_width/2, image_height/2)

        x = self.pool(layer2output)  # 第四层池化，将图片缩小到原来的 1/4
        # x的尺寸：(batch_size, depth[1], image_width/ 4, image_height/4)

        # 将立体的特征图 tensor 压成一个一维的向量
        # view 函数可以将一个tensor 按指定的方式重新排布
        # 下面这个命令就是要让x按照batch_size * (image_ size//4)^2*depth[1]的方式来排布向量
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        # x的尺寸：(batch_ size， depth[1J*image width/4*image height/4)

        x = F.relu(self.fc1(x))  # 第五层为全连接，ReLU激活函数
        # x的尺才：(batch_size, 512)

        # 以默认0.5的概率对这一层进行dropout操作，防止过拟合
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)  # 全连接
        # X的尺寸：(batch_size, num_classes)

        # 输出层为 log_Softmax，即概率对数值 log(p(×))。采用log_softmax可以使后面的交叉熵计算更快
        x = F.log_softmax(x, dim=1)
        return layer1CnnInput, layer1CnnOut, layer2input, layer2output

    def retrieve_features(self, x):
        # 该函数用于提取卷积神经网络的特征图，返回feature_map1,feature_map2为前两层卷积层的特征图
        feature_map1 = F.relu(self.conv1(x))  # 完成第一层卷积
        x = self.pool(feature_map1)  # 完成第一层池化
        # 第二层卷积，两层特征图都存储到了 feature_map1,feature map2 中
        feature_map2 = F.relu(self.conv2(x))
        return (feature_map1, feature_map2)

    def testGpu(self, x):
        target = torch.rand_like(x)
        target = target.cpu()
        y = x * target
        return y


def loss_function(X, y, model):
    try:
        if torch.cuda.is_available():
            model = model.cuda()
        predictY = model(X)

        criterion = nn.L1Loss()
        loss = criterion(predictY, y)

        # predictY = predictY.reshape(-1)
        # loss = math.sqrt(((predictY - y.reshape(-1)) ** 2).mean())
        # loss.backward()

    except Exception as e:
        traceback.print_exc()
        predictY = predictY.reshape(-1)

        loss = ((predictY - y.reshape(-1)) ** 2).mean()
        # print("loss="+str(loss))
    return loss.data.cpu()


def evolution(evlutionParam, input, target, file):
    populationSize = n_input = evlutionParam['populationSize']
    n_input = evlutionParam['n_input']
    n_output = evlutionParam['n_output']
    input_size = evlutionParam['input_size']
    output_size = evlutionParam['output_size']
    params = evlutionParam['params']
    genNum = evlutionParam['genNum']
    icgpPopulation = []
    for i in range(0, populationSize):
        icgp = ImageCGP(n_input, n_output, input_size, output_size, params)
        icgpPopulation.append(icgp)
    # 计算种群中的每个个体的适应度函数值
    for i in range(0, populationSize):
        indiv = icgpPopulation[i]
        try:
            indiv.fitness = loss_function(input, target, indiv)
        except:
            indiv.fitness = loss_function(input, target, indiv)
    # 寻找当代最优个体，并且记录了该个体
    bestIndividual = min(icgpPopulation, key=lambda x: x.fitness)

    with open(file, "a") as f:
        f.write(str(0) + " " + str(bestIndividual.fitness) + "\n")
    # 下一代种群
    newPopulation = [i for i in range(0, populationSize)]
    # 对于每一代种群
    for gen in range(0, genNum):

        if gen != 0:
            # 新种群成为下一代种群
            for i in range(0, populationSize):
                icgpPopulation[i] = newPopulation[i]
            # 寻找当代最优个体，并且记录了该个体
            bestIndividual = min(icgpPopulation, key=lambda x: x.fitness)
            if gen == genNum or gen % 30 == 0:
                with open(file, "a") as f:
                    f.write(str(gen) + " " + str(bestIndividual.fitness) + "\n")
                print(str(bestIndividual.fitness))
                print("gen=" + str(gen))
        if gen < (genNum - 1):
            # 将最优个体添加至下一代种群
            newPopulation[0] = bestIndividual
            # 变异产生新个体
            for i in range(1, populationSize):
                mutated_icgp = bestIndividual.mutate(0.4)
                # 计算新个体的适应度函数值
                mutated_icgp.fitness = loss_function(input, target, mutated_icgp)
                # 在变异产生的新个体和父代个体中选择最优秀的个体保存下来
                if mutated_icgp.fitness <= bestIndividual.fitness:
                    newPopulation[i] = mutated_icgp
                else:
                    newPopulation[i] = bestIndividual
    return bestIndividual
def evolutionNAddLamda(evlutionParam, input, target, file,run_num,item):
    populationSize = evlutionParam['populationSize']
    n_input = evlutionParam['n_input']
    n_output = evlutionParam['n_output']
    input_size = evlutionParam['input_size']
    output_size = evlutionParam['output_size']
    params = evlutionParam['params']
    genNum = evlutionParam['genNum']
    lamda = evlutionParam['lamda']
    mutate_prob = evlutionParam['mutate_prob']
    icgpPopulation = []
    for i in range(0, populationSize):
        icgp = ImageCGP(n_input, n_output, input_size, output_size, params)
        icgpPopulation.append(icgp)
    # 计算种群中的每个个体的适应度函数值
    for i in range(0, populationSize):
        indiv = icgpPopulation[i]
        try:
            indiv.fitness = loss_function(input, target, indiv)
        except:
            indiv.fitness = loss_function(input, target, indiv)
    # 寻找当代最优个体，并且记录了该个体
    bestIndividual = min(icgpPopulation, key=lambda x: x.fitness)

    with open(file, "a") as f:
        f.write(str(0) + " " + str(bestIndividual.fitness) + "\n")
    # 下一代种群
    newPopulation = [i for i in range(0, populationSize)]
    file_dir = "../result"
    writer = SummaryWriter(file_dir)
    # 对于每一代种群
    for gen in range(0, genNum):

        loss = bestIndividual.fitness
        writer.add_scalar("loss", loss, gen)
        if gen != 0:
            # 新种群成为下一代种群
            # for i in range(0, populationSize):
            #     icgpPopulation[i] = newPopulation[i]
            icgpPopulation = newPopulation
            # 寻找当代最优个体，并且记录了该个体
            bestIndividual = min(icgpPopulation, key=lambda x: x.fitness)
            if gen == genNum or gen % 30 == 0:
                with open(file, "a") as f:
                    f.write(str(gen) + " " + str(bestIndividual.fitness) + "\n")
                print(str(bestIndividual.fitness))
                print("gen=" + str(gen))
        if gen < (genNum - 1):
            # 将最优个体添加至下一代种群
            newPopulation[0] = bestIndividual
            # 变异产生新个体
            for i in range(1, math.ceil(populationSize * lamda)):
                mutated_icgp = bestIndividual.mutate(mutate_prob)
                # 计算新个体的适应度函数值
                mutated_icgp.fitness = loss_function(input, target, mutated_icgp)
                # 在变异产生的新个体和父代个体中选择最优秀的个体保存下来
                if mutated_icgp.fitness <= bestIndividual.fitness:
                    newPopulation[i] = mutated_icgp
                else:
                    newPopulation[i] = bestIndividual
            for j in range(math.ceil(populationSize * lamda), populationSize):
                new_icgp = ImageCGP(n_input, n_output, input_size, output_size, params)
                # 计算新个体的适应度函数值
                new_icgp.fitness = loss_function(input, target, new_icgp)
                # 在变异产生的新个体和父代个体中选择最优秀的个体保存下来
                newPopulation[j] = new_icgp
        if gen % 1000 == 0:
            #异步存储数据
            # bestIndividual.fitness = bestIndividual.fitness.detach()
            # bestIndividual_copy = copy.deepcopy(bestIndividual)
            p = Process(target=parallel_save_result, args=(bestIndividual,gen,run_num,item,))
            print('Child process will start.')
            p.start()
    writer.close()
    return bestIndividual
def parallel_save_result(best_individual, gen,run_num,item):
    indiv_dir = f'CGPIndiva/'
    save_dir = os.path.join(indiv_dir,f'{run_num}_{item}_indiv')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fn = f'CGPIndiva/{run_num}_{item}_indiv/bestIndiv.pkl'
    with open(fn, 'wb') as f:  # open file with write-mode
        pickle.dump(best_individual, f)  # serialize and save objec

    #保存日志信息，每个表达式
    bestExpression = best_individual.get_expressions()
    express_dict = {}
    for i, exp in enumerate(bestExpression):
        express_dict[f'exp({i})'] = str(exp)

    log_info = {
        'run_num': str(run_num),
        'run_item': str(item),
        'generation': str(gen),
        'fitness': str(best_individual.fitness),
        'Expression': express_dict
    }
    log_dir = f'CGPIndiva/{run_num}_{item}_indiv/log.json'
    with open(log_dir, 'w') as f:  # open file with write-mode
        json.dump(log_info, f ,indent=4)  # serialize and save objec
    print(f'{gen} generation program save OK!')

def get_data(batch_size):
    '''______________________________开始获取数据的过程______________________________'''
    # 加载MNIST数据 MNIST数据属于 torchvision 包自带的数据,可以直接接调用
    # 当用户想调用自己的图俱数据时，可以用torchvision.datasets.ImageFolder或torch.utils.data. TensorDataset来加载
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
    return train_loader
if __name__ == '__main__':

    '''
    得到Cnn的神经网络
    '''
    model = ConvNet()
    fname = 'ministCnn'
    model.load_state_dict(torch.load(f"ModelSave/{fname}.pth"))

    if torch.cuda.is_available():
        model = model.cuda()

    # 最大的演化次数
    itemNum = 1
    # 最大的代数
    batch_size = 60
    batch_num = 0
    run_num = 1

    train_loader = get_data(batch_size)

    evolutionParam = {
        'params': params,
        'batch_size': 320,
        'populationSize': 20,
        'genNum': 120,
        'n_input': 4,
        'n_output': 8,
        'input_size': (14, 14),
        'output_size': (14, 14),
        'lamda': 0.8,
        'mutate_prob': 0.4
    }

    for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
        if torch.cuda.is_available():
            data = data.cuda()
            nndata = model(data)
            input = nndata[2]
            target = nndata[3][:]
        else:
            nndata = model(data)
            input = nndata[2]
            target = nndata[3][:]
        for item in range(0, itemNum):
            # 每次种群演化中每代适应度被保存的路径
            filedir = os.path.join(results_dir, "Fitness")
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            file = os.path.join(filedir, str(item) + ".txt")
            # bestIndividual = evolution(evolutionParam,input,target,file)
            bestIndividual = evolutionNAddLamda(evolutionParam, input, target, file,run_num,item)
            # 保存此次种群演化最优的个体的表达式
            # 保存此次种群演化最优的个体的相关信息(表达式)
            parallel_save_result(bestIndividual, evolutionParam['genNum'], run_num, item)
            '''fileExpression = os.path.join(results_dir, "Expression.txt")
            bestExpression = bestIndividual.get_expressions()
            with open(fileExpression, "a") as f:
                f.write(str(item) + " " + str(bestExpression) + "\n")

            fn = f'CGPIndiva/{run_num}_{item}bestIndiv.pkl'
            with open(fn, 'wb') as f:  # open file with write-mode
                picklestring = pickle.dump(bestIndividual, f)  # serialize and save objec
            print("program save OK!")'''
            print(str(item) + "is Over!!!")
        batch_num = batch_num+1
        if batch_num >=1:
            break
