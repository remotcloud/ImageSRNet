import math
import os
import pickle
import traceback
import json
import torch
from torch import nn
from multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from loss import *
from matplotlib import pyplot as plt

from Minist_imageCGP import ConvNet


class DataSetManager(object):
    def __init__(self, best_individual=None, update_raw_data=False, update_network_data=False):
        if update_network_data:
            self.network_dataSet = self.__save_network_data_to_pickle(120)
        else:
            self.network_dataSet = self.__getDataPickle('data/network_data.pkl')
        if update_raw_data:
            self.raw_mnist_data = self.__save_raw_data_to_pickle(120)
        else:
            self.raw_mnist_data = self.__getDataPickle('data/raw_mnist_data.pkl')

        self.bestIndividual = best_individual

    def __getDataPickle(self, data_dir):

        with open(data_dir, 'rb') as f:
            nnMnistData = pickle.load(f)
        return nnMnistData

    def __save_raw_data_to_pickle(self, batch_size=640):
        """Save the network data to pickle file"""
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

        for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
            fn = f'data/raw_mnist_data.pkl'
            with open(fn, 'wb') as f:
                pickle.dump([data, target], f)
            return data

    def __save_network_data_to_pickle(self, batch_size=640):
        """Save the network data to pickle file"""

        global nndata
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
        fn = f'data/network_data.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(nndata, f)
        return nndata

    def getResultImageComp(self, batch_num, hidden_num):
        """

        :param hidden_num: 第几层网络
        :param batch_num:
        :return:
        """
        # batch_num = 1

        hidden_input_data = self.network_dataSet[hidden_num-1]  # layer1CnnInput,layer1CnnOut,layer2nnInput,layer2CnnOutput
        # 求出神经网络的输出
        hidden_data_one_batch = hidden_input_data[batch_num - 1:batch_num]  # 神经网络需要维度[1,4,14,14]
        sr_output = self.bestIndividual(hidden_data_one_batch)[0]

        hidden_network_output = self.network_dataSet[hidden_num][batch_num - 1]
        hidden_data_input_for_image = hidden_data_one_batch[0]  # [4,14,14]

        rowNum = 2
        colNum = math.ceil(hidden_data_input_for_image.shape[0] / rowNum)

        fig, cols = plt.subplots(rowNum, colNum)

        fig.suptitle("layer2Input")
        if len(cols.shape) == 1:  # cols只是一个一维数组
            for row in range(rowNum):
                for col in range(colNum):
                    cols[row].imshow(hidden_data_input_for_image[col, ...].cpu().data.numpy())
        if len(cols.shape) == 2:  # cols是一个二维数组
            for row in range(rowNum):
                for col in range(colNum):
                    cols[row][col].imshow(hidden_data_input_for_image[col, ...].cpu().data.numpy())

        fig = plt.figure(figsize=(10, 7))
        fig.suptitle('outOfSrNet')
        for i in range(sr_output.shape[0]):
            plt.subplot(2, math.ceil(sr_output.shape[0] / 2), i + 1)
            plt.imshow(sr_output[i, ...].cpu().data.numpy())

        fig = plt.figure(figsize=(10, 7))
        fig.suptitle('outOfNet')
        for i in range(hidden_network_output.shape[0]):
            plt.subplot(2, math.ceil(hidden_network_output.shape[0] / 2), i + 1)
            plt.imshow(hidden_network_output[i, ...].cpu().data.numpy())
        plt.show()

    def getResultImageComp2(self, number):
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
                plt.subplot(row, col, imageId + 1)
                plt.imshow(self.network_dataSet[2][imageId].unsqueeze(0)[0][0, ...].cpu().data.numpy())

        plt.show()

    def getResultJson(self, hidden_name):
        bestExpression = self.bestIndividual.get_expressions()
        express_dict = {}
        for i, exp in enumerate(bestExpression):
            express_dict[f'exp({i})'] = str(exp)
        log_dict = {
            "Expression": express_dict,
            "fitness": str(self.bestIndividual.fitness)
        }
        with open(f'CGPIndiva/{hidden_name}/result.json', 'w') as f:
            json.dump(log_dict, f, indent=4)
        print("Save json ok!")


def get_data(batch_size):
    '''
    获取手写数识别的数据加载器，训练、验证、测试的loader
    :param batch_size:
    :return:
    '''
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
    return train_loader, validation_loader, test_loader


def evolutionNAddLamdaCommon(evlutionParam, input, target, indiv_dir, run_num, item, Individual: type):
    """
    n+lamda 通用进化算法
    :param Individual:
    :param evlutionParam:
    :param input:
    :param target:
    :param indiv_dir:
    :param run_num:
    :param item:
    :return:
    """
    if isinstance(Individual, type) is False:
        raise ValueError("Individual is not a class")

    populationSize = evlutionParam.populationSize

    genNum = evlutionParam.genNum
    lamda = evlutionParam.lamda
    mutate_prob = evlutionParam.mutate_prob

    population = []
    for i in range(0, populationSize):
        idiv = Individual(evlutionParam)
        population.append(idiv)
    # 计算种群中的每个个体的适应度函数值
    for i in range(0, populationSize):
        idiv = population[i]
        try:
            # idiv.fitness = loss_function(input, target, idiv)
            setattr(idiv, 'fitness', loss_function(input, target, idiv))
        except Exception as e:
            print(e)
            setattr(idiv, 'fitness', loss_function(input, target, idiv))
    # 寻找当代最优个体，并且记录了该个体
    best_individual = min(population, key=lambda x: x.fitness)

    with open(indiv_dir, "a") as f:
        f.write(str(0) + " " + str(best_individual.fitness) + "\n")
    # 下一代种群
    newPopulation = [i for i in range(0, populationSize)]

    file_dir = "../result"
    writer = SummaryWriter(file_dir)

    # 对于每一代种群
    for gen in range(0, genNum):

        loss = best_individual.fitness
        writer.add_scalar("loss", loss, gen)
        if gen != 0:
            # 新种群成为下一代种群
            # for i in range(0, populationSize):
            #     icgpPopulation[i] = newPopulation[i]
            population = newPopulation
            # 寻找当代最优个体，并且记录了该个体
            best_individual = min(population, key=lambda x: x.fitness)

            if gen == genNum or gen % 30 == 0:
                fitness = getattr(best_individual, 'fitness')
                print("gen " + str(gen) + " fitness " + str(fitness))
        if gen < (genNum - 1):
            # 将最优个体添加至下一代种群
            newPopulation[0] = best_individual
            # 变异产生新个体
            for i in range(1, math.ceil(populationSize * lamda)):
                mutated_icgp = best_individual.mutate(mutate_prob)
                # 计算新个体的适应度函数值
                # mutated_icgp.fitness = loss_function(input, target, mutated_icgp)
                setattr(mutated_icgp, 'fitness', loss_function(input, target, mutated_icgp))
                # 在变异产生的新个体和父代个体中选择最优秀的个体保存下来
                if mutated_icgp.fitness <= best_individual.fitness:
                    newPopulation[i] = mutated_icgp
                else:
                    newPopulation[i] = best_individual
            for j in range(math.ceil(populationSize * lamda), populationSize):
                new_icgp = Individual(evlutionParam)
                # 计算新个体的适应度函数值
                # new_icgp.fitness = loss_function(input, target, new_icgp)
                setattr(new_icgp, 'fitness', loss_function(input, target, new_icgp))
                # 在变异产生的新个体和父代个体中选择最优秀的个体保存下来
                newPopulation[j] = new_icgp
        if gen % 1000 == 0:
            # 异步存储数据
            # bestIndividual.fitness = bestIndividual.fitness.detach()
            # bestIndividual_copy = copy.deepcopy(bestIndividual)
            p = Process(target=parallel_save_result, args=(indiv_dir, best_individual, gen, run_num, item,))
            print('Child process will start.')
            p.start()
    writer.close()
    return best_individual


def parallel_save_result(indiv_dir, best_individual, gen, run_num, run_times):
    """Save the best individual and the corresponding information"""

    save_dir = os.path.join(indiv_dir, f'r{run_num}_t{run_times}_indiv')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_indiv_file = save_dir + '/bestIndiv.pkl'
    with open(best_indiv_file, 'wb') as f:  # open file with write-mode
        pickle.dump(best_individual, f)  # serialize and save objec

    # 保存日志信息，每个表达式
    bestExpression = best_individual.get_expressions()
    express_dict = {}
    for i, exp in enumerate(bestExpression):
        express_dict[f'exp({i})'] = str(exp)

    log_info = {
        'run_num': str(run_num),
        'run_times': str(run_times),
        'generation': str(gen),
        'fitness': str(best_individual.fitness),
        'Expression': express_dict
    }
    log_file = save_dir + '/log.json'
    with open(log_file, 'w') as f:  # open file with write-mode
        json.dump(log_info, f, indent=4)  # serialize and save objec
    print(f'{gen} generation program save OK!')

