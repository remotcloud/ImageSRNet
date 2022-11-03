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


def evolutionNAddLamdaCommon(evlutionParam, input, target, file, run_num, item, Individual: type ):
    """
    n+lamda 通用进化算法
    :param Individual:
    :param evlutionParam:
    :param input:
    :param target:
    :param file:
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
            setattr(idiv,'fitness',loss_function(input, target, idiv))
        except Exception as e:
            print(e)
            setattr(idiv, 'fitness', loss_function(input, target, idiv))
    # 寻找当代最优个体，并且记录了该个体
    best_individual = min(population, key=lambda x: x.fitness)

    with open(file, "a") as f:
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
                with open(file, "a") as f:
                    fitness = getattr(best_individual, 'fitness')
                    f.write(str(gen) + " " + str(fitness) + "\n")
                # print(str(fitness))
                # print("gen=" + str(gen))
        if gen < (genNum - 1):
            # 将最优个体添加至下一代种群
            newPopulation[0] = best_individual
            # 变异产生新个体
            for i in range(1, math.ceil(populationSize * lamda)):
                mutated_icgp = best_individual.mutate(mutate_prob)
                # 计算新个体的适应度函数值
                #mutated_icgp.fitness = loss_function(input, target, mutated_icgp)
                setattr(mutated_icgp, 'fitness', loss_function(input, target, mutated_icgp))
                # 在变异产生的新个体和父代个体中选择最优秀的个体保存下来
                if mutated_icgp.fitness <= best_individual.fitness:
                    newPopulation[i] = mutated_icgp
                else:
                    newPopulation[i] = best_individual
            for j in range(math.ceil(populationSize * lamda), populationSize):
                new_icgp = Individual(evlutionParam)
                # 计算新个体的适应度函数值
                new_icgp.fitness = loss_function(input, target, new_icgp)
                # 在变异产生的新个体和父代个体中选择最优秀的个体保存下来
                newPopulation[j] = new_icgp
        if gen % 1000 == 0:
            #异步存储数据
            # bestIndividual.fitness = bestIndividual.fitness.detach()
            # bestIndividual_copy = copy.deepcopy(bestIndividual)
            p = Process(target=parallel_save_result, args=(best_individual,gen,run_num,item,))
            print('Child process will start.')
            p.start()
    writer.close()
    return best_individual

def parallel_save_result(best_individual, gen,run_num,item):
    """Save the best individual and the corresponding information"""

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