import os

import numpy as np
import torch
from matplotlib import pyplot as plt

# from ImageSRNet.image_functions import default_functions
# from ImageSRNet.sr_objs import ImageCGP
from image_functions import default_functions
from sr_objs import ImageCGP

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

# X = np.loadtxt('../Dataset/seismic.csv', dtype=float, delimiter=',')
# print(X[:,0])
# print(X[:,2])
# Y = np.loadtxt('../Dataset/impedance.csv', dtype=float, delimiter=',')
# 原来的
n_input, n_output, input_size, output_size = 1, 1, (512, 2721), (512, 2721)
# reshape(a, b, *input_size)a表示batch数目，b表示通道数
orignalX = np.loadtxt('../Dataset/seismic.csv', dtype=float, delimiter=',').reshape(1, 1, *input_size)
orignalY = np.loadtxt('../Dataset/impedance.csv', dtype=float, delimiter=',').reshape(1, 1, *input_size)


X = torch.tensor(orignalX, dtype=torch.float)
Y = torch.tensor(orignalY, dtype=torch.float)

print(X.shape, Y.shape)


results_dir = '../Result/CGP/AllImage'
# 处理orignalY,去掉多余的两个维度，以便画图
orignalY=orignalY.squeeze()
# orignalY=orignalY.numpy()

def loss_function(X, y, model):
        # loss=0
        # Y = y.reshape(-1)
        # predictY= model(X).reshape(-1)
        # for i in range(0,len(predictY)):
        #     loss = 1.0*i/(i+1)*loss+1.0/(i+1)*((predictY[i]-Y[i])**2)
        # return loss
        # print("Expression=",model.get_expressions())
        predictY = model(X).reshape(-1)
        # print("predictY="+str(predictY))
        loss = ((predictY-y.reshape(-1))**2).sum().mean()
        # print("loss="+str(loss))
        return loss
        # func = torch.nn.MSELoss()

        # return func(y, model(X))


if __name__ == '__main__':
# 最大的演化次数
    itemNum=30
    # 最大的代数
    genNum=100
    # 种群相关设置
    populationSize = 100

    # 对于每一次种群演化
    for item in range(0, itemNum):
        # 每次种群演化中每代适应度被保存的路径
        filedir = os.path.join(results_dir, "Fitness")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        file = os.path.join(filedir, str(item) + ".txt")
        # 创建当代种群
        icgpPopulation = []
        for i in range(0, populationSize):
            icgp = ImageCGP(n_input, n_output, input_size, output_size, params)
            icgpPopulation.append(icgp)
        # 计算种群中的每个个体的适应度函数值
        for i in range(0, populationSize):
            indiv = icgpPopulation[i]
            indiv.fitness = loss_function(X, Y, indiv)
        # 寻找当代最优个体，并且记录了该个体
        bestIndividual = min(icgpPopulation, key=lambda x: x.fitness)
        with open(file, "a") as f:
                f.write(str(0) + " " + str(bestIndividual.fitness) + "\n")
        # 下一代种群
        newPopulation = [i for i in range(0, populationSize)]
        #对于每一代种群
        for gen in range(0, genNum):
            print("gen=" + str(gen))
            if gen != 0:
                # 新种群成为下一代种群
                for i in range(0, populationSize):
                    icgpPopulation[i] = newPopulation[i]
                # 寻找当代最优个体，并且记录了该个体
                bestIndividual = min(icgpPopulation, key=lambda x: x.fitness)
                with open(file, "a") as f:
                    f.write(str(gen) + " " + str(bestIndividual.fitness) + "\n")
            if gen < (genNum - 1):
                # 将最优个体添加至下一代种群
                newPopulation[0] = bestIndividual
                # 变异产生新个体
                for i in range(1, populationSize):
                    mutated_icgp = bestIndividual.mutate(0.4)
                    # 计算新个体的适应度函数值
                    mutated_icgp.fitness = loss_function(X, Y, mutated_icgp)
                    # 在变异产生的新个体和父代个体中选择最优秀的个体保存下来
                    if mutated_icgp.fitness <= bestIndividual.fitness:
                        newPopulation[i] = mutated_icgp
                    else:
                        newPopulation[i] = bestIndividual



        #计算种群中的每个个体的适应度函数值
        # populationFitness = []
        # for i in range(0, populationSize):
        #     indiv = icgpPopulation[i]
        #     indiv.fitness = loss_function(X, Y, indiv)
        # print("gen=" + str(gen))
        # # 找到本次种群演化的最终种群的最优个体并且记录
        # bestIndividual = min(icgpPopulation, key=lambda x: x.fitness)
        # # bestFitness = bestIndividual.fitness
        # with open(file, "a") as f:
        #     f.write(str(gen) + " " + str(bestIndividual.fitness) + "\n")
        # 保存此次种群演化最优的个体的表达式
        # 保存此次种群演化最优的个体的相关信息(表达式)
        fileExpression = os.path.join(results_dir, "Expression.txt")
        bestExpression = bestIndividual.get_expressions()
        with open(fileExpression, "a") as f:
            f.write(str(item) + " " + str(bestExpression[0]) + "\n")
        # print('expr:', bestIndividual.get_expressions())
        print('y:', bestIndividual(X))
        # 保存本次种群演化预测的波阻抗
        y = bestIndividual(X).squeeze()
        y = y.numpy()
        filePredictImp = os.path.join(results_dir, "predictImp")
        if not os.path.exists(filePredictImp):
            os.makedirs(filePredictImp)
        filePredictImp = os.path.join(filePredictImp, str(item) + ".txt")
        np.savetxt(filePredictImp, y, delimiter=',')
        # 画图并保存
        plt.subplot(3,1,1)
        # 去除Y中多余的两个维度，且将Y
        # print("_____")
        # print(item)
        # print(Y)
        # Y = Y.squeeze()
        # print(Y)
        # print("_____")
        # Y = Y.numpy()
        plt.imshow(orignalY, vmin=np.min(orignalY), vmax=np.max(orignalY))
        plt.subplot(3,1,2)
        # y = bestIndividual(X)
        plt.imshow(y, vmin=np.min(orignalY), vmax=np.max(orignalY))
        plt.subplot(3, 1, 3)
        temp_y = (np.abs((Y-y))).squeeze()
        print(temp_y)
        plt.imshow(temp_y, vmin=np.min(orignalY), vmax=np.max(orignalY))
        # 保存图片
        figName = str(item) + ".png"
        figpath = os.path.join(results_dir, "Fig")
        if not os.path.exists(figpath):
            os.makedirs(figpath)
        figName = os.path.join(figpath, figName)
        plt.savefig(figName)
        plt.close()
        print(str(item) + "is Over!!!")