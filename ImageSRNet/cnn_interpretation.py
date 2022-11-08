from ImageSRNet.minis_sr_utils import evolutionNAddLamdaCommon, get_data, DataSetManager, parallel_save_result
from ImageSRNet.sr_nets import *
from ImageSRNet import *
import torch
import torch.nn as nn
from ImageSRNet.Minist_imageCGP import ConvNet  # loss函数希望最后能够通用化
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
    def __str__(self):
        return 'MnistSRWithWeight'


class Args(object):
    def __init__(self):
        super(Args, self).__init__()


if __name__ == '__main__':
    run_num = 1         # number of run the program
    run_times = 1       # times of run the program
    hidden_num = 1      # number of hidden layer
    epoch_num = 1000    # 第二步学习参数的训练步数
    checkpoint_interval = 200
    model_name = 'MnistSRWithWeight'
    # print(SRNetDataset)
    sr_data = Minist_Data("./data", get_data(60))

    model = ConvNet()
    fname = 'ministCnn'
    model.load_state_dict(torch.load(f"ModelSave/{fname}.pth"))

    if torch.cuda.is_available():
        model = model.cuda()

    layer = f'cnn_hidden{hidden_num}' #训练的层
    indiv_dir = f'CGPIndiva/{layer}/'

    # CGP的参数初始化
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
    input_sizes = [[(28, 28)],[(14,14)]]
    output_sizes = [[(28, 28)],[(14,14)]]
    # SRNN的参数初始化
    args = Args()
    args.channels_list = [1, 4]  # , 16]
    args.input_sizes = input_sizes[0]
    args.output_sizes = output_sizes[0]
    args.mlp_input = 16 * 4 * 4
    args.mlp_hiddens = [120, 84]
    args.mlp_output = 10
    args.params = params
    args.batch_size = 200

    args.populationSize = 50
    args.genNum = 12
    args.lamda = 0.8
    args.mutate_prob = 0.4

    # 第一步跑遗传规划得到相对较好的结构
    # mnist_sr = Minist_SR(args)

    bestIndividual = None
    dataset_manager = DataSetManager()

    # 加载规定的一批数据
    network_input = dataset_manager.raw_mnist_data[0]

    for batch_idx, (data, target) in enumerate(sr_data.train_loader):  # 针对容器中的每一个批进行循环
        # 暂时不使用整体数据集
        if torch.cuda.is_available():
            network_input = network_input.cuda()
            network_data = model(network_input)
        else:
            network_data = model(network_input)
        # 训练数据和目标
        train_input = network_data[hidden_num-1]
        train_target = network_data[hidden_num]

        bestIndividual = evolutionNAddLamdaCommon(args, train_input, train_target, indiv_dir, run_num, run_times, Minist_SR)
        # 保存此次种群演化最优的个体的表达式
        # 保存此次种群演化最优的个体的相关信息(表达式)
        parallel_save_result(indiv_dir, bestIndividual, args.genNum, run_num, run_times)
        break

    if bestIndividual is None:
        raise ValueError("Individual is none")
    # 在模型上添加参数
    model_weight = MnistSRWithWeight(bestIndividual)
    print(model_weight.named_parameters())

    if torch.cuda.is_available():
        model_weight = model_weight.cuda()
    criterion = nn.L1Loss() # Loss 函数的定义，交叉熵
    optimizer = torch.optim.SGD(model_weight.parameters(), lr=0.001, momentum=0.9)  # 定义优化器，普通的随机梯度下降算法

    for epoch in range(epoch_num):
        for batch_idx, (data, target) in enumerate(sr_data.train_loader):
            if torch.cuda.is_available():
                network_input = network_input.cuda()
                network_data = model(network_input)
            else:
                network_data = model(network_input)

            # 训练数据和目标
            train_input = network_data[hidden_num - 1]
            train_target = network_data[hidden_num]

            predict = model_weight(train_input)
            # print(predict)

            loss = criterion(predict, train_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if batch_idx % 500 == 0:
            #     print("batch_idx {}:".format(batch_idx), loss.data.cpu())


            # print(batch_idx)
        if (epoch % 50 == 0):
            print("epoch {}:".format(epoch), loss.data.cpu())
        if (epoch % checkpoint_interval == 0):
            fname = str(model_weight)
            torch.save(model_weight.state_dict(), f"{indiv_dir}{fname}.pth")

