import copy
import os.path
import random
from functools import reduce


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

# from ImageSRNet.image_functions import default_functions
# from ImageSRNet.utils import ImageCGPParameters, create_icgp_genes_bounds, get_active_paths, create_func_params_list, \
#     CGPParameters, create_genes_and_bounds, create_nodes, get_active_paths_by_nodes
from image_functions import default_functions
from utils import ImageCGPParameters, create_icgp_genes_bounds, get_active_paths, create_func_params_list, \
    CGPParameters, create_genes_and_bounds, create_nodes, get_active_paths_by_nodes

class ImageCGP(nn.Module):
    def __init__(
            self, n_input, n_output, input_size, output_size, params: dict,
            genes=None, bounds=None, ephs=None):
        super(ImageCGP, self).__init__()
        # 输入几张图片
        self.n_input = n_input
        # 输出几张图片
        self.n_output = n_output
        # 表示CGP每一行有多少个节点，每一列有多少个节点
        self.input_size = input_size
        self.output_size = output_size

        self.dict_params = params
        self.params = ImageCGPParameters(self.n_input, self.n_output, self.input_size, self.output_size, params)
        # a group of genes:[f_gene, finput1, finput2 ,..., finput_max_arity, row_pos1, col_pos1, row_pos2, col_pow2, ..., .]
        self.len_group = self.params.max_arity * 3 + 1

        self.genes = genes
        self.bounds = bounds
        self.ephs = ephs
        if self.genes is None:
            self.genes, self.bounds = create_icgp_genes_bounds(self.params)
            self.ephs = torch.rand((self.params.n_eph,))
        else:
            assert self.bounds is not None and self.ephs is not None

        self.func_params_list = create_func_params_list(self.genes, self.len_group, self.params)  # list of tuple
        self.active_paths = get_active_paths(self.genes, self.len_group, self.params)

        self.fitness = None

    def get_expressions(self, var_names=None):
        assert var_names is None or len(var_names) == self.n_input

        n_input_node = self.params.n_input + self.params.n_eph
        max_arity = self.params.max_arity
        outputs, symbol_stack = [], []
        node_idx_stack = []
        for active_path in self.active_paths:
            for node_idx in active_path:
                if node_idx < n_input_node:
                    if node_idx < self.n_input:
                        c = 'IMG_{}'.format(node_idx) if var_names is None else var_names[node_idx]
                    else:
                        c = self.ephs[node_idx - self.n_input].item()
                else:
                    func_gene_idx = (node_idx - n_input_node) * self.len_group
                    func = self.params.function_set[self.genes[func_gene_idx]]

                    func_params = self.func_params_list[node_idx - n_input_node]
                    # get the feature infos
                    oper_node_idxs = reversed([node_idx_stack.pop() for _ in range(func.arity)])
                    connect_infos = self.genes[func_gene_idx + max_arity + 1:func_gene_idx + max_arity + func.arity * 2 + 1]
                    feature_infos = []
                    for i, oper_node_idx in enumerate(oper_node_idxs):
                        feature_infos.append(((connect_infos[i*2], connect_infos[i*2+1]), self.output_size) if oper_node_idx < self.n_input else None)
                    # get the functional value
                    operants = reversed([symbol_stack.pop() for _ in range(func.arity)])
                    c = func.expr(*operants, feature_infos=feature_infos, params=func_params)
                node_idx_stack.append(node_idx)
                symbol_stack.append(c)
            outputs.append(symbol_stack.pop())
        return outputs

    def forward(self, input_image):
        # input image: [BxCxHxW]
        assert input_image.shape[1] == self.n_input

        n_input_node = self.params.n_input + self.params.n_eph
        max_arity = self.params.max_arity
        outputs, value_stack = [], []
        node_idx_stack = []
        for active_path in self.active_paths:
            for node_idx in active_path:
                if node_idx < n_input_node:
                    value = input_image[:, node_idx:node_idx+1] if node_idx < self.n_input else self.ephs[node_idx-self.n_input]
                else:
                    func_gene_idx = (node_idx - n_input_node) * self.len_group
                    func = self.params.function_set[self.genes[func_gene_idx]]

                    func_params = self.func_params_list[node_idx-n_input_node]
                    # get the feature infos
                    oper_node_idxs = reversed([node_idx_stack.pop() for _ in range(func.arity)])
                    connect_infos = self.genes[func_gene_idx+max_arity+1:func_gene_idx+max_arity+func.arity*2+1]
                    feature_infos = []
                    for i, oper_node_idx in enumerate(oper_node_idxs):
                        feature_infos.append(((connect_infos[i*2], connect_infos[i*2+1]), self.output_size) if oper_node_idx < self.n_input else None)
                    # get the functional value
                    operants = reversed([value_stack.pop() for _ in range(func.arity)])
                    value = func(*operants, feature_infos=feature_infos, params=func_params)
                node_idx_stack.append(node_idx)
                value_stack.append(value)
            output_value = value_stack.pop()
            if len(output_value.shape) == 0:
                output_value = output_value.repeat(input_image.shape[0], 1, self.output_size[0], self.output_size[1])
            outputs.append(output_value)
        if torch.cuda.is_available():
            return torch.cat(outputs, dim=1).cuda()
        else:
            return torch.cat(outputs, dim=1)
    def mutate(self, prob):
        # apply point mutation
        mutated_genes = self.genes[:]
        low_bounds, up_bounds = self.bounds
        for i in range(len(mutated_genes)):
            choice = random.random()
            if choice <= prob:
                # mutate the genes
                low_bound, up_bound = low_bounds[i], up_bounds[i]
                candicates = [avai_gene for avai_gene in range(low_bound, up_bound+1) if avai_gene != mutated_genes[i]]
                if len(candicates) == 0:
                    continue
                mutated_genes[i] = random.choice(candicates)
        ephs = copy.deepcopy(self.ephs)
        choice = random.random()
        if choice <= prob:
            ephs = torch.rand((self.params.n_eph,))
        return ImageCGP(self.n_input, self.n_output, self.input_size, self.output_size, self.dict_params,
                        mutated_genes, self.bounds, ephs)


class CGPLayer(nn.Module):
    def __init__(self, n_input, n_output, params:dict, genes=None, bounds=None, ephs=None):
        super(CGPLayer, self).__init__()
        self.cp = CGPParameters(n_input, n_output, params)
        self.len_group = self.cp.max_arity + 1
        if genes is None:
            self.genes, self.bounds = create_genes_and_bounds(self.cp)
        else:
            assert bounds is not None
            self.genes, self.bounds = genes, bounds

        self.nodes = None
        self.active_paths = None
        self.active_nodes = None
        self.build()

        if ephs is None:
            self.ephs = nn.Parameter(torch.normal(mean=0., std=1., size=(self.cp.n_eph,)))
        else:
            self.ephs = ephs

    def build(self):
        self.nodes = create_nodes(self.cp, self.genes)
        self.active_paths = get_active_paths_by_nodes(self.nodes)
        self.active_nodes = set(reduce(lambda l1, l2: l1 + l2, self.active_paths))

    def forward(self, x):
        """normal CGP call way. Seeing x[:, i] as a single variable.
         INPUT: Make sure x.shape[1] == n_input
        OUTPUT: y where y.shape[1] == n_output """
        for path in self.active_paths:
            for gene in path:
                node = self.nodes[gene]
                if node.is_input:
                    node.value = self.ephs[node.no - self.cp.n_input] if node.no >= self.cp.n_input else x[:, node.no]
                elif node.is_output:
                    node.value = self.nodes[node.inputs[0]].value
                else:
                    f = node.func
                    operants = [self.nodes[node.inputs[i]].value for i in range(node.arity)]
                    node.value = f(*operants)

        outputs = []
        for node in self.nodes[-self.cp.n_output:]:
            if len(node.value.shape) == 0:
                outputs.append(node.value.repeat(x.shape[0]))
            else:
                outputs.append(node.value)

        return torch.stack(outputs, dim=1)

    def get_expression(self, input_vars=None, symbol_constant=False):
        if input_vars is not None and len(input_vars) != self.cp.n_input:
            raise ValueError(f'Expect len(input_vars)={self.n_inputs}, but got {len(input_vars)}')

        symbol_stack = []
        results = []
        for path in self.active_paths:
            for i_node in path:
                node = self.nodes[i_node]
                if node.is_input:
                    if i_node >= self.cp.n_input:
                        c = f'c{i_node - self.n_inputs}' if symbol_constant \
                            else self.ephs[i_node - self.cp.n_input].item()
                    else:
                        if input_vars is None:
                            c = f'x{i_node}' if self.cp.n_input > 1 else 'x'
                        else:
                            c = input_vars[i_node]
                    symbol_stack.append(c)
                elif node.is_output:
                    results.append(symbol_stack.pop())
                else:
                    f = node.func
                    # get a sympy symbolic expression.
                    symbol_stack.append(f(*reversed([symbol_stack.pop() for _ in range(f.arity)]), is_pt=False))

        return results


if __name__ == '__main__':
    params = {
        'n_population': 100,
        'n_generation': 10,
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
    n_input, n_output, input_size, output_size = 1, 1, (28, 28), (10)
    # input_img = torch.rand(1, n_input, *input_size)
    # 加载图像（地震记录）
    image = np.loadtxt("V:\yang\code\Sesmic\Dataset\seismic.csv",dtype=np.float, delimiter=',').reshape(1, 1, 512, 2721)
    image = torch.tensor(image).float()
    # plt.imshow(image.squeeze())
    # plt.show()
    # 加载波阻抗
    y = np.loadtxt("V:\yang\code\Sesmic\Dataset\impedance.csv", dtype=np.float, delimiter=',').reshape(1, 1, 512, 2721)
    y = torch.tensor(y).float()
    # plt.imshow(y.squeeze())
    # plt.show()


    # 效果保存的目录
    results_dir = r'V:\yang\code\Sesmic\Result\CGP'

    def _print_icgp(indiv, x):
        print('genes:', indiv.genes, '\nbounds', indiv.bounds, '\nparams', indiv.func_params_list, '\nactive_paths',
              indiv.active_paths)
        print('expr:', indiv.get_expressions())
        #
        # print('input:', x.shape)
        # print('output:', indiv(x).shape)
        # print('output:', indiv(x))
        # y = indiv(x)
        # plt.imshow(y.squeeze())
        # plt.show()
    # icgp = ImageCGP(n_input, n_output, input_size, output_size, params)
    # mutated_icgp = icgp.mutate(0.4)
    # _print_icgp(icgp, image)
    # _print_icgp(mutated_icgp, image)
    # 最大的演化次数
    itemNum=30
    # 最大的代数
    genNum=10
    # 种群相关设置
    populationSize = 10

    # 对于每一次种群演化
    for item in range(11,itemNum):
        icgpPopulation = []
        # 创建种群
        for i in range(0, populationSize):
            icgp = ImageCGP(n_input, n_output, input_size, output_size, params)
            icgpPopulation.append(icgp)
            # _print_icgp(icgp,image)
        filedir = os.path.join(results_dir, "Fitness")
        file = os.path.join(filedir, str(item)+".txt")
        #对于每一代种群
        for gen in range(0,genNum):
            #计算种群中的每个个体的适应度函数值
            populationFitness = []
            for i in range(0, populationSize):
                indiv = icgpPopulation[i]
                fitness = indiv(image)
                # MSE函数
                loss = torch.nn.MSELoss(reduction='mean')
                fitness = loss(y.float(), fitness.float())
                indiv.fitness = fitness
                # print(str(gen)+" "+str(fitness.float()))
                populationFitness.append(fitness.float())
            # 选择
            # bestIndividualIndex=populationFitness.index(min(populationFitness))
            # bestIndividual = icgpPopulation[bestIndividualIndex]
            # 保存本次种群演化中每代种群演化中最优个体的适应度函数值
            # bestFitness=populationFitness[bestIndividualIndex].float()
            bestIndividual = min(icgpPopulation, key=lambda x: x.fitness)
            bestFitness = bestIndividual.fitness
            with open(file, "a") as f:
                # f.write(str(gen)+" "+str(bestFitness)+" "+bestIndividual.get_expressions()+"\n")
                f.write(str(gen) + " " + str(bestFitness)+ "\n")

            # 新种群
            newPopulation=[]
            newPopulation.append(bestIndividual)
            # 变异产生新个体
            for i in range(1,populationSize):
                mutated_icgp = bestIndividual.mutate(0.4)
                newPopulation.append(mutated_icgp)
            # print("-----")
            # print(len(newPopulation))
            # print("----")
            # 新种群成为下一代种群
            for i in range(0, populationSize):
                icgpPopulation[i] = newPopulation[i]

        # for i in range(0,populationSize):
        #             _print_icgp(icgpPopulation[i],image)
        #找到本次种群演化的最终种群的最优个体
        #计算种群中的每个个体的适应度函数值
        populationFitness = []
        for i in range(0, populationSize):
            indiv = icgpPopulation[i]
            fitness = indiv(image)
            # MSE函数
            loss = torch.nn.MSELoss(reduction='mean')
            fitness = loss(y.float(), fitness.float())
            indiv.fitness = fitness
            # print(fitness.float())
            populationFitness.append(fitness.float())
        # 打印此次种群演化最优的个体的相关信息
        # bestIndividualIndex=populationFitness.index(min(populationFitness))
        bestIndividual = min(icgpPopulation, key=lambda x: x.fitness)
        bestFitness = bestIndividual.fitness
        print('expr:', bestIndividual.get_expressions())
        print('y:', bestIndividual(image))
        print('fitness:', bestFitness)
        y = bestIndividual(image)
        plt.imshow(y.squeeze())
        figName="result"+str(item)+".png"
        figpath=os.path.join(results_dir,"Fig")
        figName=os.path.join(figpath,figName)
        plt.savefig(figName)
        # plt.savefig("result.png")
        # plt.show()

