import io
import os
import pickle

import numpy as np
import torch
import random


from torch.utils.data import Dataset

# from ImageSRNet.image_functions import function_map
# from data_utils import io
from image_functions import function_map
# from data_utils import io

class CGPParameters:
    def __init__(self, n_input, n_output, params:dict):
        # 输入几张图片，输出几张图片
        self.n_input = n_input
        self.n_output = n_output
        # 表示CGP每一行有多少个节点，每一列有多少个节点
        self.n_row = params['n_row']
        self.n_col = params['n_col']

        self.levels_back = params['levels_back']

        self.n_eph = params['n_eph']

        self.function_set = []
        self.max_arity = 1
        for str_fun in params['function_set']:
            if str_fun not in function_map:
                raise ValueError("%s function is not in 'function_map' in functions.py." % str_fun)
            self.max_arity = max(function_map[str_fun].arity, self.max_arity)
            self.function_set.append(function_map[str_fun])

        self.n_f = len(self.function_set)
        self.n_fnode = self.n_row * self.n_col
        if self.levels_back is None:
            self.levels_back = self.n_fnode + self.n_input + self.n_eph


class ImageCGPParameters(CGPParameters):
    def __init__(self, n_input, n_output, input_size, output_size, params):
        super(ImageCGPParameters, self).__init__(n_input, n_output, params)
        self.input_size = input_size
        self.output_size = output_size


class SRNetDataset(Dataset):
    def __init__(self, data, targets=None):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.targets is not None:
            return self.data[index, :], [output[index, :] for output in self.targets]
        return self.data[index, :]


def create_icgp_genes_bounds(params: ImageCGPParameters):
    genes = []
    uppers, lowers = [], []
    up_row, up_col = params.input_size[0] - params.output_size[0], params.input_size[1] - params.output_size[1]
    for i in range(params.n_input + params.n_eph, params.n_input + params.n_eph + params.n_fnode):
        f_gene = random.randint(0, params.n_f - 1)

        lowers.append(0)
        uppers.append(params.n_f - 1)
        genes.append(f_gene)

        # next bits are input of the node function.
        row_th = (i - params.n_input - params.n_eph) // params.n_row
        up = params.n_input + params.n_eph + row_th * params.n_row - 1
        low = max(0, up - params.levels_back)
        for i_arity in range(params.max_arity):
            lowers.append(low)
            uppers.append(up)
            in_gene = random.randint(low, up)
            genes.append(in_gene)

        # next bits are pos_row and pos_col of the input image, which indicates the left up position
        for i_arity in range(params.max_arity):
            lowers.append(0)
            uppers.append(up_row)
            row_gene = random.randint(0, up_row)
            lowers.append(0)
            uppers.append(up_col)
            col_gene = random.randint(0, up_col)
            genes.append(row_gene)
            genes.append(col_gene)
    # output genes
    up = params.n_input + params.n_eph + params.n_fnode - 1
    # we do not allow any output nodes connect to the input node since we have already added the idtf function
    low = max(params.n_input + params.n_eph, up - params.levels_back + 1)
    for i in range(params.n_output):
        lowers.append(low)
        uppers.append(up)
        out_gene = random.randint(low, up)
        genes.append(out_gene)

    return genes, (lowers, uppers)


def create_func_params_list(genes, len_group, params:ImageCGPParameters):
    func_params_list = []
    for node_idx in range(params.n_fnode):
        f_gene_idx = node_idx * len_group
        func = params.function_set[genes[f_gene_idx]]
        if func.params_range is not None:
            func_params = []
            for p_range in func.params_range:
                choice = random.randint(0, len(p_range)-1)
                func_params.append(p_range[choice])
            func_params = tuple(func_params)
        else:
            func_params = None
        func_params_list.append(func_params)

    return func_params_list


def get_active_paths_by_nodes(nodes):
    stack = []
    active_path, active_paths = [], []
    for node in reversed(nodes):
        if node.is_output:
            stack.append(node)
        else:
            break

    while len(stack) > 0:
        node = stack.pop()

        if len(active_path) > 0 and node.is_output:
            active_paths.append(list(reversed(active_path)))
            active_path = []

        active_path.append(node.no)

        for input in reversed(node.inputs):
            stack.append(nodes[input])

    if len(active_path) > 0:
        active_paths.append(list(reversed(active_path)))

    return active_paths


def get_active_paths(genes, len_group, params: CGPParameters):
    stack = []
    active_path, active_paths = [], []

    output_genes = genes[-params.n_output:]
    for output_gene in reversed(output_genes):
        # 1 indicates it is output
        stack.append((output_gene, 1))

    while len(stack) > 0:
        node_id = stack.pop()
        is_output = isinstance(node_id, tuple)
        if is_output:
            node_id = node_id[0]

        if len(active_path) > 0 and is_output:
            active_paths.append(list(reversed(active_path)))
            active_path = []

        active_path.append(node_id)

        if node_id < params.n_input + params.n_eph:
            continue

        node_gene_idx = (node_id - params.n_input - params.n_eph) * len_group
        arity = params.function_set[genes[node_gene_idx]].arity
        input_node_idxs = genes[node_gene_idx+1:node_gene_idx+1+arity]
        for input_node_idx in reversed(input_node_idxs):
            stack.append(input_node_idx)

    if len(active_path) > 0:
        active_paths.append(list(reversed(active_path)))

    return active_paths


class Node:
    def __init__(self, no, func, arity, inputs=[], start_gidx=None):
        self.no = no
        self.func = func
        self.arity = arity
        self.inputs = inputs
        self.value = None

        self.is_input = False
        self.is_output = False
        if func is None:
            if len(self.inputs) == 0:
                self.is_input = True
            else:
                self.is_output = True

        self.start_gidx = start_gidx

    def __repr__(self):
        return f'Node({self.no}, {self.func}, {self.inputs})'


def create_genes_and_bounds(cp: CGPParameters):
    genes = []
    uppers, lowers = [], []
    for i in range(cp.n_input + cp.n_eph, cp.n_input + cp.n_eph + cp.n_fnode):
        f_gene = random.randint(0, cp.n_f - 1)

        lowers.append(0)
        uppers.append(cp.n_f - 1)
        genes.append(f_gene)

        # next bits are input of the node function.
        col = (i - cp.n_input - cp.n_eph) // cp.n_row
        up = cp.n_input + cp.n_eph + col * cp.n_row - 1
        low = max(0, up - cp.levels_back)
        for i_arity in range(cp.max_arity):
            lowers.append(low)
            uppers.append(up)
            in_gene = random.randint(low, up)
            genes.append(in_gene)
    # output genes
    up = cp.n_input + cp.n_eph + cp.n_fnode - 1
    low = max(0, up - cp.levels_back)
    for i in range(cp.n_output):
        lowers.append(low)
        uppers.append(up)
        out_gene = random.randint(low, up)
        genes.append(out_gene)

    return genes, (lowers, uppers)


def create_nodes(cp, genes):
    nodes = []
    for i in range(cp.n_input + cp.n_eph):
        nodes.append(Node(i, None, 0, []))

    f_pos = 0
    for i in range(cp.n_fnode):
        f_gene = genes[f_pos]
        f = cp.function_set[f_gene]
        input_genes = genes[f_pos + 1: f_pos + f.arity + 1]
        nodes.append(Node(i + cp.n_input + cp.n_eph, f, f.arity, input_genes, start_gidx=f_pos))
        f_pos += cp.max_arity + 1

    idx_output_node = cp.n_input + cp.n_eph + cp.n_fnode
    for gene in genes[-cp.n_output:]:
        nodes.append(Node(idx_output_node, None, 0, [gene], start_gidx=f_pos))
        f_pos += 1
        idx_output_node += 1

    return nodes


def report(indiv=None, gen=None):
    def _sub(flist):
        str_list = []
        for f in flist:
            str_list.append(str(f)[:10]+'..')
        return str_list

    if indiv:
        print('|', format(gen, ' ^10'), '|', format(str(indiv.fitness)[:10]+'..', ' ^24'),
              '|', format(str(_sub(indiv.fitness_list)), ' ^80'), '|',
              format(str(indiv.get_cgp_expressions())[:60]+'..', ' ^80'), '|')
    else:
        print(format('', '_^207'))
        print('|', format('Gen', ' ^10'), '|', format('BestFitness', ' ^24'),
              '|', format('BestFitnessList', ' ^80'), '|', format('BestExpression', ' ^80'), '|')


def save_checkpoint(population, conv_f, checkpoint_dir):
    io.mkdir(checkpoint_dir)
    pop_dir = os.path.join(checkpoint_dir, 'populations')
    io.mkdir(pop_dir)

    population = sorted(population, key=lambda x: x.fitness)
    for i, indiv in enumerate(population):
        with open(os.path.join(pop_dir, 'SRNet_{}'.format(i)), 'wb') as f:
            pickle.dump(indiv, f)

    np.savetxt(os.path.join(checkpoint_dir, 'conv_f'), conv_f)