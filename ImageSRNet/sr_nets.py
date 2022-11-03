import copy
import random
from functools import partial

import sympy as sp

import torch
from torch import nn

# from ImageSRNet.image_functions import default_functions
# from ImageSRNet.sr_objs import ImageCGP, CGPLayer
from image_functions import default_functions
from sr_objs import ImageCGP, CGPLayer

class CNNCGPNet(nn.Module):
    # wrap of the ImageCGPNet and CGPNet for explaining CNN model
    def __init__(self, cnn_channels, cnn_input_sizes, cnn_output_sizes, active_pool,
                 mlp_input, mlp_hiddens, mlp_output, params:dict,
                 image_cgpnet=None, cgpnet=None):
        super(CNNCGPNet, self).__init__()
        self.cnn_channels = cnn_channels
        self.cnn_input_sizes = cnn_input_sizes
        self.cnn_output_sizes = cnn_output_sizes
        self.active_pool = active_pool
        self.mlp_input = mlp_input
        self.mlp_hiddens = mlp_hiddens
        self.mlp_output = mlp_output
        self.params = params

        self.image_cgpnet = image_cgpnet
        self.cgpnet = cgpnet
        if self.image_cgpnet is None:
            self.image_cgpnet = ImageCGPNet(self.cnn_channels, self.cnn_input_sizes, self.cnn_output_sizes, self.params,
                                            active_pool=self.active_pool)
        if self.cgpnet is None:
            self.cgpnet = CGPNet(self.mlp_input, self.mlp_output, self.mlp_hiddens, self.params)

    def forward(self, input_image):
        output_imgs = self.image_cgpnet(input_image)
        mlp_input = output_imgs[-1].reshape(output_imgs[-1].shape[0], -1)
        mlp_outputs = self.cgpnet(mlp_input)
        return output_imgs + mlp_outputs

    def get_cnn_expressions(self, var_names=None, active_pool_name=['MaxPool_{(2,2)}', 'ReLU']):
        return self.image_cgpnet.__repr__(var_names, active_pool_name)

    def get_mlp_expressions(self, var_names=None):
        return self.cgpnet.__repr__(var_names)

    def mutate(self, prob):
        return CNNCGPNet(self.cnn_channels, self.cnn_input_sizes, self.cnn_output_sizes, self.active_pool,
                         self.mlp_input, self.mlp_hiddens, self.mlp_output, self.params,
                         self.image_cgpnet.mutate(prob), self.cgpnet.mutate(prob))


class ImageCGPNet(nn.Module):
    def __init__(self, channels_list, input_size_list, output_size_list, params: dict,
                 image_cgps=None,
                 active_pool=nn.Sequential(nn.ReLU(), nn.MaxPool2d((2, 2)))):
        super(ImageCGPNet, self).__init__()
        assert len(channels_list)-1 == len(input_size_list) == len(output_size_list)
        self.channels_list = channels_list
        self.input_size_list = input_size_list
        self.output_size_list = output_size_list

        self.n_input = channels_list[0]
        self.hidden_channels = channels_list[1:-1]
        self.n_output = channels_list[-1]

        self.params = params
        self.active_pool = active_pool
        self.n_conv_layer = len(self.channels_list) - 1
        if image_cgps is None:
            for i, out_channel in enumerate(channels_list[1:]):
                in_channel = channels_list[i]
                input_size, output_size = self.input_size_list[i], self.output_size_list[i]
                setattr(
                    self, 'conv_chrome{}'.format(i+1),
                    ImageCGP(in_channel, out_channel, input_size, output_size, params)
                )
        else:
            assert len(image_cgps) == self.n_conv_layer
            for i, image_cgp in enumerate(image_cgps):
                setattr(
                    self, 'conv_chrome{}'.format(i+1),
                        image_cgp
                )

    def get_cgp_expressions(self):
        return list([getattr(self, 'conv_chrome{}'.format(i+1))[0].get_expressions() for i in range(self.n_conv_layer)])

    def __repr__(self, var_names=None, active_pool_name= None):
        if not var_names:
            var_names = ['IMG_{}'.format(i) for i in range(self.n_input)]
        assert len(var_names) == self.n_input
        exprs = var_names

        for i in range(self.n_conv_layer):
            image_cgp = getattr(self, 'conv_chrome{}'.format(i + 1))
            exprs = image_cgp.get_expressions(var_names=exprs)
            if active_pool_name is not None:
                for ap_oper in active_pool_name:
                    for i, expr in enumerate(exprs):
                        exprs[i] = '{}({})'.format(ap_oper, expr)

        return exprs

    def forward(self, input_img):
        # input_img: [BxCxHxW]
        assert input_img.shape[1] == self.n_input
        chrome_input = input_img
        output_imgs = []
        for i in range(self.n_conv_layer):
            chrome_input = getattr(self, 'conv_chrome{}'.format(i+1))(chrome_input)
            output_imgs.append(chrome_input)
        return output_imgs

    def mutate(self, prob):
        mutated_icgps = [getattr(self, 'conv_chrome{}'.format(i+1)).mutate(prob) for i in range(self.n_conv_layer)]
        return ImageCGPNet(
            self.channels_list, self.input_size_list, self.output_size_list,
            self.params, mutated_icgps, self.active_pool
        )


class CGPNet(nn.Module):
    def __init__(self, n_input, n_output, n_hiddens, params, chromes=None):
        super(CGPNet, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddens = n_hiddens
        self.n_layer = len(n_hiddens)+1
        if chromes is not None:
            assert self.n_layer == len(chromes)
            for i, chrome in enumerate(chromes):
                setattr(self, 'chrome{}'.format(i+1), chrome)
        else:
            n_cgp_input = n_input
            for num, n_hidden in enumerate(n_hiddens):
                setattr(self, 'chrome{}'.format(num+1), nn.Sequential(
                    CGPLayer(n_cgp_input, 1, params),
                    nn.Linear(1, n_hidden)
                ))
                n_cgp_input = n_hidden

            setattr(self, 'chrome{}'.format(self.n_layer), nn.Sequential(
                CGPLayer(n_cgp_input, 1, params),
                nn.Linear(1, n_output)
            ))

        self.fitness, self.fitness_list = None, None

    def forward(self, x):
        input_data = self.chrome1(x)
        outputs = [input_data]

        for i in range(1, self.n_layer):
            input_data = getattr(self, 'chrome{}'.format(i+1))(input_data)
            outputs.append(input_data)

        return outputs

    def predict(self, x):
        outputs = []
        with torch.no_grad():
            input_data = self.chrome1(x)
            outputs.append(input_data)

            for i in range(1, self.n_layer):
                input_data = getattr(self, 'chrome{}'.format(i + 1))(input_data)
                outputs.append(input_data)

        return outputs

    def get_cgp_expressions(self):
        exprs = []
        for i in range(self.n_layer):
            exprs.append(getattr(self, 'chrome{}'.format(i+1))[0].get_expression())
        return exprs

    def mutate(self, prob):
        """TODO: implementing another mutation method"""
        mutated_net = copy.deepcopy(self)
        for i in range(mutated_net.n_layer):
            cgp = getattr(mutated_net, 'chrome{}'.format(i+1))[0]
            low, up = cgp.bounds[0], cgp.bounds[1]

            for gidx in range(len(cgp.genes)):
                chance = random.random()
                if chance < prob:
                    candicates = [gene for gene in range(low[gidx], up[gidx]+1) if gene != cgp.genes[gidx]]
                    if len(candicates) == 0:
                        continue
                    cgp.genes[gidx] = random.choice(candicates)
            cgp.build()

        return mutated_net

    def __repr__(self, var_names=None):
        if not var_names:
            var_names = list([f'x{i}' for i in range(self.n_input)]) if self.n_input > 1 else ['x']
        exprs = var_names

        for i in range(self.n_layer):
            sequential_layer = getattr(self, 'chrome{}'.format(i + 1))
            cgp_layer, linear_layer = sequential_layer[0], sequential_layer[1]

            exprs = cgp_layer.get_expression(input_vars=exprs)
            weight = linear_layer.weight.detach().cpu()
            bias = linear_layer.bias.detach().cpu()

            exprs = sp.Matrix(exprs) * weight.T + bias.reshape(1, -1)

        return str(exprs)


LeNetCGPNet = partial(CNNCGPNet, [1, 6, 16], [(28, 28), (12, 12)], [(24, 24), (8, 8)], nn.Sequential(nn.ReLU(), nn.MaxPool2d((2, 2))),
                      16*4*4, [120, 84], 10)


if __name__ == '__main__':
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
    channels_list = [1, 6, 16]
    input_sizes = [(28, 28), (12, 12)]
    output_sizes = [(24, 24), (8, 8)]
    # icgpnet = ImageCGPNet(channels_list, input_sizes, output_sizes, params)
    # cgp_exprs = icgpnet.get_cgp_expressions()
    # for i, cgp_expr in enumerate(cgp_exprs):
    #     print('ImageCGP{}:{}'.format(i, cgp_expr))
    # print(icgpnet)
    #
    # mutated_icgpnet = ImageCGPNet(channels_list, input_sizes, output_sizes, params)
    # for i, cgp_expr in enumerate(mutated_icgpnet):
    #     print('ImageCGP{}:{}'.format(i, cgp_expr))
    # print(icgpnet)

    mlp_input, mlp_hiddens, mlp_output = 16 * 4 * 4, [120, 84], 10
    cnn_cgpnet = CNNCGPNet(channels_list, input_sizes, output_sizes, nn.Sequential(nn.ReLU(), nn.MaxPool2d((2, 2))),
                           mlp_input, mlp_hiddens, mlp_output, params)
    # outputs = cnn_cgpnet(torch.rand(2, 1, 28, 28))
    # print(list([output.shape for output in outputs]))
    # print(cnn_cgpnet.get_cnn_expressions(), '\n', cnn_cgpnet.get_mlp_expressions())
    for exp in cnn_cgpnet.get_cnn_expressions():
        print(exp)
    print(2)
    mutated = cnn_cgpnet.mutate(0.4)
    outputs = mutated(torch.rand(2, 1, 28, 28))
    print(list([output.shape for output in outputs]))
    print(mutated.get_cnn_expressions(), '\n', mutated.get_mlp_expressions())