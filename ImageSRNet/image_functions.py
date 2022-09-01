from functools import partial

import sympy as sp
import torch
import kornia as K


def image_feature_select(input_image, feature_info):
    # input_image: [B, C, R, C]; feature_info: (left_up_pos, size)
    left_up_pos, size = feature_info
    row_pos, col_pos = left_up_pos
    row_size, col_size = size

    return input_image[:, :, row_pos:row_pos + row_size, col_pos:col_pos + col_size]


def protected_div(x0, x1, symbol=False):
    if symbol:
        if isinstance(x1, sp.Float) or isinstance(x1, float):
            return 0. if abs(x1) <= 1e-5 else x0 / x1
        return x0 / x1

    return torch.where(torch.abs(x1) <= torch.tensor(1e-5), torch.tensor(0.), torch.div(x0, x1))

    # if symbol:
    #     if isinstance(x1, float):
    #         return 0. if abs(x1) <= 1e-5 else x0 / x1
    #     if not isinstance(x1, sp.Symbol) and x1.is_number and sp.Abs(x1) <= 1e-5:
    #         return 0.
    #     return x0 / x1

    # return torch.where(torch.abs(x1) <= torch.tensor(1e-5), torch.tensor(0.), torch.div(x0, x1))


def protected_sqrt(x0, symbol=False):
    if symbol:
        if isinstance(x0, sp.Float) or isinstance(x0, float):
            return x0 if x0 < 0 else sp.sqrt(x0)
        return sp.sqrt(x0)

    return torch.where(x0 < torch.tensor(0.), x0, torch.sqrt(x0))

    # if symbol:
    #     if isinstance(x0, float):
    #         return x0 if x0 < 0 else sp.sqrt(x0)
    #     if not isinstance(x0, sp.Symbol) and x0.is_number and x0 < 0:
    #         return x0
    #     return sp.sqrt(x0)
    #
    # return torch.where(x0 < torch.tensor(0.), x0, torch.sqrt(x0))


def protected_ln(x0, symbol=False):
    if symbol:
        if isinstance(x0, sp.Float) or isinstance(x0, float):
            return x0 if x0 <= 1e-5 else sp.log(x0)
        return sp.log(x0)
        
    return torch.where(x0 <= torch.tensor(1e-5), x0, torch.log(x0))
    # if symbol:
    #     if isinstance(x0, float):
    #         return x0 if x0 <= 1e-5 else sp.log(x0)
    #     if not isinstance(x0, sp.Symbol) and x0.is_number and x0 <= 1e-5:
    #         return x0
    #     return sp.log(x0)

    # return torch.where(x0 <= torch.tensor(1e-5), x0, torch.log(x0))


def identify(x0):
    return x0


def sp_add(x0, x1):
    return x0 + x1


def sp_sub(x0, x1):
    return x0 - x1


def sp_mul(x0, x1):
    return x0 * x1


def sp_sqre(x0):
    return x0 ** 2


class BaseFunction:
    def __init__(self, func_name, arity, pt_func, sp_func, params_range=None):
        self.func_name = func_name
        self.arity = arity
        self.pt_func = pt_func
        self.sp_func = sp_func
        self.params_range = params_range

    def expr(self, *var_names, feature_infos=None, params=None):
        sym_vars = []
        has_params = params is not None
        if feature_infos is None:
            for var in var_names:
                sym_vars.append(sp.Symbol(var) if isinstance(var, str) else var)
            return self.sp_func(*sym_vars, params=params) if has_params else self.sp_func(*sym_vars)
        for var, feature_info in zip(var_names, feature_infos):
            if feature_info is not None:
                sp_var = sp.Symbol('(({})_{{{}}})'.format(var, feature_info))
            else:
                sp_var = sp.Symbol(var) if isinstance(var, str) else var
            sym_vars.append(sp_var)
        return self.sp_func(*sym_vars, params=params) if has_params else self.sp_func(*sym_vars)

    def __call__(self, *args, feature_infos=None, params=None):
        select = feature_infos is not None
        has_params = params is not None
        vinputs = []
        for input_idx in range(self.arity):
            vinputs.append(
                image_feature_select(args[input_idx], feature_infos[input_idx])
                if select and feature_infos[input_idx] is not None and len(args[input_idx].shape) == 4
                else args[input_idx]
            )

        return self.pt_func(*vinputs, params=params) if has_params else self.pt_func(*vinputs)


add = BaseFunction('Add', 2, torch.add, sp_add)
sub = BaseFunction('Sub', 2, torch.sub, sp_sub)
mul = BaseFunction('Mul', 2, torch.mul, sp_mul)
div = BaseFunction('Div', 2, protected_div, partial(protected_div, symbol=True))
sqrt = BaseFunction('Sqrt', 1, protected_sqrt, partial(protected_sqrt, symbol=True))
sqre = BaseFunction('Sqre', 1, torch.square, sp_sqre)
exp = BaseFunction('Exp', 1, torch.exp, sp.exp)
ln = BaseFunction('Ln', 1, protected_ln, sp.log)
sin = BaseFunction('Sin', 1, torch.sin, sp.sin)
cos = BaseFunction('Cos', 1, torch.cos, sp.cos)
tan = BaseFunction('Tan', 1, torch.tan, sp.tan)
# this is for when the output is the original input image
idtf = BaseFunction('Identify', 1, identify, identify)


def image_operator_expr(*args, params=None, func_name=None):
    if isinstance(args[0], float):
        return args[0]
    if params is None:
        return '{}({})'.format(func_name, args[0])
    return '{}^{{{}}}({})'.format(func_name, params, args[0])


def gaussian_pt_func(*args, params=None):
    k_size, sigma = params
    return K.filters.gaussian_blur2d(args[0], (k_size, k_size), (sigma, sigma)) if len(args[0].shape) != 0 else args[0]


def laplacian_pt_func(*args, params=None):
    k_size, = params
    return K.filters.laplacian(args[0], k_size) if len(args[0].shape) != 0 else args[0]


def blur_pool_pt_func(*args, params=None):
    k_size, = params
    return K.filters.blur_pool2d(args[0], k_size) if len(args[0].shape) != 0 else args[0]


def max_pool_pt_func(*args, params=None):
    k_size, = params
    return K.filters.max_blur_pool2d(args[0], k_size) if len(args[0].shape) != 0 else args[0]


def median_blur_pt_func(*args, params=None):
    k_size, = params
    return K.filters.median_blur(args[0], (k_size, k_size)) if len(args[0].shape) != 0 else args[0]


def mean_blur_pt_func(*args, params=None):
    k_size, = params
    return K.filters.box_blur(args[0], (k_size, k_size)) if len(args[0].shape) != 0 else args[0]


def unsharp_mask_pt_func(*args, params=None):
    k_size, sigma = params
    return K.filters.unsharp_mask(args[0], (k_size, k_size), (sigma, sigma)) if len(args[0].shape) != 0 else args[0]


def canny_pt_func(*args, params=None):
    if len(args[0].shape) != 0:
        magnitude, edges = K.filters.canny(args[0])
        return magnitude
    else:
        return args[0]


def sobel_pt_func(*args, params=None):
    return K.filters.sobel(args[0]) if len(args[0].shape) != 0 else args[0]


def spatial_gradient_x_pt_func(*args, params=None):
    return K.filters.spatial_gradient(args[0])[:, :, 0] if len(args[0].shape) != 0 else args[0]


def spatial_gradient_y_pt_func(*args, params=None):
    return K.filters.spatial_gradient(args[0])[:, :, 1] if len(args[0].shape) != 0 else args[0]


def dilation_pt_func(*args, params=None):
    k_size, = params
    if len(args[0].shape) != 0:
        kernel = torch.ones(k_size, k_size)
        return K.morphology.dilation(args[0], kernel)
    else:
        return args[0]


def erosion_pt_func(*args, params=None):
    k_size, = params
    if len(args[0].shape) != 0:
        kernel = torch.ones(k_size, k_size)
        return K.morphology.erosion(args[0], kernel)
    else:
        return args[0]


def opening_pt_func(*args, params=None):
    k_size, = params
    if len(args[0].shape) != 0:
        kernel = torch.ones(k_size, k_size)
        return K.morphology.opening(args[0], kernel)
    else:
        return args[0]


def closing_pt_func(*args, params=None):
    k_size, = params
    if len(args[0].shape) != 0:
        kernel = torch.ones(k_size, k_size)
        return K.morphology.closing(args[0], kernel)
    else:
        return args[0]


def gradient_pt_func(*args, params=None):
    k_size, = params
    if len(args[0].shape) != 0:
        kernel = torch.ones(k_size, k_size)
        return K.morphology.gradient(args[0], kernel)
    else:
        return args[0]


def top_hat_pt_func(*args, params=None):
    k_size, = params
    if len(args[0].shape) != 0:
        kernel = torch.ones(k_size, k_size)
        return K.morphology.top_hat(args[0], kernel)
    else:
        return args[0]


def bottom_hat_pt_func(*args, params=None):
    k_size, = params
    if len(args[0].shape) != 0:
        kernel = torch.ones(k_size, k_size)
        return K.morphology.bottom_hat(args[0], kernel)
    else:
        return args[0]


k_size_range, sigma_range = [3, 5, 7], [0.3, 0.6, 1.0]
gaussian = BaseFunction('Gaussian', 1, gaussian_pt_func, partial(image_operator_expr, func_name='Gaussian'), (k_size_range, sigma_range))
laplacian = BaseFunction('Laplacian', 1, laplacian_pt_func, partial(image_operator_expr, func_name='Laplacian'), (k_size_range,))
blur_pool = BaseFunction('BlurPool', 1, blur_pool_pt_func, partial(image_operator_expr, func_name='BlurPool'), (k_size_range,))
max_blur_pool = BaseFunction('MaxBlurPool', 1, max_pool_pt_func, partial(image_operator_expr, func_name='MaxBlurPool'), (k_size_range,))
mean_blur = BaseFunction('MeanBlur', 1, mean_blur_pt_func, partial(image_operator_expr, func_name='MeanBlur'), (k_size_range,))
median_blur = BaseFunction('MedianBlur', 1, median_blur_pt_func, partial(image_operator_expr, func_name='MedianBlur'), (k_size_range,))
unsharp_mask = BaseFunction('UnsharpMask', 1, unsharp_mask_pt_func, partial(image_operator_expr, func_name='UnsharpMask'), (k_size_range, sigma_range))
canny = BaseFunction('Canny', 1, canny_pt_func, partial(image_operator_expr, func_name='Canny'))
sobel = BaseFunction('Sobel', 1, sobel_pt_func, partial(image_operator_expr, func_name='Sobel'))
spatial_gradient_x = BaseFunction('SpatialGradientX', 1, spatial_gradient_x_pt_func, partial(image_operator_expr, func_name='SpatialGradientX'))
spatial_gradient_y = BaseFunction('SpatialGradientY', 1, spatial_gradient_y_pt_func, partial(image_operator_expr, func_name='SpatialGradientY'))
dilation = BaseFunction('Dilation', 1, dilation_pt_func, partial(image_operator_expr, func_name='Dilation'), (k_size_range,))
erosion = BaseFunction('Erosion', 1, erosion_pt_func, partial(image_operator_expr, func_name='Erosion'), (k_size_range,))
opening = BaseFunction('Opening', 1, opening_pt_func, partial(image_operator_expr, func_name='Opening'), (k_size_range,))
closing = BaseFunction('Closing', 1, closing_pt_func, partial(image_operator_expr, func_name='Closing'), (k_size_range,))
gradient = BaseFunction('Gradient', 1, gradient_pt_func, partial(image_operator_expr, func_name='Gradient'), (k_size_range,))
top_hat = BaseFunction('TopHat', 1, top_hat_pt_func, partial(image_operator_expr, func_name='TopHat'), (k_size_range,))
bottom_hat = BaseFunction('BottomHat', 1, bottom_hat_pt_func, partial(image_operator_expr, func_name='BottomHat'), (k_size_range,))

function_map = {
    'add': add,
    'sub': sub,
    'mul': mul,
    'div': div,
    'sqrt': sqrt,
    'sqre': sqre,
    'exp': exp,
    'ln': ln,
    'sin': sin,
    'cos': cos,
    'tan': tan,
    'id': idtf,

    # IP functions
    'gaussian': gaussian,
    'laplacian': laplacian,
    'blur_pool': blur_pool,
    'max_blur_pool': max_blur_pool,
    'mean_blur': mean_blur,
    'median_blur': median_blur,
    'unsharp': unsharp_mask,
    'canny': canny,
    'sobel': sobel,
    'spatial_grad_x': spatial_gradient_x,
    'spatial_grad_y': spatial_gradient_y,
    'dilation': dilation,
    'erosion': erosion,
    'opening': opening,
    'closing': closing,
    'grad': gradient,
    'top_hat': top_hat,
    'bottom_hat': bottom_hat
}

default_functions = [
    'add', 'sub', 'mul', 'div', 'sqrt', 'sqre', 'ln', 'sin', 'cos', 'tan', 'id',
    'gaussian', 'laplacian', 'mean_blur', 'median_blur', 'unsharp', 'canny', 'sobel', 'spatial_grad_x', 'spatial_grad_y',
    'dilation', 'erosion', 'opening', 'closing', 'grad', 'top_hat', 'bottom_hat'
]

default_pooling_functions = ['blur_pool', 'max_blur_pool']