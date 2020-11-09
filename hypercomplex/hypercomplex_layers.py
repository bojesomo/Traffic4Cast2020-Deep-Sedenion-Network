##########################################################
# pytorch-qnn v1.0
# Alabi Bojesomo
# Khalifa University
# Abu Dhabi, UAE
# August 2020
##########################################################

import numpy                   as np
from numpy.random import RandomState
import torch
from torch.autograd import Variable
import torch.nn.functional      as F
import torch.nn                 as nn
from torch.nn.parameter import Parameter
from torch.nn import Module
from .hypercomplex_ops import *
import math
import sys


class HyperTransposeConv(Module):
    r"""Applies an Hypercomplex [complex, quaternion, octonion] Transposed Convolution (or Deconvolution) to
    the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, num_components=8,
                 dilatation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='hypercomplex', seed=None, operation='convolution2d',  # rotation=False,
                 hypercomplex_format=False):

        super(HyperTransposeConv, self).__init__()

        self.num_components = num_components
        self.in_channels = in_channels // self.num_components
        self.out_channels = out_channels // self.num_components
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        # self.rotation = rotation
        self.hypercomplex_format = hypercomplex_format
        self.winit = {'hypercomplex': hypercomplex_init,
                      'unitary': unitary_init,
                      'random': random_init}[self.weight_init]

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(self.operation,
                                                                       self.out_channels, self.in_channels, kernel_size)

        self.weights = []
        for component in range(self.num_components):
            weight = Parameter(torch.Tensor(*self.w_shape))
            setattr(self, f"weight_{component}", weight)
            self.weights.append(weight)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
        #                  self.e_weight, self.l_weight, self.m_weight, self.n_weight,
        #                  self.kernel_size, self.winit, self.rng, self.init_criterion)
        # weight_str = ' ,'.join([f'self.weight_{component}' for component in range(self.num_components)])
        # exec(f"affect_init_conv({weight_str}, self.kernel_size, self.winit, self.rng, self.init_criterion, "
        #      f"self.num_components)")
        # self.weights = [eval(f'self.weight_{component}') for component in range(self.num_components)]
        affect_init_conv(self.weights, self.kernel_size, self.winit, self.rng, self.init_criterion)
        # for component in range(self.num_components):
        #     setattr(self, f'self.weight_{component}.data', weights[component].data)

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):

        # if self.rotation:
        #     return octonion_tranpose_conv_rotation(input, self.weights, self.bias, self.stride, self.padding,
        #                                            self.output_padding, self.groups, self.dilatation,
        #                                            self.octonion_format)
        # else:
        #     return hypercomplex_transpose_conv(input, self.weights,
        #                                    self.bias, self.stride, self.padding, self.output_padding,
        #                                    self.groups, self.dilatation)
        return hypercomplex_transpose_conv(input, self.weights,
                                           self.bias, self.stride, self.padding, self.output_padding,
                                           self.groups, self.dilatation)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', bias=' + str(self.bias is not None) \
               + ', kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', dilation=' + str(self.dilation) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) \
               + ', operation=' + str(self.operation) + ')'


class HyperConv(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None, operation='convolution2d',  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        r"""Applies an Hypercomplex [complex, quaternion, octonion] Convolution (or Deconvolution) to
            the incoming data.
            """
        super(HyperConv, self).__init__()

        self.num_components = num_components
        self.in_channels = in_channels // self.num_components
        self.out_channels = out_channels // self.num_components
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        # self.rotation = rotation
        self.hypercomplex_format = hypercomplex_format
        self.winit = {'hypercomplex': hypercomplex_init,
                      'unitary': unitary_init,
                      'random': random_init}[self.weight_init]
        self.scale = scale

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(self.operation,
                                                                       self.in_channels, self.out_channels, kernel_size)

        # self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        # self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        # self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        # self.k_weight = Parameter(torch.Tensor(*self.w_shape))
        # self.e_weight = Parameter(torch.Tensor(*self.w_shape))
        # self.l_weight = Parameter(torch.Tensor(*self.w_shape))
        # self.m_weight = Parameter(torch.Tensor(*self.w_shape))
        # self.n_weight = Parameter(torch.Tensor(*self.w_shape))
        self.weights = []
        for component in range(self.num_components):
            weight = Parameter(torch.Tensor(*self.w_shape))
            setattr(self, f"weight_{component}", weight)
            self.weights.append(weight)

        if self.scale:
            self.scale_param = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param = None

        # if self.rotation:
        #     self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
        #                  self.e_weight, self.l_weight, self.m_weight, self.n_weight,
        #                  self.kernel_size, self.winit, self.rng, self.init_criterion)
        # if self.scale_param is not None:
        #     torch.nn.init.xavier_uniform_(self.scale_param.data)
        # if self.bias is not None:
        #     self.bias.data.zero_()

        # self.weights = [eval(f'self.weight_{component}') for component in range(self.num_components)]
        affect_init_conv(self.weights, self.kernel_size, self.winit, self.rng, self.init_criterion)
        # for component in range(self.num_components):
        #     setattr(self, f'self.weight_{component}.data', weights[component].data)
        # del weights
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):

        # if self.rotation:
        #     return octonion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
        #                                   self.k_weight, self.e_weight, self.l_weight, self.m_weight, self.n_weight,
        #                                   self.bias, self.stride, self.padding, self.groups, self.dilatation,
        #                                   self.octonion_format, self.scale_param)
        # else:
        #     return octonion_conv(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight,
        #                          self.e_weight, self.l_weight, self.m_weight, self.n_weight,
        #                          self.bias, self.stride, self.padding, self.groups, self.dilatation)
        return hypercomplex_conv(input, self.weights, self.bias, self.stride, self.padding, self.groups, self.dilation)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', bias=' + str(self.bias is not None) \
               + ', kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) \
               + ', h_format=' + str(self.hypercomplex_format) \
               + ', operation=' + str(self.operation) + ')'


class HyperLinear(Module):
    r"""Applies a octonion linear transformation to the incoming data.
    """

    def __init__(self, in_features, out_features, num_components=8, bias=True,
                 init_criterion='he', weight_init='hypercomplex',
                 seed=None):

        super(HyperLinear, self).__init__()
        self.num_components = num_components
        self.in_features = in_features // self.num_components
        self.out_features = out_features // self.num_components

        self.weights = []
        for component in range(self.num_components):
            weight = Parameter(torch.Tensor(self.in_features, self.out_features))
            setattr(self, f"weight_{component}", weight)
            self.weights.append(weight)
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features * num_components))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'hypercomplex': hypercomplex_init,
                 'unitary': unitary_init}[self.weight_init]

        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.weights, winit, self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if input.dim() == 3:
            T, N, C = input.size()
            input = input.view(T * N, C)
            output = HyperLinearFunction.apply(input, self.bias, *self.weights)
            # output = hypercomplex_linear(input, self.weights, self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = HyperLinearFunction.apply(input, self.bias, *self.weights)
            # output = hypercomplex_linear(input, self.weights, self.bias)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', seed=' + str(self.seed) + ')'


class HyperConv1D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(HyperConv1D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=2, operation='convolution1d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(HyperConv1D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class HyperConv2D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(HyperConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=2, operation='convolution2d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(HyperConv2D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class HyperConv3D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(HyperConv3D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=2, operation='convolution3d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(HyperConv3D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class ComplexConv1D(HyperConv):

        def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                     dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                     weight_init='hypercomplex', seed=None, # rotation=False,
                     hypercomplex_format=True,
                     scale=False):
            super(ComplexConv1D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                dilation=dilation, padding=padding,
                                                groups=groups, bias=bias, init_criterion=init_criterion,
                                                weight_init=weight_init, seed=seed,
                                                num_components=2, operation='convolution1d',  # rotation=False,
                                                hypercomplex_format=hypercomplex_format,
                                                scale=scale)

        def __repr__(self):
            config = super(ComplexConv1D, self).__repr__()
            config.replace(f', operation={str(self.operation)}', '')
            return config


class ComplexConv2D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(ComplexConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=2, operation='convolution2d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(ComplexConv2D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class ComplexConv3D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(ComplexConv3D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=2, operation='convolution3d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(ComplexConv3D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class QuaternionConv1D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(QuaternionConv1D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=4, operation='convolution1d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(QuaternionConv1D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class QuaternionConv2D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(QuaternionConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=4, operation='convolution2d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(QuaternionConv2D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class QuaternionConv3D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(QuaternionConv3D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=4, operation='convolution3d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(QuaternionConv3D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class OctonionConv1D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(OctonionConv1D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=8, operation='convolution1d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(OctonionConv1D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class OctonionConv2D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(OctonionConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=8, operation='convolution2d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(OctonionConv2D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class OctonionConv3D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(OctonionConv3D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=8, operation='convolution3d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(OctonionConv3D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class SedanionConv1D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(SedanionConv1D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=16, operation='convolution1d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
            config = super(SedanionConv1D, self).__repr__()
            config.replace(f', operation={str(self.operation)}', '')
            return config


class SedanionConv2D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(SedanionConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=16, operation='convolution2d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(SedanionConv2D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class SedanionConv3D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', seed=None,  # rotation=False,
                 hypercomplex_format=True,
                 scale=False):
        super(SedanionConv3D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias, init_criterion=init_criterion,
                                            weight_init=weight_init, seed=seed,
                                            num_components=16, operation='convolution3d',  # rotation=False,
                                            hypercomplex_format=hypercomplex_format,
                                            scale=scale)

    def __repr__(self):
        config = super(SedanionConv3D, self).__repr__()
        config.replace(f', operation={str(self.operation)}', '')
        return config


class ComplexLinear(HyperLinear):

    def __init__(self, in_features, out_features,  # num_components=8,
                 bias=True,
                 init_criterion='he', weight_init='hypercomplex',
                 seed=None):
        super(ComplexLinear, self).__init__(in_features, out_features, num_components=2, bias=bias,
                                            init_criterion=init_criterion, weight_init=weight_init,
                                            seed=seed)


class QuaternionLinear(HyperLinear):

    def __init__(self, in_features, out_features,  # num_components=8,
                 bias=True,
                 init_criterion='he', weight_init='hypercomplex',
                 seed=None):
        super(QuaternionLinear, self).__init__(in_features, out_features, num_components=4, bias=bias,
                                               init_criterion=init_criterion, weight_init=weight_init,
                                               seed=seed)


class OctonionLinear(HyperLinear):

    def __init__(self, in_features, out_features,  # num_components=8,
                 bias=True,
                 init_criterion='he', weight_init='hypercomplex',
                 seed=None):
        super(OctonionLinear, self).__init__(in_features, out_features, num_components=8, bias=bias,
                                             init_criterion=init_criterion, weight_init=weight_init,
                                             seed=seed)


class SedanionLinear(HyperLinear):

    def __init__(self, in_features, out_features,  # num_components=8,
                 bias=True,
                 init_criterion='he', weight_init='hypercomplex',
                 seed=None):
        super(SedanionLinear, self).__init__(in_features, out_features, num_components=16, bias=bias,
                                             init_criterion=init_criterion, weight_init=weight_init,
                                             seed=seed)




