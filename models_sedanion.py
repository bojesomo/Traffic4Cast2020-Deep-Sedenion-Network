# coding:utf-8
##########################################################
# pytorch v1.0
# Alabi Bojesomo
# Khalifa University
# Abu Dhabi, UAE
# August 2020
##########################################################

import os
import tempfile
import time
import pickle

import settings
from utils import time_to_str, compute_padding, Swish, R2_SCORE, MAAPE, NDEI, RMSE, RMSLE, MSE, MAE

import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.nn import Conv2d, Linear, BatchNorm2d, Upsample, GroupNorm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn import Conv2d as RealConv2D

from hypercomplex import ComplexConv2D, QuaternionConv2D, OctonionConv2D, SedanionConv2D
from hypercomplex import ComplexLinear, QuaternionLinear, OctonionLinear, SedanionLinear, get_c

torch.cuda.empty_cache()

ROW_AXIS = 2
COL_AXIS = 3
CHANNEL_AXIS = 1
BATCH_AXIS = 0
activation_dict = {'relu': nn.ReLU(),
                   'relu6': nn.ReLU6(),
                   'prelu': nn.PReLU(),
                   'hardtanh': nn.Hardtanh(),
                   'tanh': nn.Tanh(),
                   'elu': nn.ELU(),
                   'selu': nn.SELU(),
                   'gelu': nn.GELU(),
                   'glu': nn.GLU(),
                   'swish': Swish(),
                   'sigmoid': nn.Sigmoid(),
                   'leakyrelu': nn.LeakyReLU(),
                   # 'hardsigmoid': nn.Hardsigmoid(),
                   'softsign': nn.Softsign(),
                   'softplus': nn.Softplus,
                   'softmin': nn.Softmin(),
                   'softmax': nn.Softmax()}
optimizer_dict = {'adadelta': optim.Adadelta,
                  'adagrad': optim.Adagrad,
                  'adam': optim.Adam,
                  'adamw': optim.AdamW,
                  'sparse_adam': optim.SparseAdam,
                  'adamax': optim.Adamax,
                  'asgd': optim.ASGD,
                  'sgd': optim.SGD,
                  'rprop': optim.Rprop,
                  'rmsprop': optim.RMSprop,
                  'lbfgs': optim.LBFGS}
criterion_dict = {'mae': nn.L1Loss(),
                  'mse': nn.MSELoss(),
                  'bce': nn.BCELoss(),
                  'smoothl1': nn.SmoothL1Loss(),
                  'binary_crossentropy': nn.BCELoss(),
                  'categorical_crossentropy': nn.CrossEntropyLoss(),
                  }


def setup_model():
    global batch_size, times_out, n_divs, sf, nb_layers, opts, sf_grp
    global convArgs, bnArgs, actArgs, classifier_actArgs, other_args
    global n_frame_in, n_frame_out, n_channels, n_channels_out, frame_shape
    global data_root, city_name, opt_type, net_type
    global ModelConv2D, ModelBN, ModelLinear, concatenate_m, ModelBlock

    opts = settings.options

    stack_input = opts.stack_input
    city_name = opts.city.upper()

    net_dict = {'seda': 'sedanion', 'octo': 'octonion', 'quat': 'quaternion', 'comp': 'complex', 'real': 'real'}
    net_type = net_dict[opts.net_type.lower()[:4]]
    # Details here
    hidden_activation = opts.hidden_activation
    classifier_activation = opts.classifier_activation
    convArgs = {"padding": "same", "bias": False,
                "weight_init": 'hypercomplex'}
    l2Args = {'weight_decay': opts.weight_decay}
    bnArgs = {"momentum": 0.9, "eps": 1e-04}
    actArgs = {"activation": hidden_activation}  # "elu"}
    # classifier_actArgs = {"activation": "sigmoid"}
    classifier_actArgs = {"activation": classifier_activation}

    n_frame_in = opts.n_frame_in
    n_frame_out = opts.n_frame_out
    times_out = [int(t) for t in opts.times_out.split(':')]
    n_channels = opts.n_channels
    n_channels_out = opts.n_channels_out
    h, w = opts.image_height, opts.image_width

    sh, sw = opts.scale_height, opts.scale_width
    h //= sh
    w //= sw

    frame_shape = (h, w)
    static_shape = (h, w, 7)

    epochs = opts.epochs
    batch_size = opts.batch_size

    opt_type = opts.optimizer
    # other_args = {}
    if opt_type == 'sgd':
        other_args = {'lr': opts.lr, 'momentum': opts.momentum, 'nesterov': True, **l2Args}  # 'weight_decay': 1e-4}
    elif opt_type == 'adam':
        other_args = {'lr': opts.lr, 'eps': opts.epsilon, 'betas': (opts.beta_1, opts.beta_2),
                      **l2Args}  # 'weight_decay': 1e-4}

    nb_layers = opts.nb_layers

    data_root = opts.data_root

    if net_type == 'sedanion':
        n_divs = 16
    elif net_type == 'octonion':
        n_divs = 8
    elif net_type == 'quaternion':
        n_divs = 4
    elif net_type == 'complex':
        n_divs = 2
    elif net_type == 'real':
        n_divs = 1

    sf = opts.sf
    sf_grp = opts.sf_grp

    # ModelConv2D, ModelBN, ModelLinear, concatenate_m = None, None, None, None
    def concatenate_m(x):
        O_components = [torch.cat([get_c(x_i, component, n_divs) for x_i in x], dim=1) for component in range(n_divs)]
        return torch.cat(O_components, dim=1)

    def concatenate_real(x):
        return torch.cat(x, dim=1)

    if net_type == 'sedanion':
        ModelConv2D = SedanionConv2D
        ModelBN = BatchNorm2d  # SedanionBN
        ModelLinear = SedanionLinear
        concatenate_m = concatenate_m
    elif net_type == 'octonion':
        ModelConv2D = OctonionConv2D
        ModelBN = BatchNorm2d  # OctonionBN
        ModelLinear = OctonionLinear
        concatenate_m = concatenate_m
    elif net_type == 'quaternion':
        ModelConv2D = QuaternionConv2D
        ModelBN = BatchNorm2d  # QuaternionBN
        ModelLinear = QuaternionLinear
        concatenate_m = concatenate_m
    elif net_type == 'complex':
        ModelConv2D = ComplexConv2D
        ModelBN = BatchNorm2d  # ComplexBN
        ModelLinear = ComplexLinear
        concatenate_m = concatenate_m
    elif net_type == 'real':
        ModelConv2D = RealConv2D
        ModelBN = BatchNorm2d
        ModelLinear = Linear
        concatenate_m = concatenate_real
        _ = convArgs.pop('weight_init')  # use default initializer for real net

    if opts.blk_type == 'resnet':
        ModelBlock = ResidualBlock
    elif opts.blk_type == 'densenet':
        ModelBlock = DenseBlock


def Activation(activation):
    act = activation_dict[activation]
    if hasattr(act, 'inplace'):
        act.inplace = True if opts.modify_activation else act.inplace
    if hasattr(act, 'min_val'):
        act.min_val = 0 if opts.modify_activation else act.min_val
    return act


class LearnVectorBlock(nn.Module):
    def __init__(self, input_shape, featmaps, filter_size, actArgs, bnArgs, block_i=1):
        super(LearnVectorBlock, self).__init__()
        self.block_i = block_i
        self.input_shape = input_shape
        [_, num_features1, H_in, W_in] = input_shape
        out_channels = featmaps

        self.bn1 = BatchNorm2d(num_features=num_features1, **bnArgs)
        self.act = Activation(**actArgs)

        in_channels1 = self.bn1.num_features
        pH = compute_padding(H_in, H_in, filter_size[0], 1)
        pW = compute_padding(W_in, W_in, filter_size[-1], 1)
        padding1 = (pH, pW)
        self.conv1 = Conv2d(in_channels=in_channels1, out_channels=featmaps, kernel_size=filter_size,
                            bias=False, padding=padding1)

        in_channels2 = self.conv1.out_channels
        self.bn2 = BatchNorm2d(num_features=in_channels2, **bnArgs)

        self.conv2 = Conv2d(in_channels=in_channels2, out_channels=featmaps, kernel_size=filter_size,
                            bias=False, padding=padding1)
        self.output_shape = [None, featmaps, H_in, W_in]

    def forward(self, x):

        e1 = self.act(self.bn1(x))
        e1 = self.conv1(e1)

        e1 = self.act(self.bn2(e1))

        e1 = self.conv2(e1)

        return e1

    def name(self):
        name_p = self.__str__().split('(')[0]
        return f"{name_p}_{self.block_i}"


def Add(x_in):
    x_sum = 0
    for x in x_in:
        x_sum += x
    return x_sum


class GroupNormBlock(nn.Module):
    def __init__(self, num_channels):
        super(GroupNormBlock, self).__init__()
        if n_divs == 1:  # net_type is real
            # self.gn = BatchNorm2d(num_features=num_channels, **bnArgs)
            self.gn = GroupNorm(num_groups=32, num_channels=num_channels, eps=bnArgs['eps'])
        else:
            self.gn = GroupNorm(num_groups=n_divs, num_channels=num_channels, eps=bnArgs['eps'])
        self.num_features = num_channels

    def forward(self, x):
        return self.gn(x)

    def name(self):
        return f'group_num_'

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + self.gn.__repr__() + '\n' + ')'


class IdentityShortcut(nn.Module):
    def __init__(self, input_shape, residual_shape):
        super(IdentityShortcut, self).__init__()
        self.input_shape = input_shape
        self.residual_shape = residual_shape
        self.stride_width = int(np.ceil(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
        self.stride_height = int(np.ceil(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
        self.equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

        self.conv = None
        if self.stride_width > 1 or self.stride_height > 1 or not self.equal_channels:
            convArgs_ = convArgs.copy()
            convArgs_['padding'] = (0, 0)
            self.conv = ModelConv2D(in_channels=input_shape[CHANNEL_AXIS],
                                    out_channels=residual_shape[CHANNEL_AXIS], kernel_size=(1, 1),
                                    stride=(self.stride_width, self.stride_height), **convArgs_)
        self.output_shape = residual_shape

    def forward(self, x):  # x = [input_I, residual_R]
        [input_I, residual_R] = x
        shortcut_I = input_I
        if self.stride_width > 1 or self.stride_height > 1 or not self.equal_channels:
            shortcut_I = self.conv(input_I)
        out = Add([shortcut_I, residual_R])
        return out

    def name(self):
        return f'i_shortcut_'

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + self.conv.__repr__() + '\n' + ')'


class ProjectionShortcut(nn.Module):
    def __init__(self, input_shape, residual_shape, featmaps):
        super(ProjectionShortcut, self).__init__()
        self.input_shape = input_shape
        self.residual_shape = residual_shape

        [N_in, C_in, H_out, W_out] = residual_shape
        pH = compute_padding(H_out, input_shape[ROW_AXIS], 1, 2)
        pW = compute_padding(W_out, input_shape[COL_AXIS], 1, 2)
        convArgs_ = convArgs.copy()
        convArgs_['padding'] = (pH, pW)
        self.conv = ModelConv2D(in_channels=input_shape[CHANNEL_AXIS], out_channels=featmaps,
                                 kernel_size=(1, 1), stride=(2, 2), **convArgs_)
        self.output_shape = [N_in, C_in + self.conv.out_channels*n_divs, H_out, W_out]

    def forward(self, x):  # [input_I, residual_R]
        [input_I, residual_R] = x
        e1 = self.conv(input_I)
        return concatenate_m([e1, residual_R])

    def name(self):
        return 'p_shortcut_'

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + self.conv.__repr__() + '\n' + ')'


class ResidualBlock(nn.Module):
    def __init__(self, input_shape, filter_size, featmaps, shortcut, convArgs, bnArgs, actArgs):
        super(ResidualBlock, self).__init__()
        self.input_shape = input_shape
        self.featmaps = featmaps
        self.shortcut_type = shortcut
        pad_type = convArgs['padding']
        convArgs_ = convArgs.copy()
        [N_in, num_features1, H_in, W_in] = input_shape
        if opts.use_group_norm:
            self.bn1 = GroupNormBlock(num_channels=num_features1)
        else:
            self.bn1 = ModelBN(num_features=num_features1, **bnArgs)

        self.act = Activation(actArgs['activation'])

        if self.shortcut_type == 'regular':
            convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
            stride = (1, 1)
            H_out = H_in
            W_out = W_in
        elif self.shortcut_type == 'projection':
            H_out = int(np.ceil(H_in / 2))
            W_out = int(np.ceil(W_in / 2))
            pH = compute_padding(H_out, H_in, filter_size[0], 2)
            pW = compute_padding(W_out, W_in, filter_size[-1], 2)
            convArgs_['padding'] = (pH, pW)
            stride = (2, 2)
        self.conv1 = ModelConv2D(in_channels=num_features1, out_channels=featmaps,
                                 kernel_size=filter_size, stride=stride, **convArgs_)

        # self.bn2 = ModelBN(num_features=self.conv1.out_channels*n_divs, **bnArgs)
        if opts.use_group_norm:
            self.bn2 = GroupNormBlock(num_channels=self.conv1.out_channels*n_divs)
        else:
            self.bn2 = ModelBN(num_features=self.conv1.out_channels*n_divs, **bnArgs)
        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        self.conv2 = ModelConv2D(in_channels=self.conv1.out_channels * n_divs, out_channels=featmaps,
                                 kernel_size=filter_size, stride=(1, 1), **convArgs_)
        residual_shape = [N_in, featmaps, H_out, W_out]

        if shortcut == 'regular':
            self.shortcut = IdentityShortcut(input_shape, residual_shape)
        elif shortcut == 'projection':
            self.shortcut = ProjectionShortcut(input_shape, residual_shape, featmaps)
        self.output_shape = self.shortcut.output_shape
        self.residual_shape = residual_shape

    def forward(self, x):

        e1 = self.conv1(self.act(self.bn1(x)))
        e1 = self.conv2(self.act(self.bn2(e1)))
        e1 = self.shortcut([x, e1])
        return e1  # out

    def name(self):
        return f'residual_block_'


class SubDenseBlock(nn.Module):
    def __init__(self, in_channels, filter_size, featmaps, stride, convArgs_, bnArgs):
        super(SubDenseBlock, self).__init__()
        self.conv = ModelConv2D(in_channels=in_channels, out_channels=featmaps,
                                kernel_size=filter_size, stride=stride, **convArgs_)
        self.act = Activation(actArgs['activation'])
        if opts.use_group_norm:
            self.bn = GroupNormBlock(num_channels=featmaps)
        else:
            self.bn = ModelBN(num_features=featmaps, **bnArgs)

    def forward(self, x):
        return self.bn(self.act(self.conv(x)))


class DenseBlock(nn.Module):
    def __init__(self, input_shape, filter_size, featmaps, convArgs, bnArgs, actArgs, nb_layers=4, pool=True):
        super(DenseBlock, self).__init__()
        self.nb_layers = nb_layers
        self.input_shape = input_shape
        self.featmaps = featmaps
        pad_type = convArgs['padding']
        convArgs_ = convArgs.copy()
        [N_in, num_features1, H_in, W_in] = input_shape
        self.pool = pool
        if self.pool:
            H_out = int(np.ceil(H_in / 2))
            W_out = int(np.ceil(W_in / 2))
            self.avgpool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        else:
            H_out = H_in
            W_out = W_in

        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        stride = (1, 1)
        in_channels = num_features1
        self.layer1 = SubDenseBlock(in_channels, filter_size, featmaps, stride, convArgs_, bnArgs)

        for idx in range(1, nb_layers):
            in_channels += featmaps
            setattr(self, f'layer{idx + 1}', SubDenseBlock(in_channels, filter_size, featmaps,
                                                          stride, convArgs_, bnArgs))

        convArgs_['padding'] = (0, 0)
        in_channels += featmaps
        setattr(self, f'layer{nb_layers + 1}', SubDenseBlock(in_channels, (1, 1), featmaps,
                                                       stride, convArgs_, bnArgs))

        self.output_shape = [N_in, featmaps, H_out, W_out]

    def forward(self, x):
        if self.pool:  # hasattr(self, 'avgpool'):
            x = self.avgpool(x)
        x1 = x
        e1 = self.layer1(x1)
        for idx in range(2, self.nb_layers + 2):
            x1 = concatenate_m([x1, e1])
            exec(f'e1 = self.layer{idx}(x1)')

        return e1  # out

    def name(self):
        return f'dense_block_'


class EncodeBlock(nn.Module):
    def __init__(self, input_shape, num_filters, layer_i, nb_layers=4):
        super(EncodeBlock, self).__init__()
        self.nb_layers = nb_layers
        self.layer_i = layer_i
        # self.blocks = []
        self.input_shape = input_shape
        present_shape = input_shape
        # self.features = nn.Sequential()
        if opts.blk_type == 'resnet':
            for i in range(1, nb_layers):
                blk = ModelBlock(present_shape, (3, 3), num_filters, 'regular',
                                 convArgs, bnArgs, actArgs)
                # self.features.add_module(f'block_{i}', blk)
                setattr(self, f'block_{i}', blk)
                present_shape = blk.output_shape

            blk = ModelBlock(present_shape, (3, 3), num_filters, 'projection',
                             convArgs, bnArgs, actArgs)
            # self.features.add_module(f'block_{nb_layers}', blk)
            setattr(self, f'block_{nb_layers}', blk)
        else:  # densenet
            pool = layer_i != 1
            blk = ModelBlock(present_shape, (3, 3), num_filters,
                             convArgs, bnArgs, actArgs, nb_layers, pool)
            nb_layers = 1
            # self.features.add_module(f'block_{nb_layers}', blk)
            setattr(self, f'block_{nb_layers}', blk)
            
        exec(f"self.output_shape = blk.output_shape")
        if opts.dropout > 0.0:
            self.dropout = nn.Dropout2d(p=opts.dropout, inplace=True)

    def forward(self, x):
        # x = self.features(x)
        for i in range(1, nb_layers + 1):
            x = eval(f'self.block_{i}(x)')
        if opts.dropout > 0.0:
            x = self.dropout(x)
        return x  # self.x

    def name(self):
        return f'encode_block_{self.layer_i}'


class CreateConvBnLayer(nn.Module):
    def __init__(self, input_shape, num_filters, layer_i):
        super(CreateConvBnLayer, self).__init__()
        self.layer_i = layer_i
        self.input_shape = input_shape
        [N_in, C_in, H_in, W_in] = input_shape
        pad_type = convArgs['padding']
        convArgs_ = convArgs.copy()
        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        self.conv = ModelConv2D(in_channels=input_shape[CHANNEL_AXIS], out_channels=num_filters,
                                kernel_size=(3, 3), stride=(1, 1), **convArgs_)
        if opts.use_group_norm:
            self.bn = GroupNormBlock(num_channels=self.conv.out_channels * n_divs)
        else:
            self.bn = ModelBN(num_features=self.conv.out_channels * n_divs, **bnArgs)
        self.act = Activation(**actArgs)
        self.output_shape = [N_in, self.bn.num_features, H_in, W_in]

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def name(self):
        return f'conv_bn_{self.layer_i}'


class DecodeSubBlock(nn.Module):
    def __init__(self, x_shape, y_shape, num_filters, layer_i):
        super(DecodeSubBlock, self).__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.layer_i = layer_i
        self.up = Upsample(scale_factor=2.)
        [N_in, C_y, H_y, W_y] = y_shape

        pad_type = convArgs['padding']
        convArgs_ = convArgs.copy()
        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        self.conv = ModelConv2D(in_channels=x_shape[CHANNEL_AXIS], out_channels=num_filters,
                      kernel_size=(3, 3), stride=(1, 1), **convArgs_)
        if opts.use_group_norm:
            self.bn = GroupNormBlock(num_channels=self.conv.out_channels*n_divs + C_y)
        else:
            self.bn = ModelBN(num_features=self.conv.out_channels*n_divs + C_y, **bnArgs)
        self.act = Activation(**actArgs)
        self.output_shape = [N_in, C_y+num_filters, H_y, W_y]

    def forward(self, x):  # [x_in, y_in]
        [x_in, y_in] = x
        x1 = self.up(x_in)
        x1 = self.conv(x1)

        y_shape = list(y_in.shape)
        hy, wy = y_shape[2:]
        x1 = x1[:, :, :hy, :wy]

        x1 = concatenate_m([x1, y_in])  # xy
        x1 = self.act(self.bn(x1))
        return x1  # xy

    def name(self):
        return f'decode_subblock_{self.layer_i}'


class DecodeBlock(nn.Module):
    def __init__(self, x_shape, y_shape, num_filters, layer_i):
        super(DecodeBlock, self).__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.input_shape = [x_shape, y_shape]
        self.layer_i = layer_i

        self.decode_subblock = DecodeSubBlock(x_shape, y_shape, num_filters, layer_i)
        self.create_conv = CreateConvBnLayer(self.decode_subblock.output_shape, num_filters, layer_i)
        if opts.dropout > 0.0:
            self.dropout = nn.Dropout2d(p=opts.dropout, inplace=True)
        self.output_shape = self.create_conv.output_shape

    def forward(self, x):  # [x_in, y_in]
        e1 = self.decode_subblock(x)
        e1 = self.create_conv(e1)
        if opts.dropout > 0.0:
            e1 = self.dropout(e1)
        return e1

    def name(self):
        return f'decode_block_{self.layer_i}'


class SedanionModel(nn.Module):
    def __init__(self):  # , stack_input=True):
        super(SedanionModel, self).__init__()
        """" implement a model refresher here based on settings.options without model reloading """
        setup_model()

        self.net_type = opts.net_type
        self.is_compiled = False
        self.num_params = 0
        self.dynamic_shape = (None, n_frame_in * n_channels, *frame_shape)
        self.static_shape = (None, 7, *frame_shape)
        self.z0_shape = (None, 16 * n_channels, *frame_shape)
        n_multiples_out = 16 * n_channels_out  # n_divs = 16 for sedanion

        # Stage 1 - Vector learning and preparation
        self.static_learn = LearnVectorBlock(self.static_shape, n_channels, (3, 3), actArgs, bnArgs, block_i=1)
        
        # Stage 2 Encoder
        self.z_enc1 = EncodeBlock(self.z0_shape, sf * 2 ** 0, layer_i=1, nb_layers=nb_layers)
        self.z_enc2 = EncodeBlock(self.z_enc1.output_shape, sf * 2 ** 0, layer_i=2, nb_layers=nb_layers)
        self.z_enc3 = EncodeBlock(self.z_enc2.output_shape, sf * 2 ** 1, layer_i=3, nb_layers=nb_layers)
        self.z_enc4 = EncodeBlock(self.z_enc3.output_shape, sf * 2 ** 1, layer_i=4, nb_layers=nb_layers)
        self.z_enc5 = EncodeBlock(self.z_enc4.output_shape, sf * 2 ** 2, layer_i=5, nb_layers=nb_layers)
        self.z_enc6 = EncodeBlock(self.z_enc5.output_shape, sf * 2 ** 2, layer_i=6, nb_layers=nb_layers)
        self.z_enc7 = EncodeBlock(self.z_enc6.output_shape, sf * 2 ** 3, layer_i=7, nb_layers=nb_layers)
        self.z_enc8 = EncodeBlock(self.z_enc7.output_shape, sf * 2 ** 3, layer_i=8, nb_layers=nb_layers)
        
        # code
        self.z_code = CreateConvBnLayer(self.z_enc8.output_shape, sf * 2 ** 4, layer_i=100)
        
        # Stage 3 - Decoder
        self.z_dec8 = DecodeBlock(self.z_code.output_shape, self.z_enc8.output_shape, sf * 2 ** 3, layer_i=108)
        self.z_dec7 = DecodeBlock(self.z_dec8.output_shape, self.z_enc7.output_shape, sf * 2 ** 3, layer_i=107)
        self.z_dec6 = DecodeBlock(self.z_dec7.output_shape, self.z_enc6.output_shape, sf * 2 ** 2, layer_i=106)
        self.z_dec5 = DecodeBlock(self.z_dec6.output_shape, self.z_enc5.output_shape, sf * 2 ** 2, layer_i=105)
        self.z_dec4 = DecodeBlock(self.z_dec5.output_shape, self.z_enc4.output_shape, sf * 2 ** 1, layer_i=104)
        self.z_dec3 = DecodeBlock(self.z_dec4.output_shape, self.z_enc3.output_shape, sf * 2 ** 1, layer_i=103)
        self.z_dec2 = DecodeBlock(self.z_dec3.output_shape, self.z_enc2.output_shape, sf * 2 ** 0, layer_i=102)
        self.z_dec1 = DecodeBlock(self.z_dec2.output_shape, self.z_enc1.output_shape, sf * 2 ** 0, layer_i=101)
        self.z_dec0 = DecodeBlock(self.z_dec1.output_shape, self.z0_shape, n_multiples_out, layer_i=200)
        
        pad_type = convArgs['padding']
        convArgs_ = convArgs.copy()
        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        self.conv = ModelConv2D(in_channels=self.z_dec0.output_shape[CHANNEL_AXIS],
                                out_channels=n_multiples_out,  # n_divs = 16
                                kernel_size=(3, 3), stride=(1, 1), **convArgs_)
        self.classifier = Activation(**classifier_actArgs)
        
    def forward(self, x, x_static=None):  #x is [dynamic, static]
        if x_static != None:
            dynamic_x = x
            static_x = x_static
        else:
            dynamic_x, static_x = x

        self.static_input = self.static_learn(static_x)
        s_device = self.static_input.device
        s_shape = list(self.static_input.shape)
        s_shape[CHANNEL_AXIS] *= 3
        z0 = torch.cat([self.static_input, dynamic_x, torch.zeros(*s_shape).to(s_device)], dim=1)
        z1 = self.z_enc1(z0)
        z2 = self.z_enc2(z1)
        z3 = self.z_enc3(z2)
        z4 = self.z_enc4(z3)
        z5 = self.z_enc5(z4)
        z6 = self.z_enc6(z5)
        z7 = self.z_enc7(z6)
        z8 = self.z_enc8(z7)

        # code
        z100 = self.z_code(z8)
        
        # Stage 3 - Decoder
        z108 = self.z_dec8([z100, z8])
        z107 = self.z_dec7([z108, z7])
        z106 = self.z_dec6([z107, z6])
        z105 = self.z_dec5([z106, z5])
        z104 = self.z_dec4([z105, z4])
        z103 = self.z_dec3([z104, z3])
        z102 = self.z_dec2([z103, z2])
        z101 = self.z_dec1([z102, z1])
        z200 = self.z_dec0([z101, z0])

        model_output = self.classifier(self.conv(z200))

        model_output = torch.cat([get_c(model_output, idx//5, 16) for idx in times_out], dim=1)
        return model_output

    def name(self):
        return 'my_model'

    def compile(self):
        #, training_dataset, validation_dataset):  #compile(loss=criterion, optimizer=optimizer, metrics=metrics,
        # loss_weights=loss_weights)
        # self.optimizer = optimizer
        self.is_compiled = True
        self.other_args = other_args
        self.batch_size = batch_size

        # self.train_dataset = training_dataset
        # self.val_dataset = validation_dataset

        self.loss = nn.MSELoss()
        self.optimizer = optimizer_dict[opt_type](self.parameters(), **other_args)
        self.metrics = [MAE(), RMSE(), RMSLE(), NDEI(), R2_SCORE(), MAAPE()]


class SedanionModelScaled(nn.Module):
    def __init__(self):  # , stack_input=True):
        super(SedanionModelScaled, self).__init__()
        """" implement a model refresher here based on settings.options without model reloading """
        setup_model()

        self.net_type = opts.net_type
        self.is_compiled = False
        self.num_params = 0
        self.dynamic_shape = (None, n_frame_in * n_channels, *frame_shape)
        self.static_shape = (None, 8, *frame_shape) if opts.use_time_slot else (None, 7, *frame_shape)
        self.z0_shape = (None, 16 * n_channels, *frame_shape)
        assert n_divs in [16, 1]  # only real or sedanion implemented
        if n_divs == 16:
            n_multiples_out = 16 * n_channels_out  # n_divs = 16 for sedanion
        else:
            n_multiples_out = n_frame_out * n_channels_out

        # Stage 1 - Vector learning and preparation
        if n_divs == 16:
            self.z0_shape = (None, 16 * n_channels, *frame_shape)
            self.static_learn = LearnVectorBlock(self.static_shape, n_channels, (3, 3), actArgs, bnArgs, block_i=1)
        else:
            self.stack_shape = (None, 8 + n_frame_in * n_channels, *frame_shape) if opts.use_time_slot else \
                (None, 7 + n_frame_in * n_channels, *frame_shape)
            self.z0_shape = self.stack_shape  # (None, n_multiples_out, *frame_shape)

        # Stage 2 Encoder
        self.n_stages = np.floor(np.log2(np.array(frame_shape)/2)).astype(int).max()
        for i in range(1, self.n_stages+1):
            ii = (i - 1) // sf_grp
            sf_i = sf * 2 ** ii  # (i-1)//2
            z_shape = self.z0_shape if i == 1 else eval(f'self.z_enc{i-1}.output_shape')
            enc = EncodeBlock(z_shape, sf_i, layer_i=i, nb_layers=nb_layers)
            setattr(self, f'z_enc{i}', enc)

        # code
        self.z_code = CreateConvBnLayer(enc.output_shape, sf_i * 2, layer_i=100)

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            ii = (i - 1) // sf_grp
            sf_i = sf * 2 ** ii  # (i-1)//2
            x_shape = self.z_code.output_shape if i == self.n_stages else eval(f'self.z_dec{i + 1}.output_shape')
            y_shape = eval(f'self.z_enc{i}.output_shape')
            dec = DecodeBlock(x_shape, y_shape, sf_i, layer_i=100+i)
            setattr(self, f'z_dec{i}', dec)
        self.z_dec0 = DecodeBlock(self.z_dec1.output_shape, self.z0_shape, n_multiples_out, layer_i=200)

        pad_type = convArgs['padding']
        convArgs_ = convArgs.copy()
        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        self.conv = ModelConv2D(in_channels=self.z_dec0.output_shape[CHANNEL_AXIS],
                                out_channels=n_multiples_out,  # n_divs = 16
                                kernel_size=(3, 3), stride=(1, 1), **convArgs_)
        self.classifier = Activation(**classifier_actArgs)

    def forward(self, x, x_static=None):  #x is [dynamic, static]
        if x_static != None:
            dynamic_x = x
            static_x = x_static
        else:
            dynamic_x, static_x = x

        if n_divs == 16:
            self.static_input = self.static_learn(static_x)
            s_device = self.static_input.device
            s_shape = list(self.static_input.shape)
            s_shape[CHANNEL_AXIS] *= 3
            z0 = torch.cat([self.static_input, dynamic_x, torch.zeros(*s_shape).to(s_device)], dim=1)
        else:
            z0 = torch.cat(x, dim=1)  # self.stacked_block(torch.cat(x, dim=1))

        for i in range(1, self.n_stages+1):
            exec(f'z{i} = self.z_enc{i}(z{i-1})')

        # code
        z100 = eval(f'self.z_code(z{self.n_stages})')

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            z_ = eval(f'z100 if i == self.n_stages else z10{i+1}')
            exec(f'z10{i} = self.z_dec{i}([z_, z{i}])')
        z200 = eval(f'self.z_dec0([z101, z0])')

        model_output = self.classifier(self.conv(z200))

        if n_divs == 16:
            model_output = torch.cat([get_c(model_output, idx//5, 16) for idx in times_out], dim=1)
        return model_output

    def name(self):
        return 'my_model'

    def compile(self):
        #, training_dataset, validation_dataset):  #compile(loss=criterion, optimizer=optimizer, metrics=metrics,
        # loss_weights=loss_weights)
        # self.optimizer = optimizer
        self.is_compiled = True
        self.other_args = other_args
        self.batch_size = batch_size

        # self.train_dataset = training_dataset
        # self.val_dataset = validation_dataset

        self.loss = nn.MSELoss()
        self.optimizer = optimizer_dict[opt_type](self.parameters(), **other_args)
        self.metrics = [MAE(), RMSE(), RMSLE(), NDEI(), R2_SCORE(), MAAPE()]


class SedanionModelScaled2(nn.Module):
    def __init__(self):  # , stack_input=True):
        super(SedanionModelScaled2, self).__init__()
        """" implement a model refresher here based on settings.options without model reloading """
        setup_model()

        self.net_type = opts.net_type
        self.is_compiled = False
        self.num_params = 0
        self.dynamic_shape = (None, n_frame_in * n_channels, *frame_shape)
        self.static_shape = (None, 8, *frame_shape) if opts.use_time_slot else (None, 7, *frame_shape)
        assert n_divs in [16, 1]  # only real or sedanion implemented
        if n_divs == 16:
            n_multiples_out = 16 * n_channels_out  # n_divs = 16 for sedanion
        else:
            n_multiples_out = n_frame_out * n_channels_out

        # Stage 1 - Vector learning and preparation
        if n_divs == 16:
            self.z0_shape = (None, 16 * n_channels, *frame_shape)
            self.static_learn = LearnVectorBlock(self.static_shape, n_channels, (3, 3), actArgs, bnArgs, block_i=1)
        else:
            self.stack_shape = (None, 8 + n_frame_in * n_channels, *frame_shape) if opts.use_time_slot else \
                (None, 7 + n_frame_in * n_channels, *frame_shape)
            self.z0_shape = self.stack_shape  # (None, n_multiples_out, *frame_shape)

        # Stage 2 Encoder
        self.n_stages = np.ceil(np.log2(np.array(frame_shape) / 2)).astype(int).max()
        sf_i = sf
        for i in range(1, self.n_stages + 1):
            z_shape = self.z0_shape if i == 1 else eval(f'self.z_enc{i-1}.output_shape')
            enc = EncodeBlock(z_shape, sf_i, layer_i=i, nb_layers=nb_layers)
            setattr(self, f'z_enc{i}', enc)
            if (sf_i + 32) <= 128 + 32 * (opts.sf_grp - 1):  # *2:
                sf_i += 32

        # code
        self.z_code = CreateConvBnLayer(enc.output_shape, sf_i, layer_i=100)

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            x_shape = self.z_code.output_shape if i == self.n_stages else eval(f'self.z_dec{i + 1}.output_shape')
            y_shape = eval(f'self.z_enc{i}.output_shape')
            dec = DecodeBlock(x_shape, y_shape, sf_i, layer_i=100+i)
            setattr(self, f'z_dec{i}', dec)

        pad_type = convArgs['padding']
        convArgs_ = convArgs.copy()
        convArgs_['padding'] = (1, 1) if pad_type is 'same' else (0, 0)
        self.conv = ModelConv2D(in_channels=self.z_dec1.output_shape[CHANNEL_AXIS],
                                out_channels=n_multiples_out,  # n_divs = 16
                                kernel_size=(3, 3), stride=(1, 1), **convArgs_)
        self.classifier = Activation(**classifier_actArgs)

    def forward(self, x, x_static=None):  #x is [dynamic, static]
        if x_static != None:
            dynamic_x = x
            static_x = x_static
        else:
            dynamic_x, static_x = x

        if n_divs == 16:
            # self.dynamic_input = self.dynamic_learn(dynamic_x)
            self.static_input = self.static_learn(static_x)
            s_device = self.static_input.device
            s_shape = list(self.static_input.shape)
            s_shape[CHANNEL_AXIS] *= 3
            z0 = torch.cat([self.static_input, dynamic_x, torch.zeros(*s_shape).to(s_device)], dim=1)
        else:
            z0 = torch.cat(x, dim=1)  #self.stacked_block(torch.cat(x, dim=1))

        for i in range(1, self.n_stages+1):
            exec(f'z{i} = self.z_enc{i}(z{i-1})')

        # code
        z100 = eval(f'self.z_code(z{self.n_stages})')

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            z_ = eval(f'z100 if i == self.n_stages else z200')
            z200 = eval(f'self.z_dec{i}([z_, z{i}])')

        model_output = self.classifier(self.conv(z200))

        if n_divs == 16:
            model_output = torch.cat([get_c(model_output, idx//5, 16) for idx in times_out], dim=1)
        return model_output

    def name(self):
        return 'my_model'

    def compile(self):
        self.is_compiled = True
        self.other_args = other_args
        self.batch_size = batch_size

        self.loss = nn.MSELoss()
        self.optimizer = optimizer_dict[opt_type](self.parameters(), **other_args)
        self.metrics = [MAE(), RMSE(), RMSLE(), NDEI(), R2_SCORE(), MAAPE()]


def model_trainer(model,
                  training_datagen,
                  validation_datagen,
                  criterion,
                  optimizer,
                  model_path,
                  epochs=1,
                  metrics=None,  # must be list
                  loss_weights=None,
                  verbose=1,
                  initial_epoch=0,
                  callbacks=None,
                  steps_per_epoch=None,
                  steps_per_validation=None,
                  use_cuda=True):
    epoch_str_width = len(str(epochs))
    if model.is_compiled:
        criterion = model.loss
        optimizer = model.optimizer
        metrics = model.metrics
        if use_cuda:
            model = model.cuda()
    else:
        if isinstance(criterion, str):
            criterion = criterion_dict[criterion]
        if isinstance(optimizer, str):
            optimizer = optimizer_dict[criterion](model.parameters())
        model.compile(loss=criterion, optimizer=optimizer, metrics=metrics, loss_weights=loss_weights)
        # model = nn.DataParallel(model, device_ids=[0])
        if use_cuda:
            model = model.cuda()
        # model.to(device)

    metrics = metrics if metrics else []
    callbacks = callbacks if callbacks else []
    for idx, metric in enumerate(metrics):
        if isinstance(metric, str):
            metrics[idx] = criterion_dict[metric]
            # print(metrics[idx])

    # metrics_name = [x.__class__.__name__.lower()[:-4] for x in metrics]
    metrics_name = [metric.name for metric in metrics]
    # metrics_dict = {x: 0 for x in metrics_name}

    logs = {'loss': 0}
    logs.update({x: 0 for x in metrics_name})
    train_dict = logs.copy()
    validation_dict = {f'val_{key}': 0 for key in logs}
    logs.update(validation_dict)
    logs.update({'time': 0})

    logs_df = pd.DataFrame(columns=logs.keys())

    train_steps = training_datagen.__len__()
    valid_steps = validation_datagen.__len__()
    steps_per_epoch = steps_per_epoch if steps_per_epoch else training_datagen.__len__()
    steps_per_validation = steps_per_validation if steps_per_validation else validation_datagen.__len__()

    best_loss = -np.inf
    for epoch in range(initial_epoch, epochs):
        # epoch_loss = 0
        # epoch_metrics = [0] * len(metrics)
        training_datagen.on_epoch_end()
        train_df = pd.DataFrame(columns=list(train_dict.keys()))
        validation_df = pd.DataFrame(columns=list(validation_dict.keys()))
        model.train()  # model.train(mode=True)
        start_time = time.time()
        for step in range(steps_per_epoch):
            # step_time = time.time()
            x, y = training_datagen.__getitem__(step % train_steps)
            if use_cuda:
                x = [x_.cuda(non_blocking=True) for x_ in x]
                y = y.cuda(non_blocking=True)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            with torch.no_grad():
                # for idx, metric in enumerate(metrics):
                #     # epoch_metrics[idx_metric] += metric(output, y)
                #     train_dict[metrics_name[idx]] = metric(output, y).item()
                for metric in metrics:
                    train_dict.update({metric.name: metric(output, y).item()})
            # epoch_loss += loss
            train_dict['loss'] = loss.item()
            train_df = train_df.append(train_dict, ignore_index=True)

            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time_so_far = (time.time() - start_time)
            step_time = time_so_far / (step + 1)
            if verbose >= 1:
                time_spent_str = time_to_str(time_so_far)
                time_str = time_to_str(step_time * (steps_per_epoch - step))
                other_str = ' - '.join([f"{key}: {value:0.4f}" for key, value in train_dict.items()])
                # print(f'{step + 1}/{steps_per_epoch} -- ETA: {time_str} - {other_str}', end='\r')
                print(f'Epoch [{epoch + 1}/{opts.epochs}] - Step [{step + 1}/{steps_per_epoch}] - ETA: '
                      f'[{time_spent_str}<{time_str}] - {other_str}', end='\r')

            # if step == steps_per_epoch - 1:
            #     train_dict = steps_df.mean(axis=0).to_dict()

        epoch_time = (time.time() - start_time)
        # epoch_loss /= steps_per_epoch
        # epoch_mae /= steps_per_epoch
        train_dict = train_df.mean(axis=0).to_dict()
        for key, value in train_dict.items():
            logs[key] = value
        logs['time'] = epoch_time
        if use_cuda:
            del x, y, output
            torch.cuda.empty_cache()
        model.eval()  # model.train(mode=False)
        for step in range(steps_per_validation):
            step_time = time.time()
            x, y = validation_datagen.__getitem__(step % valid_steps)
            if use_cuda:
                x = [x_.cuda(non_blocking=True) for x_ in x]
                y = y.cuda(non_blocking=True)

            with torch.no_grad():
                output = model(x)
                val_loss = criterion(output, y)
                # mae = metric_mae(output, y)
                # for idx, metric in enumerate(metrics):
                #     # epoch_metrics[idx_metric] += metric(output, y)
                #     validation_dict[f"val_{metrics_name[idx]}"] = metric(output, y).item()
                for metric in metrics:
                    validation_dict.update({f'val_{metric.name}': metric(output, y).item()})
            validation_dict['val_loss'] = val_loss.item()
            validation_df = validation_df.append(validation_dict, ignore_index=True)

        validation_dict = validation_df.mean(axis=0).to_dict()
        for key, value in validation_dict.items():
            logs[key] = value
        logs_df = logs_df.append(logs, ignore_index=True)
        if use_cuda:
            del x, y, output
            torch.cuda.empty_cache()
        # scheduler.step(epoch_val_loss)
        if not callbacks:  # is not None
            for callback in callbacks:
                callback.step(logs['val_loss'])

        """Saving the present best model"""
        present_best_loss = logs['val_loss']
        if present_best_loss < best_loss:
            # saving the best checkpoint
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}

            torch.save(checkpoint, model_path)
            print(f'The model improves from {best_loss:0.6f} to {present_best_loss:0.6f} and has been saved in'
                  f' {model_path}')
            best_loss = present_best_loss
        else:
            print(f'The model does not improve from {best_loss:0.6f}')
        other_str = ' - '.join([f"{key}: {value:0.4f}" for key, value in logs.items()])
        print(f'epoch {epoch + 1:0{epoch_str_width}d}/{epochs} -- {other_str}')

    return model, optimizer, logs_df


