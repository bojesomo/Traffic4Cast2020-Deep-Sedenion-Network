import numpy as np
import torch
import torch.nn as nn

# class Swish(nn.Module):
#     def __init__(self):
#         super(Swish, self).__init__()
#
#     def forward(self, inputs):
#         inputs = inputs * torch.sigmoid(inputs)
#         return inputs


class SwishAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        # sigmoid_i = torch.sigmoid(i)
        result = i * torch.sigmoid(i)
        # result = i * sigmoid_i
        ctx.save_for_backward(i)  # , sigmoid_i)
        del i
        torch.cuda.empty_cache()
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        # i, sigmoid_i = ctx.saved_variables
        sigmoid_i = torch.sigmoid(i)
        out = grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
        del sigmoid_i, i
        torch.cuda.empty_cache()
        return out


class Swish(nn.Module):
    def forward(self, input_tensor):
        return SwishAutoGrad.apply(input_tensor)




class R2Loss(object):
    def __init__(self):
        self.name = 'r2'

    def __call__(self, y_pred, y_target):
        sse = nn.MSELoss()(y_pred, y_target)
        sst = nn.MSELoss()(y_target, torch.zeros_like(y_pred))
        return 1 - torch.div(sse, sst)


class MSE(object):
    def __init__(self):
        self.name = 'mse'

    def __call__(self, y_pred, y_target):
        sse = nn.MSELoss()(y_pred, y_target)
        return sse


class RMSE(object):
    def __init__(self):
        self.name = 'rmse'

    def __call__(self, y_pred, y_target):
        sse = nn.MSELoss()(y_pred, y_target)
        return torch.sqrt(sse)


def ndei(pred, target):
    """nondiemensional error index"""
    # sse = torch.mean(torch.pow(pred - target, 2.0))
    # sst = torch.mean(torch.pow(target, 2.0))
    sse = nn.MSELoss()(pred, target)
    sst = nn.MSELoss()(target, torch.zeros_like(pred))
    return torch.div(sse, sst)


class R2_SCORE(object):
    def __init__(self):
        self.name = 'r2'

    def __call__(self, pred, target):
        return 1 - ndei(pred, target)


class NDEI(object):
    """nondiemensional error index"""
    def __init__(self):
        self.name = 'ndei'

    def __call__(self, pred, target):
        return ndei(pred, target)


class MAE(object):
    """mean absolute error"""
    def __init__(self):
        self.name = 'mae'

    def __call__(self, pred, target):
        return nn.L1Loss()(pred, target)


class MAAPE(object):
    """mean arctangent absolute percentage error"""
    def __init__(self):
        self.name = 'maape'

    def __call__(self, pred, target):
        return torch.atan(torch.div(pred - target, target).abs()).mean()
        # return maape


class RMSLE(object):
    def __init__(self):
        self.name = 'rmsle'

    def __call__(self, y_pred, y_target):
        return RMSE()(torch.log(y_pred+1), torch.log(y_target+1))


def time_to_str(t_seconds):
    """Express time in string"""
    t_seconds = int(round(t_seconds))
    if t_seconds // (365*24*60*60):
        f_str = 'year:day:hr:mins:secs'
    elif t_seconds // (24*60*60):
        f_str = 'day:hr:mins:secs'
    else:
        f_str = 'hr:mins:secs'

    value_ = {'year': t_seconds // (60*60*24*365),
              'day': (t_seconds // (60*60*24)) % 365,
              'hr': (t_seconds // (60*60)) % 24,
              'mins': (t_seconds // 60) % 60,
              'secs': t_seconds % 60,
              }
    format_ = {'year': 1,
               'day': 3,
               'hr': 2,
               'mins': 2,
               'secs': 2,
              }

    f_split = f_str.split(':')
    result = ':'.join([f"{value_[x]:0{format_[x]}d}" for x in f_split])
    return result


def compute_padding(size_out, size_in, kernel_size, stride, padding='same'):
    pad_size = 0
    if padding == 'same':
        pad_size = (stride * (size_out - 1) + kernel_size - size_in) / 2
    return int(np.ceil(pad_size))


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

