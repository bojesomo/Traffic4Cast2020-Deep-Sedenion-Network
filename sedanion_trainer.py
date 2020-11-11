# coding:utf-8
##########################################################
# pytorch v1.0
# Alabi Bojesomo
# Khalifa University
# Abu Dhabi, UAE
# August 2020
##########################################################

import time
import settings
from torchvision import transforms
import os

from utils import time_to_str, Swish, R2_SCORE, MAAPE, NDEI, RMSE, RMSLE, MSE, MAE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_model_summary import summary

from torch.optim.lr_scheduler import ReduceLROnPlateau

from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
torch.autograd.set_detect_anomaly(True)

from sedanion_loader import DataGenerator, ToTensor
from models_sedanion import (SedanionModel, SedanionModelScaled, SedanionModelScaled2,
                                                      optimizer_dict, criterion_dict)
import tempfile
from torch.nn.parallel import DistributedDataParallel as DDP

# opts = settings.options


parser = argparse.ArgumentParser()
parser.add_argument('--stack_input', type=bool, default=True, help='is the input to be stacked [True, False]')
parser.add_argument('--use_group_norm', type=bool, default=False, help='is the input to be stacked [True, False]')
parser.add_argument('--use_time_slot', type=bool, default=False, help='use the time slots encoding')
parser.add_argument('--net_type', default='sedanion', help='type of network', choices=['sedanion', 'real'])
parser.add_argument('--blk_type', default='resnet', help='type of block', choices=['resnet', 'densenet'])
parser.add_argument('--nb_layers', type=int, default=1, help='depth of resnet/densenet blocks')  # 3
# working on sf divisible by 16
parser.add_argument('--sf', type=int, default=32, help='number of feature maps')  # 16*4
parser.add_argument('--sf_grp', type=int, default=2, help='number of feature groups before expansion')  # 2
parser.add_argument('--hidden_activation', default='relu', help='hidden layer activation')  # need to try swish
parser.add_argument('--classifier_activation', default='hardtanh', help='classifier layer activation',
                    choices=['sigmoid', 'hardtanh', 'tanh'])
parser.add_argument('--modify_activation', type=bool, default=True, help='modify the range of hardtanh activation')
parser.add_argument('--inplace_activation', type=bool, default=True, help='inplace activation')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=6, help='batch size')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
parser.add_argument('--normalize_data', type=bool, default=True, help='normalize the data or not')

parser.add_argument('--data_root', default='D:/KU_Works/Datasets/Traffic4cast_2019/2019/',
                    help='root directory for data')
parser.add_argument('--city', default='all', help='city data to train with or all ')

parser.add_argument('--seed', default=7, type=int, help='manual global seed')

parser.add_argument('--image_width', type=int, default=436, help='the height / width of the input image to network')
parser.add_argument('--image_height', type=int, default=495, help='the height / width of the input image to network')
parser.add_argument('--scale_height', type=int, default=1, help='factor to scale down image height / width')
parser.add_argument('--scale_width', type=int, default=1, help='factor to scale down image height / width')
parser.add_argument('--scale_type', default='pool', help='type of scaling to consider', choices=['pool', 'crop'])
parser.add_argument('--dataset', default='traffic', help='dataset to train with')

parser.add_argument('--n_frame_in', type=int, default=12, help='number of incoming frames')
parser.add_argument('--n_frame_out', type=int, default=6, help='number of frames to predict')
parser.add_argument('--times_out', type=str, default='5:10:15:30:45:60', help='actual timing out in minutes sep by :')
parser.add_argument('--n_channels', type=int, default=9, help='total number of channels')
parser.add_argument('--n_channels_out', type=int, default=8, help='number of channels to predict')

parser.add_argument('--optimizer', default='adam', help='optimizer to train with [sgd, adam]')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum term for sgd')
parser.add_argument('--beta_1', default=0.9, type=float, help='beta_1 term for adam')
parser.add_argument('--beta_2', default=0.999, type=float, help='beta_2 term for adam')
parser.add_argument('--epsilon', default=1e-8, type=float, help='epsilon term for adam')
parser.add_argument('--weight_decay', default=1e-8, type=float, help='weight decay for regularization')

parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', 
                    help='identifier for model if already exist')
parser.add_argument('--initial_epoch', type=int, default=0, help='number of epochs done')
parser.add_argument('--model_part', type=bool, default=False, help='use part based or not')
parser.add_argument('--model_num', type=int, default=1, choices=[1, 2, 3, 4])

opts = parser.parse_args()

net_dict = {'seda': 'sedanion', 'octo': 'octonion', 'quat': 'quaternion', 'comp': 'complex', 'real': 'real'}
net_type = net_dict[opts.net_type.lower()[:4]]

city_name = opts.city.upper()
seed = opts.seed
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

n_frame_in = opts.n_frame_in
n_frame_out = opts.n_frame_out
times_out = [int(t) for t in opts.times_out.split(':')]
n_channels = opts.n_channels
n_channels_out = opts.n_channels_out
h, w = opts.image_height, opts.image_width

sh, sw = opts.scale_height, opts.scale_width
h //= sh
w //= sw
scale = (sh, sw)

frame_shape = (h, w)
static_shape = (h, w, 7)

epochs = opts.epochs
batch_size = opts.batch_size

opt_type = opts.optimizer
lr = opts.lr
momentum = opts.momentum
beta_1 = opts.beta_1
beta_2 = opts.beta_2
epsilon = opts.epsilon

nb_layers = opts.nb_layers
data_root = opts.data_root
sf = opts.sf

stack_input = opts.stack_input
use_group_norm = opts.use_group_norm

# Save model and weights path
save_dir = os.path.join(os.getcwd(), 'saved_models')
os.makedirs(save_dir, exist_ok=True)
time_code = opts.name[opts.name.index('2020'):-3] if opts.name else '_'.join(np.array(time.localtime(), dtype=str)[:-3])
model_name = opts.name if opts.name else f'{city_name}_{net_type}_s{stack_input}_g{use_group_norm}_{time_code}.pt'
# model_name = f'{city_name}_{net_type}_s{stack_input}_g{use_group_norm}_{time_code}.pt'
model_path = os.path.join(save_dir, 'best', model_name)
ckpt_path = os.path.join(save_dir, 'ckpt', model_name)

# Load data
training_dir = [data_root, 'training'] if city_name == 'ALL' else os.path.join(data_root, city_name, 'training')
testing_dir = [data_root, 'testing'] if city_name == 'ALL' else os.path.join(data_root, city_name, 'testing')
validation_dir = [data_root, 'validation'] if city_name == 'ALL' else os.path.join(data_root, city_name, 'validation')

shuffle = True if city_name == 'ALL' else False
start_time = time.time()
training_datagen = DataGenerator(data_dir=training_dir, batch_size=batch_size, n_channels=n_channels,
                                 n_frame_in=n_frame_in,  # n_frame_out=n_frame_out,
                                 n_out_channel=n_channels_out, times_out=times_out, shuffle=shuffle,
                                 scale=scale, scale_type=opts.scale_type, use_time_slot=opts.use_time_slot,
                                 model_part=opts.model_part, model_num=opts.model_num,
                                 transform=transforms.Compose([ToTensor()]))
training_datagen_time = time.time() - start_time
print(f'training datagen loaded in {training_datagen_time} seconds')

start_time = time.time()
validation_datagen = DataGenerator(data_dir=validation_dir, batch_size=batch_size,  # np.max([2, batch_size // np.prod(scale)]),
                                   n_channels=n_channels,
                                   n_frame_in=n_frame_in,  # n_frame_out=n_frame_out,
                                   n_out_channel=n_channels_out, times_out=times_out, scale=scale,  # (1, 1)
                                   scale_type=opts.scale_type, use_time_slot=opts.use_time_slot,
                                   model_part=opts.model_part, model_num=opts.model_num,
                                   transform=transforms.Compose([ToTensor()]))
print(f'validation datagen loaded in {time.time() - start_time} seconds')


# GET MODEL
start_time = time.time()
settings.init(opts)  # add opts to global variables {required before calling FullModel
model = SedanionModelScaled() 
model_time = time.time() - start_time
if opts.use_time_slot:
    x_in = (torch.rand(1, 108, 495, 436), torch.rand(1, 8, 495, 436))
else:
    x_in = (torch.rand(1, 108, 495, 436), torch.rand(1, 7, 495, 436))
_ = summary(model, x_in, print_summary=True)
del x_in
print(f'model created in {model_time} seconds')

opt_type = opts.optimizer
lr = opts.lr
momentum = opts.momentum
beta_1 = opts.beta_1
beta_2 = opts.beta_2
epsilon = opts.epsilon

criterion = criterion_dict['mse']  # nn.MSELoss()
metric_mae = criterion_dict['mae']  # nn.L1Loss()

other_args = {}
if opt_type == 'sgd':
    other_args = {'lr': lr, 'momentum': momentum, 'weight_decay': opts.weight_decay, 'nesterov': True}
elif opt_type == 'adam':
    other_args = {'lr': lr, 'eps': epsilon, 'betas': (beta_1, beta_2), 'weight_decay': 1e-4}

optimizer = optimizer_dict[opt_type](model.parameters(), **other_args)
# if 'optim_dict' in locals():
#     optimizer.load_state_dict(optim_dict)
#     del optim_dict
#     torch.cuda.empty_cache()

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, cooldown=0, min_lr=1e-7)

CUDA = torch.cuda.is_available()
# start_time = time.time()
# model.compile(loss=criterion, optimizer=optimizer)  # , metrics=['acc', 'mae'])
# print(f'model compiles in {time.time() - start_time} seconds')
print(opts)

verbose = 1
initial_epoch = opts.initial_epoch  # 0
steps_per_epoch = None
steps_per_validation = None

parser1 = argparse.ArgumentParser()
args_in = parser1.parse_args()

args_in.model = model
args_in.training_datagen = training_datagen
args_in.validation_datagen = validation_datagen
args_in.criterion = criterion
args_in.optimizer = optimizer
args_in.epochs = epochs
args_in.metrics = [MAE(), RMSE(), RMSLE(), MAAPE(), R2_SCORE()]  # NDEI(), MAAPE()]
args_in.verbose = verbose
args_in.initial_epoch = initial_epoch
args_in.callbacks = [scheduler]
args_in.steps_per_epoch = steps_per_epoch
args_in.steps_per_validation = steps_per_validation

args_in.save_dir = save_dir
args_in.model_name = model_name
args_in.model_path = model_path
args_in.ckpt_path = ckpt_path
args_in.time_code = time_code


def trainer(args_in):
    parser2 = argparse.ArgumentParser()
    parser2.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # parser2.add_argument('-g', '--gpus', default=1, type=int,
    #                     help='number of gpus per node')
    parser2.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser2.parse_args()
    args.args_in = args_in
    # args.world_size = args.gpus * args.nodes
    args.directory = args_in.save_dir  # os.path.dirname(os.path.abspath(__file__))  # os.getcwd()
    args.logs_dir = os.path.join(args.directory, 'logs')
    args.best_dir = os.path.join(args.directory, 'best')
    args.ckpt_dir = os.path.join(args.directory, 'ckpt')
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.best_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    time_1 = datetime.now()
    # print(opts.epochs)
    # mp.spawn(train_ddp, nprocs=args.gpus, args=(args,))
    train(args)
    print(f"Back from trainer in {str(datetime.now() - time_1)}")


def train(args):
    logs_temp_file = os.path.join(args.logs_dir, '_'.join(['steps_log', args.args_in.time_code])+'.csv')
    epochs_temp_file = os.path.join(args.logs_dir, '_'.join(['epochs_log', args.args_in.time_code])+'.csv')
    CHECKPOINT_PATH = os.path.join(args.logs_dir, args.args_in.model_name)  # 'checkpoint.pt')
    # rank = args.nr * args.gpus + gpu
    # print(rank)
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    # torch.manual_seed(0)
    # print(f'initialied gpu {gpu}')
    model = args.args_in.model
    num_params = model.num_params
    # torch.cuda.set_device(gpu)
    # model.cuda(gpu)
    model = model.cuda()
    batch_size = opts.batch_size  # 100
    # criterion = args.args_in.criterion.cuda(gpu)
    criterion = args.args_in.criterion.cuda()
    optimizer = args.args_in.optimizer  # torch.optim.SGD(model.parameters(), 1e-4)
    # # Wrap the model
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    # load model if already exist
    if (os.path.exists(args.args_in.model_path) or os.path.exists(args.args_in.ckpt_path)):
        '''load the existing model accurately'''
        print('load the existing model accurately')
        if os.path.exists(args.args_in.model_path):
            ckpt = torch.load(args.args_in.model_path)
        else:
            ckpt = torch.load(args.args_in.ckpt_path)
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            optim_dict = ckpt['optimizer']

            model.load_state_dict(state_dict)
            optimizer.load_state_dict(optim_dict)
            del ckpt, state_dict, optim_dict
            torch.cuda.empty_cache()
        else:
            state_dict = ckpt
            model.load_state_dict(state_dict)
            del ckpt, state_dict
            torch.cuda.empty_cache()
    else:
        print('Model does not exist in directory')

    start = datetime.now()
    
    verbose = args.args_in.verbose

    training_datagen = args.args_in.training_datagen
    validation_datagen = args.args_in.validation_datagen

    steps_per_epoch = args.args_in.steps_per_epoch if args.args_in.steps_per_epoch else training_datagen.__len__()
    steps_per_validation = args.args_in.steps_per_validation if args.args_in.steps_per_validation else validation_datagen.__len__()

    # gpus = args.gpus  # torch.cuda.device_count()
    # steps_per_epoch //= gpus
    # steps_per_validation //= gpus

    callbacks = args.args_in.callbacks
    metrics = args.args_in.metrics
    # metrics_name = [x.__class__.__name__.lower()[:-4] for x in metrics]
    metrics_name = [metric.name for metric in metrics]

    logs = {'loss': 0}
    logs.update({x: 0 for x in metrics_name})
    train_dict = logs.copy()
    validation_dict = {f'val_{key}': 0 for key in logs}
    logs.update(validation_dict)
    logs.update({'time': 0, 'lr': 0, 'epoch': 0})

    logs_df = pd.DataFrame(columns=logs.keys())

    epoch_str_width = len(str(opts.epochs))
    best_loss = np.inf
    # group = dist.new_group([rank_i for rank_i in range(args.world_size)])
    # print(group)
    for epoch in range(args.args_in.initial_epoch, opts.epochs):
        training_datagen.on_epoch_end()
        logs['lr'] = optimizer.param_groups[0]['lr']
        train_df = pd.DataFrame(columns=list(train_dict.keys()))
        validation_df = pd.DataFrame(columns=list(validation_dict.keys()))
        model.train()  # model.train(mode=True)
        start_time = time.time()

        """###### Training  ######## """
        for step in range(steps_per_epoch):
            # x, y = training_datagen.__getitem__(gpus * step + gpu)
            x, y = training_datagen.__getitem__(step)
            if not opts.normalize_data:
                x = [x_*255. for x_ in x]
                y *= 255.
            x = [x_.cuda(non_blocking=True) for x_ in x]
            y = y.cuda(non_blocking=True)

            # Forward pass
            optimizer.zero_grad()
            output = model(x)
            if not opts.normalize_data:
                output = output * 255.
            loss = criterion(output, y)

            # Backward and optimize
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for metric in metrics:
                    temp = metric(output, y)
                    train_dict.update({metric.name: temp.item()})
                del x, y, output, temp
                torch.cuda.empty_cache()
                # for idx, metric in enumerate(metrics):
                #     train_dict[metrics_name[idx]] = metric(output, y).item()
            train_dict['loss'] = loss.item()
            train_df = train_df.append(train_dict, ignore_index=True)

            time_so_far = (time.time() - start_time)
            step_time = time_so_far / (step + 1)
            if verbose >= 1:
                time_spent_str = time_to_str(time_so_far)
                time_str = time_to_str(step_time * (steps_per_epoch - step))
                other_str = ' - '.join([f"{key}: {value:0.5f}" for key, value in train_dict.items()])
                print(f'Epoch [{epoch+1}/{opts.epochs}] - Step [{step + 1}/{steps_per_epoch}] - ETA: '
                      f'[{time_spent_str}<{time_str}] - {other_str}', end='\r')

            logs_temp = {'Epoch': epoch + 1, 'Step': step + 1}
            logs_temp.update(train_dict)
            logs_temp['city'] = opts.city.lower()
            logs_temp_df = pd.DataFrame([logs_temp])
            if os.path.exists(logs_temp_file):
                logs_temp_df.to_csv(logs_temp_file, mode='a', index=False, header=False)
            else:
                logs_temp_df.to_csv(logs_temp_file, mode='a', index=False)
        # del x, y, output, temp
        # torch.cuda.empty_cache()

        epoch_time = (time.time() - start_time)
        train_dict = train_df.mean(axis=0).to_dict()
        for key, value in train_dict.items():
            logs[key] = value
        logs['time'] = epoch_time
        logs['epoch'] = epoch + 1

        """##### Validation #####"""
        model.eval()  # model.train(mode=False)
        val_start_time = time.time()
        for step in range(steps_per_validation):
            step_time = time.time()
            x, y = validation_datagen.__getitem__(gpus * step + gpu)
            if not opts.normalize_data:
                x = [x_*255. for x_ in x]
                y *= 255.
            x = [x_.cuda(non_blocking=True) for x_ in x]
            y = y.cuda(non_blocking=True)

            with torch.no_grad():
                output = model(torch.cat(x, dim=1)) if isinstance(args.args_in.model, UNet_3Plus) else model(x)
                if not opts.normalize_data:
                    output = output * 255.
                val_loss = criterion(output, y)
                for metric in metrics:
                    temp = metric(output, y)
                    dist.all_reduce(temp, op=dist.ReduceOp.SUM,  group=group)
                    validation_dict.update({f'val_{metric.name}': temp.item()/args.world_size})
                # for idx, metric in enumerate(metrics):
                #     validation_dict[f"val_{metrics_name[idx]}"] = metric(output, y).item()
                del x, y, output, temp
                torch.cuda.empty_cache()
            validation_dict['val_loss'] = val_loss.item()
            validation_df = validation_df.append(validation_dict, ignore_index=True)

        # del x, y, output, temp, val_loss
        # torch.cuda.empty_cache()

        validation_dict = validation_df.mean(axis=0).to_dict()
        for key, value in validation_dict.items():
            logs[key] = value
        logs['val_time'] = (time.time() - val_start_time)
        logs['city'] = opts.city.lower()
        logs_df = logs_df.append(logs, ignore_index=True)

        # scheduler.step(epoch_val_loss)
        if not callbacks:  # is not None
            for callback in callbacks:
                callback.step(logs['val_loss'])

        other_str = ' - '.join([f"{key}: {value:0.6f}" for key, value in logs.items() if not isinstance(value, str)])
        print(f'epoch {epoch + 1:0{epoch_str_width}d}/{epochs} -- {other_str}')

        """Updating the epoch log state"""
        epochs_temp_df = pd.DataFrame([logs])
        if os.path.exists(epochs_temp_file):
            epochs_temp_df.to_csv(epochs_temp_file, mode='a', index=False, header=False)
        else:
            epochs_temp_df.to_csv(epochs_temp_file, mode='a', index=False)

        """Saving the ckpts"""
        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, args.args_in.ckpt_path.replace('.', f'_ckpt{epoch + 1}.'))

        """Saving the present best model"""
        present_best_loss = logs['val_mse'] if 'val_mse' in logs else logs['val_loss']
        if present_best_loss < best_loss:
            # saving the best checkpoint
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}

            torch.save(checkpoint, args.args_in.model_path)
            # torch.save(checkpoint, os.paths.join(args.best, args.args_in.model_name + '.pt')) #also accepted
            print(f'The model improves from {best_loss:0.6f} to {present_best_loss:0.6f} and has been saved in'
                  f' {args.args_in.model_path}')
            best_loss = present_best_loss
        else:
            print(f'The model does not improve from {best_loss:0.6f}')

    # Save the logs
    if os.path.exists(args.args_in.model_path[:-3] + '.csv'):
        logs_df.to_csv(args.args_in.model_path[:-3] + '.csv', mode='a', index=False, header=False)
    else:
        logs_df.to_csv(args.args_in.model_path[:-3] + '.csv', mode='a', index=False)
    # Save the last state of the model
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, CHECKPOINT_PATH)
    # save model parameters used
    readme_file = os.path.join(save_dir, 'SedanionScaledReadMe.csv')
    opts_dict = vars(argparse.Namespace(**{'filename': args.args_in.model_name[:-3], 'num_params': num_params,
                                           'val_mse': best_loss}, **vars(opts)))
    # opts_dict = vars(argparse.Namespace(**{'filename': model_name[:-3], 'num_params': num_params,
    #                                        'val_loss': best_loss}, **vars(opts)))
    opts_df = pd.DataFrame([opts_dict])
    if os.path.exists(readme_file):
        opts_df.to_csv(readme_file, mode='a', index=False, header=False)
    else:
        opts_df.to_csv(readme_file, mode='a', index=False)
    print("Training complete in: " + str(datetime.now() - start))



if __name__ == '__main__':
    trainer(args_in)
