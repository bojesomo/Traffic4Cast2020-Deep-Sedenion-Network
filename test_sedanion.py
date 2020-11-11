# coding:utf-8
##########################################################
# pytorch v1.0
# Alabi Bojesomo
# Khalifa University
# Abu Dhabi, UAE
# September 2020
##########################################################

import settings
from sedanion_loader import DataGenerator, ToCuda, ToTensor
from utils import time_to_str
from models_sedanion import SedanionModel, SedanionModelScaled, SedanionModelScaled2
import torch
import time
import os
import json
import pandas as pd
import numpy as np
import argparse
from matplotlib import pyplot as plt
from torchvision import transforms

use_cuda = torch.cuda.is_available()  #True
transform = transforms.Compose([ToTensor()])
all_cities = ['BERLIN', 'ISTANBUL', 'MOSCOW']

data_root = 'D:/KU_Works/Datasets/Traffic4cast_2019/2019/'
times_out = [int(t) for t in '5:10:15:30:45:60'.split(':')]

testing_dir = [data_root, 'testing']
testing_datagen = DataGenerator(data_dir=testing_dir, batch_size=1, n_channels=9,
                                n_frame_in=12,  transform=transform,
                                n_out_channel=8, times_out=times_out)

saved_result_dir = 'test_result'
os.makedirs(saved_result_dir, exist_ok=True)
steps_per_testing = len(testing_datagen.indexes_test)

with open('test_slots.json') as json_file:
    test_list = json.load(json_file)
    test_slots = {}
    for test_point in test_list:
        test_slots.update(test_point)

fname = os.path.join('saved_models', 'SedanionReadMe.csv')
checkpoint_dir = os.path.join('saved_models', 'best')

df = pd.read_csv(fname)

mask = df.index == df.index[-1]
df_used = df[mask]
# df_used = df
df_used.index = range(len(df_used.index))
n_models = df_used.__len__()

'''trained models'''
for itr, idx in enumerate(df_used.index):
    temp = df_used.loc[idx].to_dict()
    model_name = temp['filename']
    # print(f'Working on {model_name}')
    saved_dir = os.path.join(saved_result_dir, model_name)
    os.makedirs(saved_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}.pt')
    model_city = model_name.split('_')[0]
    opts = argparse.Namespace(**temp)

    # if not('scale_height' in opts):
    #     opts.scale_height = 1
    # if not ('scale_width' in opts):
    #     opts.scale_width = 1
    opts.scale_height = 1
    opts.scale_width = 1

    settings.init(opts)  # add opts to global variables

    model = SedanionModelScaled()  # get the model
    
    '''load the checkpoint accurately'''
    ckpt = torch.load(checkpoint_path)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    old_keys = list(state_dict.keys())
    for old_key in old_keys:
        new_key = old_key.split('module.')[-1]
        state_dict[new_key] = state_dict.pop(old_key)
    model.load_state_dict(state_dict)
    '''get set for evaluation mode'''
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    if use_cuda:
        model.cuda()  # move model to GPU

    start_time = time.time()
    for step in range(steps_per_testing):
        
        file_index, _, city_index = testing_datagen.indexes_test[step]
        city_name = testing_datagen.cities[city_index]
        filename = testing_datagen.files_ID[city_index][file_index]

        saved_city_dir = os.path.join(saved_dir, city_name)
        os.makedirs(saved_city_dir, exist_ok=True)
        file_path = os.path.join(saved_city_dir, filename)

        if os.path.exists(file_path):
            continue

        x_test = testing_datagen.__gettest__(step)
        # x_test = [x_.cuda() for x_ in x_test]
        if opts.use_time_slot:
            x_time = (torch.tensor(test_slots[filename.split('_')[0]], device=x_test[1].device) * torch.ones_like(
                x_test[1][:, :1, ...]).transpose(0, 3)).transpose(0, 3) / 288.
            x_test[1] = torch.cat([x_test[1], x_time], dim=1)

        if not opts.normalize_data:
            x_test = [x_ * 255. for x_ in x_test]

        # y_pred = model(x_test)
        y_pred = []
        for i in range(x_test[0].shape[0]):
            if use_cuda:
                x_now = [x_[i:i+1].cuda() for x_ in x_test]
            else:
                x_now = [x_[i:i + 1] for x_ in x_test]
            y_pred.append(model(x_now))
        y_pred = torch.cat(y_pred, dim=0)

        # model.cpu()  # move model to CPU
        # x_test = [x_.cpu() for x_ in x_test]
        y_out = testing_datagen.process_output(y_pred)
        testing_datagen.write_data(y_out, file_path)

        del x_test, y_pred
        # assert len(test_slots[filename.split('_')[0]]) == batch_size

        time_so_far = (time.time() - start_time)
        step_time = time_so_far / (step + 1)
        time_spent_str = time_to_str(time_so_far)
        time_str = time_to_str(step_time * (steps_per_testing - step))
        print(f'[{itr+1}/{n_models}]: {model_name} : ETA [{time_spent_str}<{time_str}]: done - '
              f'[{step + 1}/{steps_per_testing}]', end='\r')
