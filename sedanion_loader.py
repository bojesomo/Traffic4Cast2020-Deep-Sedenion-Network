import numpy as np
import os
import h5py
from os.path import dirname, basename
import torch
from torch.utils.data import Dataset  # , DataLoader
from torchvision import transforms


def pool(data):
    if g_pool:
        data = torch.nn.AvgPool2d(kernel_size=g_scale, stride=g_scale)(data)
    return data

class ToCuda(object):
    """ put on cuda """
    def __call__(self, sample):
        if isinstance(sample, tuple):
            X, y = sample
            X = [x.cuda() for x in X]
            y = y.cuda()
            return X, y
        else:  # test case {only X list}
            X = sample
            X = [x.cuda() for x in X]
            return X


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors """
    def __call__(self, sample):
        if isinstance(sample, tuple):
            X, y = sample
            # swap channel in image data
            # numpy/keras(chanel_last): H x W x C
            # torch                   : C x H x W
            X = [torch.from_numpy(x.transpose(0, 3, 1, 2)) for x in X]
            y = torch.from_numpy(y.transpose(0, 3, 1, 2))
            return X, y
        else:  # test case {only X list}
            X = sample
            # swap channel in image data
            # numpy/keras(chanel_last): H x W x C
            # torch                   : C x H x W
            X = [torch.from_numpy(x.transpose(0, 3, 1, 2)) for x in X]
            return X


class ToNumpy(object):
    """ Convert ndarrays in sample to Tensors """
    def __call__(self, sample):
        if isinstance(sample, tuple):
            X, y = sample
            # swap channel in image data
            # numpy/keras(chanel_last): H x W x C
            # torch                   : C x H x W
            X = [torch.from_numpy(x.transpose(0, 3, 1, 2)) for x in X]
            y = torch.from_numpy(y.transpose(0, 3, 1, 2))
            return X, y
        else:  # test case {only X list}
            X = sample
            # swap channel in image data
            # numpy/keras(chanel_last): H x W x C
            # torch                   : C x H x W
            X = [torch.from_numpy(x.transpose(0, 3, 1, 2)) for x in X]
            return X


class DataGenerator(Dataset):
    """Generates data for PyTorch"""

    def __init__(self, data_dir, batch_size=32, dim=(495, 436), n_channels=3, n_out_channel=None,
                 n_partitions=288, n_frame_in=12, times_out=None, transform=transforms.Compose([ToTensor(), ToCuda()]),
                 shuffle=False, scale=None, scale_type='crop', use_time_slot=False,
                 model_part=False, model_num=1):
        'Initialization'
        self.use_time_slot = use_time_slot
        self.mod_num = 1
        self.mod_size = [256, 224]
        mod_start = [(0, 0), (0, -224), (-256, 0), (-256, -224)]
        mod_end = [(256, 224), (256, 436), (495, 224), (495, 436)]
        if model_part:
            self.mod_start = mod_start[model_num - 1]
            self.mod_end = mod_end[model_num - 1]
        self.model_part = model_part
        assert scale_type in ['pool', 'crop']
        self.scale_type = scale_type
        self.dim = dim
        self.scale = scale if scale else (1, 1)
        self.height, self.width = self.dim[0] // self.scale[0], self.dim[1] // self.scale[1]
        self.hs = self.dim[0] - self.height + 1
        self.ws = self.dim[1] - self.width + 1
        self.do_pool = np.prod(self.scale) > 1  # only do pooling when we need to scale data
        self.do_crop = self.do_pool
        global g_scale, g_pool
        g_scale, g_pool = self.scale, self.do_pool
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.n_partitions = n_partitions
        self.n_frame_in = n_frame_in
        self.times_out = times_out if times_out else [5, 10, 15, 30, 45, 60]  # in mins
        self.n_frame_out_last = self.times_out[-1] // 5
        self.parts_per_file = self.n_partitions - (self.n_frame_in + self.n_frame_out_last) + 1
        assert self.batch_size <= self.parts_per_file
        self.n_channels = n_channels
        self.n_out_channel = n_out_channel

        if isinstance(data_dir, str):  # with specific city provided
            self.data_dir = [data_dir]  # data directory
        elif isinstance(data_dir, list):  # using all cities
            self.data_dir = [os.path.join(data_dir[0], x, data_dir[1]) for x in os.listdir(data_dir[0])]  # directories

        self.cities = [basename(dirname(data_dir_i)) for data_dir_i in self.data_dir]

        self.n_cities = len(self.data_dir)
        self.files_ID = [[this_file for this_file in os.listdir(x) if this_file.endswith('.h5')] for x in
                         self.data_dir]
        self.files_hw = [[(np.random.randint(self.hs), np.random.randint(self.ws)) for _ in file_ids] for
                         file_ids in self.files_ID]

        self.n_files = len(self.files_ID[0])
        self.file_num = np.arange(self.n_files)
        self.part_num = np.arange(self.parts_per_file)
        self.city_num = np.repeat(range(self.n_cities), self.n_files)
        self.file_frame = [[(self.file_num[itr % self.n_files], x, y) for x in self.part_num] for itr, y in
                           enumerate(self.city_num)]
        self.indexes = [xx for sublist in self.file_frame for xx in sublist]
        self.start_hw = [(np.random.randint(self.hs), np.random.randint(self.ws)) for _ in self.indexes]
        self.list_start_hw = None

        self.file_frame_test = [[(self.file_num[itr % self.n_files], x, y) for x in range(1)] for itr, y in
                                enumerate(self.city_num)]
        self.indexes_test = [xx for sublist in self.file_frame_test for xx in sublist]

        self.city_index = 0
        self.file_index = 0
        self.batch_end = 0
        self.data = (self.get_data(os.path.join(self.data_dir[self.city_index],
                                                self.files_ID[self.city_index][self.file_index])) /
                     255.).astype(np.float32)
        self.static_dir = [dirname(x) for x in self.data_dir]
        self.static_filename = [[x for x in os.listdir(y) if x.endswith('.h5')][0] for y in self.static_dir]
        self.static_data = [(self.get_data(os.path.join(self.static_dir[city_n], self.static_filename[city_n])) /
                             255.).astype(np.float32) for city_n in np.arange(self.n_cities)]
        self.length = self.__len__()
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        if torch.is_tensor(index):
            index = index.tolist()

        self.batch_end = (index + 1) * self.batch_size
        list_indexes = self.indexes[index * self.batch_size:self.batch_end]
        self.list_start_hw = self.start_hw[index * self.batch_size:self.batch_end]

        # Generate data
        # X, y = self.__data_generation(list_indexes)
        sample = self.__data_generation(list_indexes)
        if self.transform:
            # X = self.transform(X)
            # y = self.transform(y)
            sample = self.transform(sample)
        return sample  # X, y

    def __gettest__(self, index):
        """Generate one batch of test data"""
        if torch.is_tensor(index):
            index = index.tolist()

        # self.batch_end = (index + 1) * self.batch_size
        # list_indexes = self.indexes[index * self.batch_size:self.batch_end]
        list_indexes = self.indexes_test[index]

        # Generate data
        # X, y = self.__data_generation(list_indexes)
        sample = self.__test_generation(list_indexes)
        if self.transform:
            # X = self.transform(X)
            # y = self.transform(y)
            sample = self.transform(sample)
        return sample  # X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.RandomState(self.length).shuffle(self.file_frame)
            self.indexes = [xx for sublist in self.file_frame for xx in sublist]
            self.start_hw = [(np.random.randint(self.hs), np.random.randint(self.ws)) for _ in self.indexes]
            self.files_hw = [[(np.random.randint(self.hs), np.random.randint(self.ws)) for _ in file_ids] for
                             file_ids in self.files_ID]

    def pool(self, data):
        if self.do_pool:
            length = len(data.shape)
            if length == 3:
                data = np.expand_dims(data, axis=0)
            data = data.astype(np.float32).transpose(0, 3, 1, 2)
            data = torch.nn.AvgPool2d(kernel_size=g_scale, stride=g_scale)(torch.from_numpy(data))
            data = data.numpy().transpose(0, 2, 3, 1).astype('b')
            if length == 3:
                s = data.shape
                data = data.reshape(*s[1:])
        return data

    def crop(self, data):
        if self.do_crop:
            h_i, w_i = self.files_hw[self.city_index][self.file_index]
            length = len(data.shape)
            if length == 4:
                data = data[:, h_i:(h_i + self.height), w_i:(w_i + self.width), :]
            # else:  # static data (don't crop here)
            #     data = data[h_i:(h_i + self.height), w_i:(w_i + self.width), :]

        return data

    def get_data(self, file_path):
        """
        Given a file path, loads test file (in h5 format).
        Returns: tensor of shape (number_of_test_cases = 288, 496, 435, 3)
        """
        # load h5 file
        fr = h5py.File(file_path, 'r')
        # a_group_key = list(fr.keys())[0]
        # data_out = fr[a_group_key][()]
        data_out = fr['array'][()]
        if self.model_part:
            return data_out[..., self.mod_start[0]: self.mod_end[0], self.mod_start[1]: self.mod_end[1], :]

        if self.scale_type in ['crop']:
            return self.crop(data_out)
        else:  # pooling
            return self.pool(data_out)

    def write_data(self, data_in, file_path):
        """
        write data in gzipped h5 format.
        """
        f = h5py.File(file_path, 'w', libver='latest')
        dset = f.create_dataset('array', shape=data_in.shape, data=data_in, compression='gzip', compression_opts=9)
        # _ = f.create_dataset('array', shape=data_in.shape, data=data_in,
        #                      compression='gzip', compression_opts=9)
        f.close()

    def process_output(self, data):
        x = data.cpu().numpy() * 255.0
        x_shape = x.shape
        return x.reshape(x_shape[0], -1, self.n_out_channel, *x_shape[2:]).transpose(0, 1, 3, 4, 2).astype(np.uint8)

    def process_input(self, data):
        d_shape = data.shape
        return data.transpose(1, 2, 0, 3).reshape(*d_shape[1:3], -1)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []  # np.empty((self.batch_size, *self.dim, self.n_channels))
        y = []  # np.empty((self.batch_size), dtype=int)
        X_static = []
        for i, list_index in enumerate(list_indexes):
            file_index = list_index[0]
            start_idx = list_index[1]
            city_index = list_index[2]
            if (self.file_index != file_index) and (self.city_index != city_index):
                self.file_index = file_index
                self.city_index = city_index
                self.data = (self.get_data(os.path.join(self.data_dir[self.city_index],
                                                        self.files_ID[self.city_index][self.file_index])) /
                             255.).astype(np.float32)
            # Store sample
            mid_idx = start_idx + self.n_frame_in
            end_idx = [mid_idx + x // 5 - 1 for x in self.times_out]
            # working with random crop
            h_i, w_i = self.files_hw[self.city_index][self.file_index]
            # seq_x = self.data[start_idx:mid_idx, :, :, :]
            # x_shape = seq_x.shape
            # X.append(seq_x.transpose(1, 2, 0, 3).reshape(*x_shape[1:3], -1))
            X.append(self.process_input(self.data[start_idx:mid_idx, :, :, :]))
            # if self.scale_type is 'crop':
            #     X.append(self.process_input(self.data[start_idx:mid_idx, h_i:(h_i + self.height),
            #                                 w_i:(w_i + self.width), :]))
            # else:  # 'pooling'
            #     X.append(self.process_input(self.data[start_idx:mid_idx, :, :, :]))

            # Store result
            # seq_y = self.data[end_idx, :, :, :self.n_out_channel]
            # y_shape = seq_y.shape
            # y.append(seq_y.transpose(1, 2, 0, 3).reshape(*y_shape[1:3], -1))
            y.append(self.process_input(self.data[end_idx, :, :, :self.n_out_channel]))
            # if self.scale_type is 'crop':
            #     y.append(self.process_input(self.data[end_idx, h_i:(h_i + self.height),
            #                                 w_i:(w_i + self.width), :self.n_out_channel]))
            # else:  # pooling
            #     y.append(self.process_input(self.data[end_idx, :, :, :self.n_out_channel]))

            # Store static data
            # X_static.append(self.static_data[self.city_index])
            # if self.scale_type in ['crop']:
            #     x_static = self.static_data[self.city_index][h_i:(h_i + self.height), w_i:(w_i + self.width), ...]
            # else:  # pooling
            #     x_static = self.static_data[self.city_index]
            x_static = self.static_data[self.city_index]
            if self.use_time_slot:
                x_static = np.concatenate([x_static, (start_idx / 288) * np.ones_like(x_static[..., :1])], axis=-1)

            X_static.append(x_static)
        return [np.stack(X), np.stack(X_static)], np.stack(y)

    def __test_generation(self, list_indexes):
        'Generates test data containing varied samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        self.file_index, start_idx, self.city_index = list_indexes
        self.data = (self.get_data(os.path.join(self.data_dir[self.city_index],
                                                self.files_ID[self.city_index][self.file_index])) /
                     255.).astype(np.float32)
        seq_x = self.data
        x_shape = seq_x.shape
        X = seq_x.transpose(0, 2, 3, 1, 4).reshape(x_shape[0], *x_shape[2:4], -1)
        X_static = np.repeat(self.static_data[self.city_index][np.newaxis, ...], x_shape[0], axis=0)

        return [np.stack(X), np.stack(X_static)]  # , np.stack(y) # no target
