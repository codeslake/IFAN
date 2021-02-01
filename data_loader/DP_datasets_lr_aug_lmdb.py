'''
REDS dataset
support reading images from lmdb, image folder and memcached
'''
import os
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data

from data_loader.utils import *
import pyarrow as pa
from PIL import Image
import lmdb
import six

class datasets(data.Dataset):
    def __init__(self, config, is_train):
        super(datasets, self).__init__()
        self.config = config
        self.is_train = is_train
        self.h = config.height
        self.w = config.width
        self.norm_val = config.norm_val

        self.env = None
        if is_train:
            l_folder_path_list, l_file_path_list, _ = load_file_list(config.l_path, config.input_path)
        else:
            l_folder_path_list, l_file_path_list, _ = load_file_list(config.VAL.l_path, config.VAL.input_path)

        self.len = int(np.ceil(len(l_file_path_list)))

    def _init_db(self):
        self.env = lmdb.open(self.config.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        keys = pa.deserialize(self.txn.get(b'__keys__'))

        if self.is_train:
            self.keys = [key for key in keys if 'train' in key]
            self.is_augment = True
        else:
            self.keys = [key for key in keys if 'test' in key]
            self.is_augment = False

        self.len = int(np.ceil(len(self.keys)))

    def _read_imgbuf(self, imgbuf):
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = np.array(Image.open(buf).convert('RGB'))/255.
        img = np.expand_dims(img, 0)
        return img

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()

        index = index % self.len

        byteflow = self.txn.get(self.keys[index].encode('ascii'))
        unpacked = pa.deserialize(byteflow)

        l_frame = self._read_imgbuf(unpacked[0])
        r_frame = self._read_imgbuf(unpacked[1])
        gt_frame = self._read_imgbuf(unpacked[2])

        if self.is_augment:
            # Flip
            # if random.uniform(0, 1) <= 0.5:
            #     l_frame = np.flip(l_frame, axis = 3)
            #     r_frame = np.flip(r_frame, axis = 3)
            #     gt_frame = np.flip(gt_frame, axis = 3)

            # Noise
            if random.uniform(0, 1) <= 0.05:
            # if random.uniform(0, 1) >= 0.0:
                row,col,ch = l_frame[0].shape
                mean = 0
                var = random.uniform(0.001, 0.005)
                sigma = var**0.5
                gauss = np.random.normal(mean,sigma,(row,col,ch))
                gauss = gauss.reshape(row,col,ch)

                l_frame = np.expand_dims(l_frame[0] + gauss, axis = 0)
                r_frame = np.expand_dims(r_frame[0] + gauss, axis = 0)
                gt_frame = np.expand_dims(gt_frame[0] + gauss, axis = 0)

            # Grayscale
            if random.uniform(0, 1) <= 0.3:
            # if random.uniform(0, 1) >= 0.0:
                l_frame = np.expand_dims(color_to_gray(l_frame[0]), axis = 0)
                r_frame = np.expand_dims(color_to_gray(r_frame[0]), axis = 0)
                gt_frame = np.expand_dims(color_to_gray(gt_frame[0]), axis = 0)

            # Scaling
            if random.uniform(0, 1) <= 0.5:
            # if random.uniform(0, 1) >= 0.0:
                scale = random.uniform(0.7, 1.0)
                row,col,ch = l_frame[0].shape

                l_frame = np.expand_dims(cv2.resize(l_frame[0], dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA), axis = 0)
                r_frame = np.expand_dims(cv2.resize(r_frame[0], dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA), axis = 0)
                gt_frame = np.expand_dims(cv2.resize(gt_frame[0], dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA), axis = 0)

        cropped_frames = np.concatenate([l_frame, r_frame, gt_frame], axis = 3)

        if self.is_train:
            cropped_frames = crop_multi(cropped_frames, self.h, self.w, is_random = True)
        else:
            cropped_frames = cropped_frames

        l_patches = cropped_frames[:, :, :, :3]
        shape = l_patches.shape
        h = shape[1]
        w = shape[2]
        l_patches = l_patches.reshape((h, w, -1, 3))
        l_patches = torch.FloatTensor(np.transpose(l_patches, (2, 3, 0, 1)))

        r_patches = cropped_frames[:, :, :, 3:6]
        r_patches = r_patches.reshape((h, w, -1, 3))
        r_patches = torch.FloatTensor(np.transpose(r_patches, (2, 3, 0, 1)))

        gt_patches = cropped_frames[:, :, :, 6:9]
        gt_patches = gt_patches.reshape((h, w, -1, 3))
        gt_patches = torch.FloatTensor(np.transpose(gt_patches, (2, 3, 0, 1)))

        return {'l': l_patches[0], 'r': r_patches[0], 'gt': gt_patches[0]}


