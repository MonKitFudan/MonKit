# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])-1

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path,pos=None,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        self.pos = pos
        # if self.dense_sample:
            # print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

    def __getitem__(self, index):
        images = list()
        # print('index',index)
        if self.modality == 'RGB':
            for seg_ind in self.pos:
                img = [Image.open(self.root_path + "/img_" + str(seg_ind).zfill(8) + '.jpg').convert('RGB')]
                images.extend(img)
            process_data = self.transform(images)
            return process_data, 1
        elif self.modality == 'Flow':
            images = list()
            for seg_ind in self.pos:
                for i in range(self.new_length):
                    x_img = Image.open(self.root_path + "/flow_x_" + str(seg_ind).zfill(8) + '.jpg').convert('L')
                    y_img = Image.open(self.root_path + "/flow_y_" + str(seg_ind).zfill(8) + '.jpg').convert('L')
                    seg_imgs = [x_img, y_img]
                    images.extend(seg_imgs)

            process_data = self.transform(images)
            return process_data,1


    def __len__(self):
        return 1
