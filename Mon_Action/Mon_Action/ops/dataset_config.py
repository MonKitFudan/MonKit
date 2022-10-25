# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

# ROOT_DATASET = '/home/fair/Desktop/xzf/PiL14/'
# ROOT_DATASET = '/home1/xiaozf/pycharmpro/tsn/PiL13/'
ROOT_DATASET = '/home1/lyr/CODE/DATA/'
# ROOT_DATASET = '/home1/xiaozf/UCF/'
# ROOT_DATASET = '/home/fair/Desktop/xzf/HMDB51/'
# ROOT_DATASET = '/home1/xiaozf/HMDB/'
# ROOT_DATASET = '/Users/amy/PycharmProjects/temporal-shift-module-master/'
def return_pil13(modality):
    filename_categories = 3
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'monkey_frame'
        filename_imglist_train = 'file_list/monkey_train_list.txt'
        filename_imglist_val = 'file_list/monkey_val_list.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'monkey_frame'
        filename_imglist_train = 'file_list/monkey_train_list.txt'
        filename_imglist_val = 'file_list/monkey_val_list.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_pil12(modality):
    filename_categories = 12
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'frames'
        filename_imglist_train = 'file_list/pil12_train_list.txt'
        filename_imglist_val = 'file_list/pil12_val_list.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'frames'
        filename_imglist_train = 'file_list/pil12_train_list.txt'
        filename_imglist_val = 'file_list/pil12_val_list.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_ucf101(modality):
    filename_categories = "file_list/classInd.txt"
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'ucf-image'
        filename_imglist_train = 'file_list/ucf101_train_split_1.txt'
        filename_imglist_val = 'file_list/ucf101_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf-image'
        filename_imglist_train = 'file_list/ucf101_train_split_1.txt'
        filename_imglist_val = 'file_list/ucf101_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    print("hmdb51@@@@@@@@@@@@@@@@@@@@")
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51_frames'
        filename_imglist_train = 'file_list/hmdb51_train_split_1.txt'
        filename_imglist_val = 'file_list/hmdb51_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51_frames'
        filename_imglist_train = 'file_list/hmdb51_train_split_1.txt'
        filename_imglist_val = 'file_list/hmdb51_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 'something/v2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-frames'
        filename_imglist_train = 'something/v2/train_videofolder.txt'
        filename_imglist_val = 'something/v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'ucf101_frames'
        filename_imglist_train = 'file_list/ucf101_rgb_train_split_1.txt'
        filename_imglist_val = 'file_list/ucf101_rgb_train_split_index_list.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        print(modality)
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'pil13': return_pil13,'pil12': return_pil12,'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51,
                   'kinetics': return_kinetics }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix