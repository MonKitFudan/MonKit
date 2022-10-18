import os
import glob
import fnmatch
import random

root = r"/ssd1/cai/TSM-action/UCF101/jpg/"      # 抽帧后的图片存放目录文件夹，用于写到txt文件中在构建数据集的时候读取

def parse_ucf_splits():
    class_ind = [x.strip().split() for x in open(r'G:\UCF101\label/classInd.txt')]       # 类别txt
    class_mapping = {x[1]:int(x[0])-1 for x in class_ind}

    def line2rec(line):
        items = line.strip().split('/')
        label = class_mapping[items[0]]
        vid = items[1].split('.')[0]
        return vid, label

    splits = []
    for i in range(1, 4):
        train_list = [line2rec(x) for x in open(r'G:\UCF101\label/trainlist{:02d}.txt'.format(i))]       # 训练集txt
        test_list = [line2rec(x) for x in open(r'G:\UCF101\label/testlist{:02d}.txt'.format(i))]          # 测试集txt
        splits.append((train_list, test_list))
    return splits

split_parsers = dict()
split_parsers['ucf101'] = parse_ucf_splits()

def parse_split_file(dataset):
    sp = split_parsers[dataset]
    return tuple(sp)

def parse_directory(path, rgb_prefix='image_', flow_x_prefix='flow_x_', flow_y_prefix='flow_y_'):
    """
    Parse directories holding extracted frames from standard benchmarks
    """
    print('parse frames under folder {}'.format(path))
    frame_folders = []
    frame = glob.glob(os.path.join(path, '*'))
    for frame_name in frame:
        frame_path = glob.glob(os.path.join(frame_name, '*'))
        frame_folders.extend(frame_path)

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list]
        return cnt_list

    # check RGB
    rgb_counts = {}
    flow_counts = {}
    dir_dict = {}
    for i,f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix, flow_x_prefix, flow_y_prefix))
        k = f.split('\\')[-1]
        rgb_counts[k] = all_cnt[0]
        dir_dict[k] = f

        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError('x and y direction have different number of flow images. video: '+f)
        flow_counts[k] = x_cnt
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

    print('frame folder analysis done')
    return dir_dict, rgb_counts, flow_counts

def build_split_list(split_tuple, frame_info, split_idx, shuffle=False):
    split = split_tuple[split_idx]

    def build_set_list(set_list):
        rgb_list, flow_list = list(), list()
        for item in set_list:
            frame_dir = frame_info[0][item[0]]
            frame_dir = root + frame_dir.split('\\')[-2] +'/'+ frame_dir.split('\\')[-1]

            rgb_cnt = frame_info[1][item[0]]
            flow_cnt = frame_info[2][item[0]]
            rgb_list.append('{} {} {}\n'.format(frame_dir, rgb_cnt, item[1]))
            flow_list.append('{} {} {}\n'.format(frame_dir, flow_cnt, item[1]))
        if shuffle:
            random.shuffle(rgb_list)
            random.shuffle(flow_list)
        return rgb_list, flow_list
    train_rgb_list, train_flow_list = build_set_list(split[0])
    test_rgb_list, test_flow_list = build_set_list(split[1])
    return (train_rgb_list, test_rgb_list), (train_flow_list, test_flow_list)

spl = parse_split_file('ucf101')
f_info = parse_directory(r"G:\UCF101\jpg")    # 存放抽帧后的图片

out_path = r"G:\UCF101\label"      # 标签路径
dataset = "ucf101"

for i in range(max(3,len(spl))):
    lists = build_split_list(spl,f_info,i)
    open(os.path.join(out_path, '{}_rgb_train_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[0][0])
    open(os.path.join(out_path, '{}_rgb_val_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[0][1])
    # open(os.path.join(out_path, '{}_flow_train_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[1][0])
    # open(os.path.join(out_path, '{}_flow_val_split_{}.txt'.format(dataset, i + 1)), 'w').writelines(lists[1][1])
