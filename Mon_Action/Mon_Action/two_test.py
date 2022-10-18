# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# Notice that this file has been modified to support ensemble testing

import argparse
import time
import os
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('--dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics','PiL'],default='PiL')
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'],default='RGB')
parser.add_argument('--arc', type=str, default="resnest50")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.8)
parser.add_argument('--flow_prefix', type=str, default='flow_')
parser.add_argument('--weights', type=str,default='checkpoint/TSM_pil10_RGB_resnest50_shift8_blockres_avg_segment8_e50_dense_2022_01_07/ckpt.pth.tar')
parser.add_argument('--weights2', type=str,default='checkpoint/TSM_pil10_Flow_resnest50_shift8_blockres_avg_segment8_e50_dense_2022_01_07/ckpt.pth.tar')

# may contain splits
parser.add_argument('--test_segments', type=str, default=8)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--test_list', type=str, default="/home2/lyr/CODE/DATA/PiL14/file_list/pil10_val_list.txt")
parser.add_argument('--csv_file', type=str, default=None)

parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0，1"

class AverageMeter(object):  #
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None

weights = args.weights
weights2 = args.weights2
modality_list = []
modality_list_flow = []

test_segments_list = [args.test_segments]
this_test_segments = test_segments_list[0]

# if args.test_list is not None:
#     test_file_list = args.test_list.split(',')
# else:
#     test_file_list = [None] * len(weights_list)

is_shift, shift_div, shift_place = parse_shift_option_from_log_name(weights)
this_arch = args.arc
modality_list.append('RGB')
modality_list_flow.append('Flow')
if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
elif args.dataset == 'PiL':
    num_class = 10
else:
    raise ValueError('Unknown dataset '+args.dataset)


print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))

print(num_class,this_test_segments,is_shift,this_arch,args.crop_fusion_type,args.img_feature_dim,args.pretrain,shift_div,shift_place,)
net = TSN(num_class,this_test_segments, 'RGB', #### 改了
              base_model=this_arch,
              consensus_type=args.crop_fusion_type,
              img_feature_dim=args.img_feature_dim,
              pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in weights,
              # dropout=args.dropout,
              )

is_shift, shift_div, shift_place = parse_shift_option_from_log_name(weights2)
net2 = TSN(num_class,this_test_segments, 'Flow',   ###改了
          base_model=this_arch,
          consensus_type=args.crop_fusion_type,
          img_feature_dim=args.img_feature_dim,
          pretrain=args.pretrain,
          is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
          non_local='_nl' in weights2,
          # dropout=args.dropout,
          )

checkpoint = torch.load("/home2/lcx_lyr/temporal-shift-module-master/"+args.weights)
print("rgb model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
checkpoint2 = torch.load("/home2/lcx_lyr/temporal-shift-module-master/"+args.weights2)
print("flow model epoch {} best prec@1: {}".format(checkpoint2['epoch'], checkpoint2['best_prec1']))



base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias',
                }
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)
net.load_state_dict(base_dict)
# print(net)
base_dict2 = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint2['state_dict'].items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias',
                }

for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)
net2.load_state_dict(base_dict2)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
    cropping2 = torchvision.transforms.Compose([
        GroupScale(net2.scale_size),
        GroupCenterCrop(net2.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
    cropping2 = torchvision.transforms.Compose([
        GroupOverSample(net2.input_size, net2.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))
data_loader = torch.utils.data.DataLoader(
            TSNDataSet("", args.test_list,num_segments=args.test_segments,
                       new_length=1,
                       modality='RGB',
                       image_tmpl='img_{:05d}.jpg',
                       test_mode=True,
                       remove_missing = 1,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                       ]), dense_sample=True, twice_sample=args.twice_sample),
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True  ##原来num_workers=args.workers
        )
data_loader2 = torch.utils.data.DataLoader(
            TSNDataSet("", args.test_list,num_segments=args.test_segments,
                       new_length=5,
                       modality='Flow',
                       image_tmpl='flow_{}_{:05d}.jpg',
                       test_mode=True,
                       remove_missing= 1,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net2.input_mean, net2.input_std),
                       ]), dense_sample=True, twice_sample=args.twice_sample),
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True,  ###注意乘以了2
        )

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))


net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=[0])
net.eval()
net2 = torch.nn.DataParallel(net2.cuda(devices[0]), device_ids=[0])
net2.eval()

data_gen = enumerate(data_loader)
data_gen2 = enumerate(data_loader2)

total_num = len(data_loader.dataset)
output = []
output2 = []


def eval_video(video_data):
    net.eval()
    i, data, label = video_data
    num_crop = args.test_crops*2
    batch_size = label.numel()

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)
    data_in = data.view(-1, length, data.size(2), data.size(3))
    data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
    rst = net(data_in)
    rst = rst.reshape(batch_size, num_crop, -1).mean(1)

    # rst = net(input_var).data.cpu().numpy().copy()
    rst = rst.data.cpu().numpy().copy()
    rst = rst.reshape(batch_size, num_class)
    return i, rst.reshape(batch_size, num_class), label[0]


def eval_video2(video_data):
    net2.eval()
    i, data, label = video_data
    num_crop = args.test_crops*2
    batch_size = label.numel()
    length =10

    data_in = data.view(-1, length, data.size(2), data.size(3))
    data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
    rst = net2(data_in)
    rst = rst.reshape(batch_size, num_crop, -1).mean(1)
    # rst = net(input_var).data.cpu().numpy().copy()
    rst = rst.data.cpu().numpy().copy()
    rst = rst.reshape(batch_size, num_class)


    return i, rst.reshape(batch_size, num_class), label[0]


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)



for i, (data, label) in data_gen:
    if i >= max_num:
        break
    rst = eval_video((i, data, label))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) /int((i+1))))


for i, (data, label) in data_gen2:
    if i >= max_num:
        break
    rst2 = eval_video2((i, data, label))
    output2.append(rst2[1:])
    cnt_time2 = time.time() - proc_start_time
    print('2 - video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time2) / (i+1)))


video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
video_pred2 = [np.argmax(np.mean(x[0], axis=0)) for x in output2]
video_labels = [x[1] for x in output]
video_labels2 = [x[1] for x in output2]

output_1 = []
output_2 = []
for x in output:
    # print(x[0])
    output_1.append(x[0]/100)
for x in output2:
    # print(x[0])
    output_2.append(x[0]/100)
output_two = []
for i in range(len(output)):
    output_two.append(output_1[i]+1.5*output_2[i])

video_pres_two = [np.argmax(np.mean(x, axis=0)) for x in output_two]
video_labels_two = [x[1] for x in output]

# cf = confusion_matrix(video_labels, video_pred).astype(float)
cf_two = confusion_matrix(video_labels_two, video_pres_two).astype(float)

# cls_cnt = cf.sum(axis=1)
# cls_hit = np.diag(cf)

cls_cnt_two = cf_two.sum(axis=1)
cls_hit_two = np.diag(cf_two)

# cls_acc = cls_hit / cls_cnt

cls_acc_two = cls_hit_two/cls_cnt_two

print(cls_acc_two)

print('Accuracy {:.02f}%'.format(np.mean(cls_acc_two) * 100))

if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)







