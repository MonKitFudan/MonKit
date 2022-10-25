import argparse
import time
import cv2
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset_test import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from torch.nn import functional as F
from numpy.random import randint
import xlwt
import os
import atexit

# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
# parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
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
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)

parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=[0])
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')

parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--arc', type=str, default='resnest50')
parser.add_argument('--root_path', type=str, default='../data_flow/')
parser.add_argument('--excel_path', type=str, default='../path')
parser.add_argument('--video_list', type=str, default='videoname')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0，1"
print(args.root_path)

max_length = 8

class AverageMeter(object):
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

weights_list = ['checkpoint/TSM_pil10_RGB_resnest50_shift8_blockres_avg_segment8_e50_dense_2022_01_22/ckpt.pth.tar',
                'checkpoint/TSM_pil10_Flow_resnest50_shift8_blockres_avg_segment8_e50_dense_2022_01_22/ckpt.pth.tar']
this_weights = weights_list[0]
test_segments_list = [args.test_segments]
this_test_segments = test_segments_list[0]

if args.coeff is None:
    coeff_list = [1] * len(weights_list)
else:
    coeff_list = [float(c) for c in args.coeff.split(',')]

if args.test_list is not None:
    test_file_list = args.test_list.split(',')
else:
    test_file_list = [None] * len(weights_list)

data_iter_list = []
net_list = []
data_iter_list_flow = []
net_list_flow = []
modality_list = []
modality_list_flow = []

total_num = None

is_shift, shift_div, shift_place = parse_shift_option_from_log_name(weights_list[0])
this_arch = args.arc
modality_list.append('RGB')
modality_list_flow.append('Flow')
num_class=args.num_class
print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))

net_rgb = TSN(num_class, this_test_segments if is_shift else 1, 'RGB',
              base_model=this_arch,
              consensus_type=args.crop_fusion_type,
              img_feature_dim=args.img_feature_dim,
              pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in this_weights,
              )

this_weights = weights_list[1]
is_shift, shift_div, shift_place = parse_shift_option_from_log_name(weights_list[1])

net_flow = TSN(num_class, this_test_segments if is_shift else 1, 'Flow',
          base_model=this_arch,
          consensus_type=args.crop_fusion_type,
          img_feature_dim=args.img_feature_dim,
          pretrain=args.pretrain,
          is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
          non_local='_nl' in this_weights,
          )


if 'tpool' in this_weights:
    from ops.temporal_shift import make_temporal_pool
    make_temporal_pool(net_rgb.base_model, test_segments_list)  # since DataParallel
    make_temporal_pool(net_flow.base_model, test_segments_list)

checkpoint = torch.load('/home2/lyr/CODE/temporal-shift-module-master/checkpoint/TSM_pil10_RGB_resnest50_shift8_blockres_avg_segment8_e50_dense_2022_01_22/ckpt.pth.tar')
checkpoint = checkpoint['state_dict']
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}

replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias',
                }
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)


net_rgb.load_state_dict(base_dict)

checkpoint_flow = torch.load('/home2/lyr/CODE/temporal-shift-module-master/checkpoint/TSM_pil10_Flow_resnest50_shift8_blockres_avg_segment8_e50_dense_2022_01_22/ckpt.pth.tar')#,map_location='cpu'
checkpoint_flow = checkpoint_flow['state_dict']

base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint_flow.items())}

replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias',
                }

for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)
net_flow.load_state_dict(base_dict)


input_size = net_rgb.scale_size if args.full_res else net_rgb.input_size
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net_rgb.scale_size),
        GroupCenterCrop(input_size),
    ])
elif args.test_crops == 3:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupFullResSample(input_size, net_rgb.scale_size, flip=False)
    ])
elif args.test_crops == 5:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net_rgb.scale_size, flip=False)
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net_rgb.scale_size)
    ])
else:
    raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))


net_rgb.eval()
net_flow.eval()

output = []


def eval_video(video_data, net, this_test_segments, modality):
    net.eval()
    with torch.no_grad():
        i, data, label = video_data
        batch_size = label.numel()
        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if args.twice_sample:
            num_crop *= 2

        if modality == 'RGB':
            length = 3
        elif modality == 'Flow':
            length = 10
        elif modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality "+ modality)

        data_in = data.view(-1, length, data.size(2), data.size(3))
        if is_shift:
            data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
        rst = net(data_in)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        if args.softmax:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)

        rst = rst.data.cpu().numpy().copy()
        if net.is_shift:
            rst = rst.reshape(batch_size, num_class)
        else:
            rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

        return i, rst, label


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else total_num

top1 = AverageMeter()
top5 = AverageMeter()


def behaviour(num):
    if num == 0:
        return 'Climb'
    if num == 1:
        return 'Turn'
    if num == 2:
        return 'Hang'
    if num == 3:
        return 'Jump'
    if num == 4:
        return 'MoveDown'
    if num == 5:
        return 'LieDown'
    if num == 6:
        return 'Walk'
    if num == 7:
        return 'Shake'
    if num == 8:
        return 'SitDown'
    if num == 9:
        return 'StandUp'

def computeVar(ran_flow,path):
    var = 0
    for i in ran_flow:
        flow_x = cv2.imread(path + "/flow_x_" + str(i).zfill(max_length) + '.jpg')
        flow_y = cv2.imread(path + "/flow_y_" + str(i).zfill(max_length) + '.jpg')

        var += max(np.var(np.array(np.sum(flow_x, axis=2)/3).flatten()),
                   np.var(np.array(np.sum(flow_y, axis=2)/3).flatten()))
    return var/len(ranp_flow)

#==================================================
# 创建一个workbook 设置编码
workbook = xlwt.Workbook(encoding='utf-8')
# 创建一个worksheet
worksheet0 = workbook.add_sheet('0.5_'+args.video_list, cell_overwrite_ok=True)
worksheet1 = workbook.add_sheet('0.75_'+args.video_list, cell_overwrite_ok=True)
worksheet2 = workbook.add_sheet('1.0_'+args.video_list, cell_overwrite_ok=True)
worksheet3 = workbook.add_sheet('1.25_'+args.video_list, cell_overwrite_ok=True)
worksheet4 = workbook.add_sheet('1.5_'+args.video_list, cell_overwrite_ok=True)

# 写入表头
worksheet0.write(0, 0, label='video/behavior')
worksheet0.write(0, 1, label='time1')
worksheet0.write(0, 2, label='time2')
worksheet0.write(0, 3, label='duration/s')
worksheet0.write(0, 4, label='variance(f)')

worksheet1.write(0, 0, label='video/behavior')
worksheet1.write(0, 1, label='time1')
worksheet1.write(0, 2, label='time2')
worksheet1.write(0, 3, label='duration/s')
worksheet1.write(0, 4, label='variance(f)')

worksheet2.write(0, 0, label='video/behavior')
worksheet2.write(0, 1, label='time1')
worksheet2.write(0, 2, label='time2')
worksheet2.write(0, 3, label='duration/s')
worksheet2.write(0, 4, label='variance(f)')

worksheet3.write(0, 0, label='video/behavior')
worksheet3.write(0, 1, label='time1')
worksheet3.write(0, 2, label='time2')
worksheet3.write(0, 3, label='duration/s')
worksheet3.write(0, 4, label='variance(f)')

worksheet4.write(0, 0, label='video/behavior')
worksheet4.write(0, 1, label='time1')
worksheet4.write(0, 2, label='time2')
worksheet4.write(0, 3, label='duration/s')
worksheet4.write(0, 4, label='variance(f)')

i = 0  # test video position
n = 1  # frame position

window = 40
fps = 30
ranp_rgb = np.multiply(list(range(8)), 10) + randint(10, size=8) + randint(i * window, i * window + 1, size=8) + 1
ranp_flow = np.multiply(list(range(8)), 10) + randint(10, size=8) + randint(i * window, i * window + 1, size=8) + 1

e_row_0 = 0
e_row_1 = 0
e_row_2 = 0
e_row_3 = 0
e_row_4 = 0

lastbeh_0 = ''
lastbeh_1 = ''
lastbeh_2 = ''
lastbeh_3 = ''
lastbeh_4 = ''

lasttime1_0 = 0
lasttime1_1 = 0
lasttime1_2 = 0
lasttime1_3 = 0
lasttime1_4 = 0

duration_0 = 0
duration_1 = 0
duration_2 = 0
duration_3 = 0
duration_4 = 0

lastvar_0 = 0
lastvar_1 = 0
lastvar_2 = 0
lastvar_3 = 0
lastvar_4 = 0

while True:
    if not (os.path.exists(args.root_path+'/'+args.video_list+"/img_" + str(ranp_rgb[-1]).zfill(max_length) + '.jpg') and os.path.exists(args.root_path+'/'+args.video_list+"/flow_x_" + str(ranp_flow[-1]).zfill(max_length) + '.jpg')):
        break
    if i%500==0:
        print(i)

    #每隔80帧，约1.x秒判断一次
    if(n%window==0):
        data_loader_rgb = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path+'/'+args.video_list, num_segments=test_segments_list,
                       pos=ranp_rgb,
                       new_length=1,
                       modality='RGB',
                       image_tmpl='img_{:08d}.jpg',
                       test_mode=True,
                       remove_missing=len(weights_list) == 1,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net_rgb.input_mean, net_rgb.input_std),
                       ]), dense_sample=True, twice_sample=args.twice_sample),
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

        data_loader_flow = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path+'/'+args.video_list, num_segments=test_segments_list,
                       pos=ranp_flow,
                       new_length=5,
                       modality='Flow',
                       image_tmpl='flow_{}_{:08d}.jpg',
                       test_mode=True,
                       remove_missing=len(weights_list) == 1,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           GroupNormalize(net_flow.input_mean, net_flow.input_std),
                       ]), dense_sample=True, twice_sample=args.twice_sample),
            batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )


        proc_start_time = time.time()

        total_num = len(data_loader_rgb.dataset)
        for j, (data, label) in enumerate(data_loader_rgb):
            if j > 0:
                break
            i,r_rgb,label_rgb = eval_video((i, data, label),net_rgb,this_test_segments,'RGB')
            beh_rgb = behaviour(label_rgb)

        for j, (data, label) in enumerate(data_loader_flow):
            if j > 0:
                break
            i,r_flow,label_flow = eval_video((i, data, label),net_flow,this_test_segments, 'Flow')

            beh_flow = behaviour(label_flow)

        r_two = r_rgb+1.5*r_flow

        if r_two.argmax()==13:
            if r_two[:,:-1].argmax()==8 and r_two[0][8]>=6.5:
                beh_two = behaviour(8)
            else:
                beh_two = behaviour(r_two.argmax())
        else:
            beh_two = behaviour(r_two.argmax())

        var = computeVar(ranp_flow,args.root_path+'/'+args.video_list) #算出子视频flow的方差，用来判断猴子是不是no benavior

        if var<0.5:#0.75 1.0 1.25 1.5
            beh_two_0 = 'No Behaviour'
        else:
            beh_two_0 = beh_two
        if var<0.75:
            beh_two_1 = 'No Behaviour'
        else:
            beh_two_1 = beh_two
        if var<1.0:
            beh_two_2 = 'No Behaviour'
        else:
            beh_two_2 = beh_two
        if var<1.25:
            beh_two_3 = 'No Behaviour'
        else:
            beh_two_3 = beh_two
        if var<1.5:
            beh_two_4 = 'No Behaviour'
        else:
            beh_two_4 = beh_two

        if n==window:
            lastvar_0 = var
            lastvar_1 = var
            lastvar_2 = var
            lastvar_3 = var
            lastvar_4 = var
        # if not beh_two =='Walk Upright':
        print(beh_two_0,beh_two_1,beh_two_2,beh_two_3,beh_two_4)
        time2 = n

        if beh_two_0 == lastbeh_0:
            time1_0 = lasttime1_0
            var_0 = lastvar_0+var
        else:
            var_0 = var
            time1_0 = n-window
            e_row_0 += 1

        if beh_two_1 == lastbeh_1:
            time1_1 = lasttime1_1
            var_1 = lastvar_1+var
        else:
            var_1 = var
            time1_1 = n-window
            e_row_1 += 1

        if beh_two_2 == lastbeh_2:
            time1_2 = lasttime1_2
            var_2 = lastvar_2+var
        else:
            var_2 = var
            time1_2 = n-window
            e_row_2 += 1

        if beh_two_3 == lastbeh_3:
            time1_3 = lasttime1_3
            var_3 = lastvar_3+var
        else:
            var_3 = var
            time1_3 = n-window
            e_row_3 += 1

        if beh_two_4 == lastbeh_4:
            time1_4 = lasttime1_4
            var_4 = lastvar_4+var
        else:
            var_4 = var
            time1_4 = n-window
            e_row_4 += 1

        duration_0 = round((time2 - time1_0) / fps, 2)
        lastbeh_0 = beh_two_0
        lasttime1_0 = time1_0
        lastvar_0 = var_0

        duration_1 = round((time2 - time1_1) / fps, 2)
        lastbeh_1 = beh_two_1
        lasttime1_1 = time1_1
        lastvar_1 = var_1

        duration_2 = round((time2 - time1_2) / fps, 2)
        lastbeh_2 = beh_two_2
        lasttime1_2 = time1_2
        lastvar_2 = var_2

        duration_3 = round((time2 - time1_3) / fps, 2)
        lastbeh_3 = beh_two_3
        lasttime1_3 = time1_3
        lastvar_3 = var_3

        duration_4 = round((time2 - time1_4) / fps, 2)
        lastbeh_4 = beh_two_4
        lasttime1_4 = time1_4
        lastvar_4 = var_4
        # 写入excel
        # 参数对应 行, 列, 值

        worksheet0.write(e_row_0, 0, label=beh_two_0)
        worksheet0.write(e_row_0, 1, label="%02d:%02d" % (divmod(time1_0/fps, 60)))
        worksheet0.write(e_row_0, 2, label="%02d:%02d" % (divmod(time2/fps, 60)))
        worksheet0.write(e_row_0, 3, label=duration_0)
        worksheet0.write(e_row_0, 4, label=var_0/int((time2 - time1_0)/window))

        worksheet1.write(e_row_1, 0, label=beh_two_1)
        worksheet1.write(e_row_1, 1, label="%02d:%02d" % (divmod(time1_1 / fps, 60)))
        worksheet1.write(e_row_1, 2, label="%02d:%02d" % (divmod(time2 / fps, 60)))
        worksheet1.write(e_row_1, 3, label=duration_1)
        worksheet1.write(e_row_1, 4, label=var_1)

        worksheet2.write(e_row_2, 0, label=beh_two_2)
        worksheet2.write(e_row_2, 1, label="%02d:%02d" % (divmod(time1_2 / fps, 60)))
        worksheet2.write(e_row_2, 2, label="%02d:%02d" % (divmod(time2 / fps, 60)))
        worksheet2.write(e_row_2, 3, label=duration_2)
        worksheet2.write(e_row_2, 4, label=var_2)

        worksheet3.write(e_row_3, 0, label=beh_two_3)
        worksheet3.write(e_row_3, 1, label="%02d:%02d" % (divmod(time1_3 / fps, 60)))
        worksheet3.write(e_row_3, 2, label="%02d:%02d" % (divmod(time2 / fps, 60)))
        worksheet3.write(e_row_3, 3, label=duration_3)
        worksheet3.write(e_row_3, 4, label=var_3)

        worksheet4.write(e_row_4, 0, label=beh_two_4)
        worksheet4.write(e_row_4, 1, label="%02d:%02d" % (divmod(time1_4 / fps, 60)))
        worksheet4.write(e_row_4, 2, label="%02d:%02d" % (divmod(time2 / fps, 60)))
        worksheet4.write(e_row_4, 3, label=duration_4)
        worksheet4.write(e_row_4, 4, label=var_4)

        cnt_time = time.time() - proc_start_time
        print('video {} done, total {}/{}, average {} sec/video'.format(i, i + 1,
                                                                            total_num,
                                                                            float(cnt_time) / (i + 1)))

        i = i+1
        ranp_rgb = np.multiply(list(range(8)), 10) + randint(10, size=8) + randint(i * window, i * window + 1, size=8)
        ranp_flow = np.multiply(list(range(8)), 10) + randint(10, size=8) + randint(i * window, i * window + 1, size=8)

    n = n + 1

print(args.excel_path+"/"+args.video_list+'.xls')
workbook.save(args.excel_path+"/"+args.video_list+'.xls')
