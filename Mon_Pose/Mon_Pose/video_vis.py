from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import json
import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from lib.config import cfg
from core.loss import JointsMSELoss
import dataset.monkey as monkey
from lib.models.pose_hrnet import get_pose_net
from lib.core.function import AverageMeter
from lib.core.function import _print_name_value
from lib.core.evaluate import accuracy, test_accuracy
from lib.core.inference import get_final_preds
from lib.utils.transforms import flip_back
from lib.utils.vis import *

import argparse
from SiamMask.utils.config_helper import load_config
from SiamMask.utils.load_helper import load_pretrain, restore_from
from SiamMask.tools.test import *
from SiamMask.experiments.siammask_sharp.custom import Custom
from os.path import join, isdir, isfile

def main(videoname):
    #dataset='VAL'
    dataset = 'TEST_14'
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset=='MONKEY':
        cfg.merge_from_file(
            '/Volumes/ORANGE/CODE/deep-high-resolution-cpu/experiments/monkey/hrnet/w48_256x256_adam_lr1e-3.yaml')
        # cfg.merge_from_file(
        #     '/Volumes/ORANGE/CODE/deep-high-resolution-cpu/experiments/monkey/hrnet/w32_256x192_adam_Ir1e-3.yaml')
        final_output_dir = '/Volumes/ORANGE/CODE/deep-high-resolution-cpu/output/monkey/pose_hrnet/w48_256x256_adam_lr1e-3'
        tb_log_dir = '/Volumes/ORANGE/CODE/deep-high-resolution-cpu/log/monkey/pose_hrnet/w48_256x256_adam_lr1e-3_2020-10-26-19-12'
    if dataset=='TEST':
        cfg.merge_from_file('/Volumes/ORANGE/CODE/deep-high-resolution-cpu/experiments/TEST/hrnet/w48_256x256_adam_lr1e-3.yaml')
        #cfg.merge_from_file(
        #    '/Volumes/ORANGE/CODE/deep-high-resolution-cpu/experiments/TEST/hrnet/w32_256x192_adam_Ir1e-3.yaml')
        final_output_dir = 'output/TEST/pose_hrnet/w48_256x256_adam_lr1e-3'
        tb_log_dir = 'log/TEST/pose_hrnet/w48_256x256_adam_lr1e-3_2020-10-26-19-12'
        annot_json=cfg.DATASET.ROOT+'/annot/test_'+videoname+'.json'
        path=cfg.DATASET.ROOT+'/frames/'+videoname+'/'
        print('annot_json', annot_json, 'path', path)
        track(annot_json,path)
    if dataset=='TEST_14':
        cfg.merge_from_file('/home1/lcx/CODE/deep-high-resolution/experiments/monkey/hrnet/w48_256x256_adam_lr1e-3_joints_14.yaml')
        #cfg.merge_from_file(
        #    '/Volumes/ORANGE/CODE/deep-high-resolution-cpu/experiments/TEST/hrnet/w32_256x192_adam_Ir1e-3.yaml')
        final_output_dir = 'output/TEST_14/pose_hrnet/w48_256x256_adam_lr1e-3'
        tb_log_dir = 'log/TEST_14/pose_hrnet/w48_256x256_adam_lr1e-3_2020-10-26-19-12'
        annot_json=cfg.DATASET.ROOT+'/annot/test_'+videoname+'.json'
        path=cfg.DATASET.ROOT+'/frames/'+videoname+'/'
        if not os.path.exists(annot_json):
            track(annot_json,path)
        model_state_file= '/home1/lcx/CODE/deep-high-resolution/output/PTH/model_best.pth'
        cfg.freeze()
        model = get_pose_net(cfg, is_train=False)
        model.load_state_dict(torch.load(model_state_file, strict=False))


    #model_state_file= '/Volumes/ORANGE/CODE/deep-high-resolution/output/monkey/pose_hrnet/w32_256x192_adam_Ir1e-3/model_best_WalkUprigh.pth'
    #model_state_file= '/Volumes/ORANGE/CODE/deep-high-resolution/output/monkey/pose_hrnet/w48_256x256_adam_lr1e-3/model_best.pth'
    #model_state_file = '/Volumes/ORANGE/CODE/deep-high-resolution/output/monkey/pose_hrnet/w32_256x192_adam_Ir1e-3/model_best_2020_11.pth'
    #model.load_state_dict(torch.load(model_state_file, map_location=torch.device('cpu')), strict=False)

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    )

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    print('cfg.DATASET.TEST_SET',cfg.DATASET.TEST_SET)

    if dataset == 'MONKEY':
        valid_dataset = monkey(
            cfg, cfg.DATASET.ROOT, 'valid', 'valid',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            #batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            batch_size=1,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=True
        )
        validate(cfg, valid_loader, valid_dataset, model, criterion, final_output_dir, tb_log_dir)

    if dataset == 'SICK':
        test_dataset = monkey(
            cfg, cfg.DATASET.ROOT, 'test', 'test',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=True
        )
        test(cfg,test_loader,test_dataset,model,final_output_dir)

    if dataset == 'TEST':
        test_dataset = monkey(
            cfg, cfg.DATASET.ROOT, 'test_'+videoname, 'test',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=True
        )

        test(cfg,test_loader,test_dataset,model,final_output_dir)

def validate(config, val_loader, val_dataset, model, criterion, output_dir,tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            #tensor_imshow(input[0], title=None)
            #tensor_imshow(outputs[0], title=None)
            #tensor_imshow(target[0], title=None)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.numpy(),
                                               val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy())

                    # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5
                #tensor_imshow(output[0], title=None)
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.numpy(),
                                                 target.numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            msg = 'Test: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time,
                loss=losses, acc=acc)
            print(msg)

            prefix = '{}_{}'.format(
                os.path.join(output_dir, 'val'), i
            )

            #gt,pred,heatmaps_gt,hm_pred=save_debug_images(config, input, meta,[], pred * 4, output, prefix)
            joints, head, gt, pred, heatmaps_gt, hm_pred, pic = save_debug_images(config, input, meta, [], pred * 4,
                                                                                  output, prefix)
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        return perf_indicator

def test(config, test_loader, test_dataset, model, output_dir):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(test_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, meta) in enumerate(test_loader):
            # tensor_imshow(input,'input')
            # compute output
            outputs = model(input)

            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                               test_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy())

                    # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            num_images = input.size(0)
            # measure accuracy and record loss

            pred = test_accuracy(output.numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                    config, output.clone().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            msg = 'Test: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                i, len(test_loader), batch_time=batch_time)
            print(msg)

            prefix = '{}_{}'.format(
                os.path.join(output_dir, 'val'), i
            )
            joints,head,gt,pred,heatmaps_gt,hm_pred,pic=save_debug_images(config, input, meta,[], pred * 4, output, prefix)
            save_result_json(cfg,joints,head,i)

def tensor_imshow(tensor, title=None):
    image = tensor.clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.show()
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def track(annot_json,path):
    track_root='/home1/lcx/CODE/deep-high-resolution-cpu/SiamMask'

    sys.path.append(track_root)
    sys.path.append(track_root+'/experiments/siammask_sharp')
    parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

    parser.add_argument('--resume',
                            default=track_root+'/experiments/siammask_sharp/SiamMask_DAVIS.pth',
                            type=str,
                            metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--config', dest='config',
                            default=track_root+'/experiments/siammask_sharp/config_davis.json',
                            help='hyper-parameter of SiamMask in json format')
    parser.add_argument('--base_path', default=path,
                            help='datasets')
    parser.add_argument('--cpu', action='store_true', help='cpu mode')
    args = parser.parse_args()

    all = []
    lis = [1] * 16
    joints_vis = {'joints_vis': lis}

    if __name__ == '__main__':
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        # Setup Model
        cfg = load_config(args)
        siammask = Custom(anchors=cfg['anchors'])
        if args.resume:
            #assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
            siammask = load_pretrain(siammask, args.resume)

        siammask.eval().to(device)
        # print('siammask',siammask)
        # Parse Image file
        img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))

        index = img_files.index(args.base_path+'img_00001.jpg')
        img_files = img_files[index:]
        ims = [cv2.imread(imf) for imf in img_files]
        # Select ROI
        cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
        center = {'center': [int(x + w / 2), int(y + h / 2)]}
        scale = {'scale': h / 200}
        image = {'image': img_files[0].split('/')[-2]+'/'+img_files[0].split('/')[-1]}
        all_temp = dict(image, **joints_vis)
        all_temp = dict(all_temp, **center)
        all_temp = dict(all_temp, **scale)
        all.append(all_temp)
        with open(annot_json, 'w') as f:
            json.dump(all, f, indent=4)

        toc = 0
        for f, im in enumerate(ims):
            tic = cv2.getTickCount()
            if f == 0:  # init
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            elif f > 0:  # tracking
                state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
                location = state['ploygon'].flatten()
                mask = state['mask'] > state['p'].seg_thr
                im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
                rect = [np.int0(location).reshape((-1, 1, 2))]
                cv2.circle(im, (
                int((rect[0][1][0][0] + rect[0][3][0][0]) / 2), int((rect[0][1][0][1] + rect[0][3][0][1]) / 2)), 1,
                               (255, 255, 255), thickness=6)
                cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                cv2.imshow('SiamMask', im)
                center = {'center': [int((rect[0][1][0][0] + rect[0][3][0][0]) / 2),
                                         int((rect[0][1][0][1] + rect[0][3][0][1]) / 2)]}

                scale1=max(rect[0][0][0][0], rect[0][1][0][0], rect[0][2][0][0],rect[0][3][0][0]) - min(rect[0][0][0][0], rect[0][1][0][0], rect[0][2][0][0],rect[0][3][0][0])
                scale2 = max(rect[0][0][0][1], rect[0][1][0][1], rect[0][2][0][1], rect[0][3][0][1]) - min(rect[0][0][0][1], rect[0][1][0][1], rect[0][2][0][1], rect[0][3][0][1])
                scale=max([scale1,scale2])
                scale = {'scale': scale / 200}
                image = {'image': img_files[f].split('/')[-2]+'/'+img_files[f].split('/')[-1]}
                all_temp = dict(image, **joints_vis)
                all_temp = dict(all_temp, **center)
                all_temp = dict(all_temp, **scale)
                all.append(all_temp)
                with open(annot_json, 'w') as file:
                    json.dump(all, file, indent=4)
                key = cv2.waitKey(1)
                if key > 0:
                    break

            toc += cv2.getTickCount() - tic
        toc /= cv2.getTickFrequency()
        fps = f / toc
        print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))

def save_result_json(cfg,joints,head,i):
    train_json=cfg.DATASET.ROOT+'/annot/train_'+videoname+'.json'
    test_json=cfg.DATASET.ROOT+'/annot/test_'+videoname+'.json'
    with open(test_json, 'r') as f:
        all_temp = json.load(f)
    keypoints_dict = {'joints': joints.astype(int).tolist()}
    all_temp_train=dict(all_temp[i], **keypoints_dict)
    headboxes_src_dict = {'headboxes_src': head}
    all_temp_train = dict(all_temp_train, **headboxes_src_dict)
    all.append(all_temp_train)
    with open(train_json, 'w') as file:
        json.dump(all, file, indent=4)

    #print(all_temp[i])

if __name__ == '__main__':
    #main('')

    files=sorted(os.listdir('/Volumes/ORANGE/CODE/DATA/MONKEY/PiL14/frames'))
    videoname='Depression_010'

    if 'train_'+videoname+'.json' in os.listdir('/Volumes/ORANGE/CODE/DATA/MONKEY/PiL14/annot'):
        print('change a file (annot)')
        sys.exit()


    files=files[files.index(videoname):]
    for videoname in sorted(files):
        all = []
        print(videoname + ' is processing')
        main(videoname)

'''
右腿：紫色  左腿：绿色   右胳膊：棕色  左胳膊：浅蓝
'''

#scp /local/path/local_filename username@servername:/path
#scp /Volumes/ORANGE/CODE bear@10.154.57.19:/home1/lcx/