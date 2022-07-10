# Posture estimation Documentation

## Installation: how to install MonKit-Posture estimation
### Experimental setup
All experiments were implemented in PyTorch 1.7.1 for deep models with Nvidia Quadro RTX8000 GPUs (Memory: 48GB). 
The CPU is Intel Xeon Gold 5220R (2.2GHz, 24 Cores). The OS is Ubuntu 18.04.

### Environment configuration

## Code
This code is based on the [HRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) codebase.
### Train
RGB Train
```
python main.py pil12 RGB \
    --arch resnest50 --num_segments 8 \
    --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
    --batch-size 1 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres --dense_sample
```
### Test
```
python .../tools/test.py --cfg .../experiments/monkey/hrnet/w48_256x256_adam_lr1e-3_joints_14.yaml TEST.MODEL_FILE .../output/monkey_joints_14/pose_hrnet/w48_256x256_adam_lr1e-3_joints_14/model_best.pth
```
