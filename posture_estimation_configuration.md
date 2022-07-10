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
python /.../tools/train.py --cfg .../experiments/monkey/hrnet/w48_256x256_adam_lr1e-3_joints_14.yaml
```
```
python /.../tools/train.py --cfg /.../experiments/monkey/hrnet/w32_256x192_adam_Ir1e-3_joints_14.yaml
```
### Test
```
python /.../tools/test.py --cfg /.../experiments/monkey/hrnet/w48_256x256_adam_lr1e-3_joints_14.yaml TEST.MODEL_FILE /.../output/monkey_joints_14/pose_hrnet/w48_256x256_adam_lr1e-3_joints_14/model_best.pth
```
