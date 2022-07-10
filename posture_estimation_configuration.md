# Posture Estimation Documentation

## Installation: How to install MonKit-Posture estimation
### Experimental setup
All experiments were implemented in PyTorch 1.7.1 for deep models with Nvidia Quadro RTX8000 GPUs (Memory: 48GB). 
The CPU is Intel Xeon Gold 5220R (2.2GHz, 24 Cores). The OS is Ubuntu 18.04.

### Environment configuration
If using pip install, run the following code.
```
pip install -r monkit_pose.txt
```
If using conda install, run the following code.
```
conda env create -f monkit_pose.yaml
```

## Code
This code is based on the [HRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) codebase.
### Train
If you want to perform high-precision training, run the following code.
```
python /.../tools/train.py --cfg .../experiments/monkey/hrnet/w48_256x256_adam_lr1e-3_joints_14.yaml
```
If you want to run faster, run the following code.
```
python /.../tools/train.py --cfg /.../experiments/monkey/hrnet/w32_256x192_adam_Ir1e-3_joints_14.yaml
```
### Test
The code to run the high precision test is as follows.
```
python /.../tools/test.py --cfg /.../experiments/monkey/hrnet/w48_256x256_adam_lr1e-3_joints_14.yaml TEST.MODEL_FILE /.../output/monkey_joints_14/pose_hrnet/w48_256x256_adam_lr1e-3_joints_14/model_best.pth
```
The code to run the quick test is as follows.
```
python /.../tools/test.py --cfg /.../experiments/monkey/hrnet/w32_256x192_adam_Ir1e-3_joints_14.yaml TEST.MODEL_FILE /.../output/monkey_joints_14/pose_hrnet/w32_256x192_adam_Ir1e-3_joints_14/model_best.pth
```

## Demo
The results of running MonKit are as follows.
![Pose_demo](/images/Pose_demo.jpg)
