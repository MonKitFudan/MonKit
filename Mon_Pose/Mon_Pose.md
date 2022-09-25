# Monkey Pose

**Based on MiL dataset, we further established a MiL2D dataset of images with 2D skeleton and key bone points. The MiL2D dataset consists of 15,175 annotated images that span a large variation of poses and positions seen in 13 categories of MiL dataset. Fifteen keypoints of skeleton were marked in details (Figure 4a). MaskTrack R-CNN 28 was used to track monkey’s position (Figure 4b). The dataset includes diverse configurations of cage environments, and the monkeys with the corresponding skeleton points were detected clearly in the MiL2D dataset (Figure 4c).**

![image](https://user-images.githubusercontent.com/58841760/192129066-447be3f9-87a0-45f7-897b-a66d8b5263a2.png)

## MonkeyMonitorKit (MonKit) toolbox and keypoint prediction
By using the MiL2D dataset, the monkey bone recognition task was performed for training and testing process as a toolbox of MonkeyMonitorKit (MonKit) based on high resolution network (HRNet) 29 (Figure 5a). The images from original input video were processed to 256× 340 pixels. MaskTrack R-CNN was used to track monkey’s position (Figure 5b). The rectangle information of the monkey’s position was intercepted and input to the HRNet network to obtain the heatmap (Figure 5c). The mean square error (MSE) loss function was used to compare the target to get loss (Figure 5d). Finally, the 15 bone points were transformed into x and y space coordinates (Figure 5e). The accuracy has reached 98.8% accuracy in the MiL2D dataset and OpenMonkeyStudio dataset.

![image](https://user-images.githubusercontent.com/58841760/192129000-0700959c-5a09-4cab-bd06-c954014b761b.png)

This study is based on TSM: Temporal Shift Module for Efficient Video Understanding.(***Lin, J., Gan, C., & Han, S. (2019). TSM: Temporal shift module for efficient video understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision.***).

## Installation: How to install MonKit-Posture estimation
### Experimental setup
All experiments were implemented in PyTorch 1.7.1 for deep models with Nvidia Quadro RTX8000 GPUs (Memory: 48GB). 
The CPU is Intel Xeon Gold 5220R (2.2GHz, 24 Cores). The OS is Ubuntu 18.04.

### Environment configuration
If using pip install, run the following code. The monkit_pose.txt is under the env folder.
```
pip install -r monkit_pose.txt
```
If using conda install, run the following code. The monkit_pose.yaml is under the env folder.
```
conda env create -f monkit_pose.yaml
```

## Code
This code is based on the [HRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) codebase.
### Training
If you want to perform high-precision training, run the following code.
```
python /.../tools/train.py --cfg .../experiments/monkey/hrnet/w48_256x256_adam_lr1e-3_joints_14.yaml
```
If you want to run faster, run the following code.
```
python /.../tools/train.py --cfg /.../experiments/monkey/hrnet/w32_256x192_adam_Ir1e-3_joints_14.yaml
```
### Validation
The code to run the high precision Validation is as follows.
```
python /.../tools/test.py --cfg /.../experiments/monkey/hrnet/w48_256x256_adam_lr1e-3_joints_14.yaml TEST.MODEL_FILE /.../output/monkey_joints_14/pose_hrnet/w48_256x256_adam_lr1e-3_joints_14/model_best.pth
```
The code to run the quick Validation is as follows.
```
python /.../tools/test.py --cfg /.../experiments/monkey/hrnet/w32_256x192_adam_Ir1e-3_joints_14.yaml TEST.MODEL_FILE /.../output/monkey_joints_14/pose_hrnet/w32_256x192_adam_Ir1e-3_joints_14/model_best.pth
```
### Testing
If you have a new video, you can run the following code for inference.
```
python /.../inference.py --video 'video_name'
```
## Demo
The results of running MonKit are as follows.
