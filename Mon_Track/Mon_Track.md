# Monkey Track

**MaskTrack R-CNN was used to track monkey’s position. The rectangle information of the monkey’s position was intercepted and 
input to the HRNet network to obtain the heatmap. The mean square error (MSE) loss function was used to compare the target 
to get loss. Finally, the 15 bone points were transformed into x and y space coordinates. The accuracy has reached 98.8% 
accuracy in the MiL2D dataset and OpenMonkeyStudio dataset.**

# Monkey Track

![image](https://user-images.githubusercontent.com/58841760/192137415-4bb54ce9-7c90-41c4-85fe-6eb92e74c19b.png)


This study is based on TSM: Temporal Shift Module for Efficient Video Understanding.(***Lin, J., Gan, C., & Han, S. (2019). TSM: Temporal shift module for efficient video understanding. In Proceedings of the IEEE/CVF International Conference on  Computer Vision.***).

## Installation
### Experimental setup
All experiments were implemented in PyTorch 1.7.1 for deep models with Nvidia Quadro RTX8000 GPUs (Memory: 48GB). 
The CPU is Intel Xeon Gold 5220R (2.2GHz, 24 Cores). The OS is Ubuntu 18.04.


### Environment configuration


## Code Usage


### Train

