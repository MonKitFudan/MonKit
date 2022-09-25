# Monkey Track

**MaskTrack R-CNN was used to track monkey’s position.**

# Monkey Track

![image](https://user-images.githubusercontent.com/58841760/192137415-4bb54ce9-7c90-41c4-85fe-6eb92e74c19b.png)

The representative images with bounding box of MaskTrack R-CNN tracking and 2D skeleton.

This study is based on MaskTrack R-CNN

## Installation
### Experimental setup
All experiments were implemented in PyTorch 1.7.1 for deep models with Nvidia Quadro RTX8000 GPUs (Memory: 48GB). 
The CPU is Intel Xeon Gold 5220R (2.2GHz, 24 Cores). The OS is Ubuntu 18.04.


### Environment configuration


## Code Usage


### Testing
```
python demo/demo_mot_vis.py configs/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py --input /home3/lcx/CODE/DATA/MONKEY/DISEASE_ANALYSIS/data/AUTISM/AUTISM_004/AUTISM_004_5_sample.mp4 --output /home1/lyr2/LCX_CODE/mmtracking-master/LCX/AUTISM_004_5.mp4 --video 'AUTISM_004_5' --device 'cuda:1'
```

