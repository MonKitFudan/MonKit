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
Run the following code to configure the environment，If there is any problem with the installation, please refer to [MaskTrack R-CNN](https://github.com/youtubevos/MaskTrackRCNN)
```
conda create -n MaskTrackRCNN -y
conda activate MaskTrackRCNN
conda install -c pytorch pytorch=0.4.1 torchvision cuda92 -y
conda install -c conda-forge cudatoolkit-dev=9.2 opencv -y
conda install cython pillow=7 -y
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
bash compile.sh
pip install .
```

## Code
If you have a new video, you can run the following code to track the monkey.
### Testing
```
python demo/demo_mot_vis.py configs/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019.py --input .../video.mp4 --output video_output.mp4 --video 'video_name' --device 'cuda:1'
```

