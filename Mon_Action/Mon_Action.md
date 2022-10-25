# Monkey Action

The specific network block diagram is implemented (Figure a). The tag of the category was firstly input into each action video. The duration of action videos were about 1-4s, which can cover all the daily actions of the monkey. From each short video, through random sampling strategy with sparse temporal grouping, 8 frames representing each short video action were extracted. The corresponding optical flow picture and RGB flow information were input into the temporal shift (TS) and split attention (SA) (TSSA) network, respectively through the parts of RGB net and optical flow net for training. In each net module, resnet-50 is used as the backbone network, and the self-attention mechanism module of the group is added to the feature map position of each layer, and the feature map is divided into different cardinals. To increase the importance of certain cardinal and improve the performance of the entire network, a series of transformations are multiplied by different weights. Finally, the corresponding action category is output through softmax and onehot encoding. The evaluation index used here was top-1 and the final accuracy rate reached to 98.99% (RGB 89.83%, Flow 93.05%, RGB+Flow 98.99%) using two-stream neural network based on TSSA fusion.

![image](https://user-images.githubusercontent.com/58841760/192127509-964e16dd-2c38-457d-a6be-7805235d987a.png)

Two-stream action recognition model and its heatmaps of feature visualization. (a) Architecture Overview of TSSA Network. (i) The input of model by randomly sampling of video segments; (ii) RGB and optical flow are two modalities of videos, and each serves as input into the two-stream model; (iii) The two nets do not share parameters but own the same architecture. The stack of Res-blocks is used as the backbone; The shift module and the split attention module are contained in blocks. (iv) The output of the previous block was used as the input of the next block for feature extraction. Using average fusion to get the prediction of the single-stream net, and finally combining these class scores to get the final prediction. (b) Grad-Cam++ Heat Maps of Action Recognition. The heat maps obtained test videos which were classified under the trained model. Colors represent different weights (scale to 0-1, blue to red) that means the importance of the area related to the prediction result. The red area in the frames provided the most important discriminative features for the model in the final predictions.

## Two-Stream Model based on TSSA(temporal shift and split attention) for action detection

We utilized the Grad-Cam++ 27, a method of feature visualization of CNN model predictions, to construct heat maps shown in Figure b. The heat map reflected the different contributions (attention) in different areas for the results of classification. The weighted combination of the positive partial derivatives from last convolutional layer in the feature maps was used to provide specific class score for corresponding label.

This study is based on TSM: Temporal Shift Module for Efficient Video Understanding.(***Lin, J., Gan, C., & Han, S. (2019). TSM: Temporal shift module for efficient video understanding. In Proceedings of the IEEE/CVF International Conference on  Computer Vision.***).

## Installation: how to install MonKit-Action Recognition
### Experimental setup
All experiments were implemented in PyTorch 1.7.1 for deep models with Nvidia Quadro RTX8000 GPUs (Memory: 48GB). 
The CPU is Intel Xeon Gold 5220R (2.2GHz, 24 Cores). The OS is Ubuntu 18.04.


### Environment configuration


## Code Usage
This code is based on the [TSM](https://github.com/mit-han-lab/temporal-shift-module) codebase.
### Training
Detailed command reference scripts

RGB Train
```
python main.py pil13 RGB \
    --arch resnest50 --num_segments 8 \
    --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
    --batch-size 1 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres --dense_sample
```
Flow Train
```
python main.py pil13 Flow \
    --arch resnest50 --num_segments 8 \
    --gd 20 --lr 0.001 --lr_steps 20 40 --epochs 50 \
    --batch-size 72 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres  --dense_sample
```
### Validation
Detailed command reference scripts.(Note: pil13 refers to dataset)
```
python test_models.py pil13 \
    --weights=checkpoint/TSM_pil14_Flow_resnest50_shift8_blockres_avg_segment8_e50_dense/ckpt.pth.tar \
    --test_segments=8 --batch_size=16 -j 24 --test_crops=3  --twice_sample --crop_fusion_type=avg
```
### Testing
```
python tsm_classify.py --video_list 'videoname'
```


