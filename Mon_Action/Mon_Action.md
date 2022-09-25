# Monkey Action

**We utilized the Grad-Cam++ 27, a method of feature visualization of CNN model predictions, to construct heat maps shown in Figure 2b. The heat map reflected the different contributions (attention) in different areas for the results of classification. The weighted combination of the positive partial derivatives from last convolutional layer in the feature maps was used to provide specific class score for corresponding label.**

***Figure 2***

![image](https://user-images.githubusercontent.com/58841760/192127509-964e16dd-2c38-457d-a6be-7805235d987a.png)

## Two-Stream Model based on TSSA(temporal shift and split attention) for action detection

The specific network block diagram is implemented (Figure 2a). The tag of the category was firstly input into each action video. The duration of action videos were about 1-4s, which can cover all the daily actions of the monkey. From each short video, through random sampling strategy with sparse temporal grouping, 8 frames representing each short video action were extracted. The corresponding optical flow picture and RGB flow information were input into the temporal shift (TS) and split attention (SA) (TSSA) network, respectively through the parts of RGB net and optical flow net for training. In each net module, resnet-50 is used as the backbone network, and the self-attention mechanism module of the group is added to the feature map position of each layer, and the feature map is divided into different cardinals. To increase the importance of certain cardinal and improve the performance of the entire network, a series of transformations are multiplied by different weights. Finally, the corresponding action category is output through softmax and onehot encoding. The evaluation index used here was top-1 and the final accuracy rate reached to 98.99% (RGB 89.83%, Flow 93.05%, RGB+Flow 98.99%) using two-stream neural network based on TSSA fusion.

## Code Usage Instructions

## Code
