# Monkey Pose

**Based on MiL dataset, we further established a MiL2D dataset of images with 2D skeleton and key bone points. The MiL2D dataset consists of 15,175 annotated images that span a large variation of poses and positions seen in 13 categories of MiL dataset. Fifteen keypoints of skeleton were marked in details (Figure 4a). MaskTrack R-CNN 28 was used to track monkey’s position (Figure 4b). The dataset includes diverse configurations of cage environments, and the monkeys with the corresponding skeleton points were detected clearly in the MiL2D dataset (Figure 4c).**

![image](https://user-images.githubusercontent.com/58841760/192129066-447be3f9-87a0-45f7-897b-a66d8b5263a2.png)

## MonkeyMonitorKit (MonKit) toolbox and keypoint prediction
By using the MiL2D dataset, the monkey bone recognition task was performed for training and testing process as a toolbox of MonkeyMonitorKit (MonKit) based on high resolution network (HRNet) 29 (Figure 5a). The images from original input video were processed to 256× 340 pixels. MaskTrack R-CNN was used to track monkey’s position (Figure 5b). The rectangle information of the monkey’s position was intercepted and input to the HRNet network to obtain the heatmap (Figure 5c). The mean square error (MSE) loss function was used to compare the target to get loss (Figure 5d). Finally, the 15 bone points were transformed into x and y space coordinates (Figure 5e). The accuracy has reached 98.8% accuracy in the MiL2D dataset and OpenMonkeyStudio dataset.

![image](https://user-images.githubusercontent.com/58841760/192129000-0700959c-5a09-4cab-bd06-c954014b761b.png)

##
