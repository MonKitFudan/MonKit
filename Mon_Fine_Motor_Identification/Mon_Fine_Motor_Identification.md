# Monkey Fine Motor Identification
**Three postures (huddle, sit and stand) were detected in the daily life of the monkeys and predicted by using the accurate height information of bone points.  The y-axis coordinates of four skeleton points (0, right ankle;  3, left ankle;  4, hip;  7, head top) obtained by the HRNet network was collected and the (*ymax - ymin*) was calculated for height information (Figure A), since the monkey’s tail acts as interference in this calculation.  The three groups of monkeys (WT in cage, WT in test and *MECP2* mutant) showed no significant difference in the time spent in postures huddle, sit or stand (Figure B-D).**

## Posture Recognition and Estimates of Fine Motor Activities 
Based on the MonKit dataset and keypoints prediction, we also detected fine motor activities of stereotyped pattern and head-down behaviors with a relatively small motion amplitude. Stereotyped pattern behaviors mainly refer to repetitive and purposeless body movements at a fixed frequency, which are mostly observed in the patients with Rett syndrome as a feature of autism. In monkeys, the stereotyped behaviors include turning over, circling, pacing left and right, and shaking the cage. Here, we provide results of bone point recognition obtained by HRNet for the estimation of stereotyped behaviors in monkeys. The sum of the 15 bone points with their vector direction was used for the coordinates of center point in the monkeys.Three out of 5 ***MECP2*** mutant monkeys showed over 20% time spent in stereotyped behavior, while 4 WT monkeys also performed stereotyped behaviors in the test cage, suggesting the stereotyped pattern behaviors may also represent anxiety status related to more activities in monkeys (Figure F). 
Depressive behavior has also been identified in a sizeable minority of female patients with RTT syndrome. Besides the posture of huddle, a foetal-like, self-enclosed posture, with the head at or below the shoulders during the awake state has been also considered as measure of depression-like behavior in monkeys. We detected the head down behavior here by using the y axis coordinate of the bone point of neck and chin in the monkey during the category of low activity. In the MonKit detection, the time of Head down posture was not significantly increased in the ***MECP2*** mutant monkeys nor in the group of WT in test compared to WT in cage (Figure H).

![image](https://user-images.githubusercontent.com/58841760/192136875-01854fb6-04cb-4645-9262-47c4875a4035.png)

Huddle, sit, stand, stereotyped pattern behavior and head down posture detected by MonKit in WT in cage, WT in test and MECP2 mutant monkeys. (A) Diagram of height calculation for the postures of huddle, sit and stand. (B-D) Statistics of time spent in the category of Huddle (B), Sit (C) and Stand (D). Each dot represents an individual monkey with averaged time spent in each video clip. (E) A series of representative images of stereotyped behavior of pacing left and right. (F) Statistics of the stereotyped behavior in the groups of WT in cage, WT in test and MECP2 mutant monkeys. (G) Representative images of normal and head down behavior and the diagram of chin and neck bone points. (H) Statistics of the head down posture in the groups of WT in cage, WT in test and ***MECP2*** mutant monkeys. Each dot represents an individual monkey with averaged time spent in each video clip.

This study is based on TSM: Temporal Shift Module for Efficient Video Understanding.(***Lin, J., Gan, C., & Han, S. (2019). TSM: Temporal shift module for efficient video understanding. In Proceedings of the IEEE/CVF International Conference on  Computer Vision.***).

## Installation
### Experimental setup
All experiments were implemented in PyTorch 1.7.1 for deep models with Nvidia Quadro RTX8000 GPUs (Memory: 48GB). 
The CPU is Intel Xeon Gold 5220R (2.2GHz, 24 Cores). The OS is Ubuntu 18.04.


### Environment configuration


## Code Usage


### Train

