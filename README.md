# MonKit
**MonkeyMonitorKit (MonKit) is a toolbox for the automatic action recognition, posture estimation and identification of fine motor activities in the monkeys.**

![image](https://user-images.githubusercontent.com/58841760/192126392-9713bd37-77a7-4a9c-8c9a-218113ff776d.png)

Video-based action recognition is becoming a vital tool in clinical research and neuroscientific study for disorder detection and prediction. However, action recognition currently used in non-human primate (NHP) research relies heavily on intense manual labor and lacks standardized assessment. In this work, we established two standard benchmark datasets of NHPs in laboratory: MonkeyinLab (MiL) including 13 categories of actions and postures, and MiL2D with sequences of 2D skeleton features. Furthermore, based on recent advances in deep learning and skeleton visualization methods, a toolbox MonkeyMonitorKit (MonKit) is introduced for automatic action recognition, posture estimation and identification of fine motor activities in monkeys. By using datasets and MonKit, we evaluated daily behaviors of wildtype cynomolgus monkeys in home cages and experimental environments, and compared them with MECP2 gene mutant cynomolgus monkeys as a disease model of Rett syndrome (RTT). MonKit was used to assess the motor function, stereotyped behaviors and depressive phenotypes, with results compared with human manual detection. MonKit defines uniform standards for identifying behavior in NHPs with high accuracy and efficiency.


**Keywords:**
Action recognition, Fine motor identification, Two stream deep model, 2D skeleton, Non-human primates, Rett syndrome.


## Dataset
### MiL Dataset
The below picture is examples of The MiL Dataset. Videos have been recognized as corresponding actions and postures (labels). (a-j) Ten categories of actions have been shown: (a) Climb, (b) Hang, (c) Turn, (d) Walk, (e) Shake, (f) Jump, (g) Move Down, (h) Lie Down, (i) Stand Up and (j) Sit Down. (k-m) Three categories of postures: (k) Stand, (l) Sit and (m) Huddle. Each row represents the non-contiguous frames randomly sampled in the corresponding video. The length of all videos is from 20 to 110 frames.

<img width="451" alt="d7fc751d21c519265a99c89bc2373cd" src="https://user-images.githubusercontent.com/58841760/192126568-dffb7f1a-0110-473c-a25a-30e02040a69e.png">

### MiL2D Dataset
The below picture is illustration of MiL2D dataset with 15 keypoints of skeleton. (a) The detailed description of the definition and location of the 15 bone points. 0, right ankle; 1 right knee; 2, left knee; 3, left ankle; 4, hip; 5, tail; 6, chin; 7, head top; 8, right wrist; 9, right elbow; 10, right shoulder; 11, left shoulder; 12, left elbow; 13, left wrist; 14, neck.

<img width="382" alt="b10c7190b9f72e78ba3b3e4bbb9fa3a" src="https://user-images.githubusercontent.com/58841760/192126606-3faef41b-e790-45b0-af9c-77c80329c72c.png">

## Installation: How to install MonKit
The configuration MonKit documentation is divided into two parts. MonKit-Action recognition refers to [Mon_Action.md](https://github.com/MonKitFudan/MonKit/blob/main/Mon_Action/Mon_Action.md). MonKit-Posture estimation refers to [Mon_Pose.md](https://github.com/MonKitFudan/MonKit/blob/main/Mon_Pose/Mon_Pose.md). Monkey posture recognition and estimates of fine motor activities refers to [Mon_Fine_Motor_Identification.md](https://github.com/MonKitFudan/MonKit/blob/main/Mon_Fine_Motor_Identification/Mon_Fine_Motor_Identification.md). Monkey track refers to [Mom_Track.md](https://github.com/MonKitFudan/MonKit/blob/main/Mon_Track/Mon_Track.md).
