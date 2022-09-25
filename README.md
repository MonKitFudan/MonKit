# MonKit
**MonkeyMonitorKit (MonKit) is a toolbox for the automatic action recognition, posture estimation and identification of fine motor activities in the monkeys.**
![image](https://user-images.githubusercontent.com/58841760/192126392-9713bd37-77a7-4a9c-8c9a-218113ff776d.png)
Video-based action recognition is becoming a vital tool in clinical research and neuroscientific study for disorder detection and prediction. However, action recognition currently used in non-human primate (NHP) research relies heavily on intense manual labor and lacks standardized assessment. In this work, we established two standard benchmark datasets of NHPs in laboratory: MonkeyinLab (MiL) including 13 categories of actions and postures, and MiL2D with sequences of 2D skeleton features. Furthermore, based on recent advances in deep learning and skeleton visualization methods, a toolbox MonkeyMonitorKit (MonKit) is introduced for automatic action recognition, posture estimation and identification of fine motor activities in monkeys. By using datasets and MonKit, we evaluated daily behaviors of wildtype cynomolgus monkeys in home cages and experimental environments, and compared them with MECP2 gene mutant cynomolgus monkeys as a disease model of Rett syndrome (RTT). MonKit was used to assess the motor function, stereotyped behaviors and depressive phenotypes, with results compared with human manual detection. MonKit defines uniform standards for identifying behavior in NHPs with high accuracy and efficiency, thus providing a novel and comprehensive tool for monkey phenotypic behavior assessment.


**Keywords:**
Action recognition, Fine motor identification, Two stream deep model, 2D skeleton, Non-human primates, Rett syndrome.


## Dataset
### MiL Dataset
The below picture is examples of The MiL Dataset. Videos have been recognized as corresponding actions and postures (labels). (a-j) Ten categories of actions have been shown: (a) Climb, (b) Hang, (c) Turn, (d) Walk, (e) Shake, (f) Jump, (g) Move Down, (h) Lie Down, (i) Stand Up and (j) Sit Down. (k-m) Three categories of postures: (k) Stand, (l) Sit and (m) Huddle. Each row represents the non-contiguous frames randomly sampled in the corresponding video. The length of all videos is from 20 to 110 frames.
![MiL_dataset](/images/MiL_dataset.jpg)

### MiL2D Dataset
The below picture is illustration of MiL2D dataset with 15 keypoints of skeleton. (a) The detailed description of the definition and location of the 15 bone points. 0, right ankle; 1 right knee; 2, left knee; 3, left ankle; 4, hip; 5, tail; 6, chin; 7, head top; 8, right wrist; 9, right elbow; 10, right shoulder; 11, left shoulder; 12, left elbow; 13, left wrist; 14, neck.
![MiL2D_dataset](/images/MiL2D_dataset.jpg)
![MiL2D_dataset](https://github.com/MonKitFudan/MonKit/blob/main/images/MiL2D_dataset.jpg)

## Installation: How to install MonKit
The configuration MonKit documentation is divided into two parts. MonKit-Action recognition refers to [action_recognition_doc.md](https://github.com/MonKitFudan/MonKit/blob/main/action_recognition_doc.md). MonKit-Posture estimation refers to [posture_estimation_doc.md](https://github.com/MonKitFudan/MonKit/blob/main/posture_estimation_doc.md).
