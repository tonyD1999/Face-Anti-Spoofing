# Face Anti Spoofing
> Focus on Printed Attack
> Single Modal
> If you need pickle files or have any questions, you can contact me via email: duc.nguyensy1999@gmail.com
## Dataset 

**NUAA (testing)**
* Each individual’s face image is collected in three separate sessions (raw, detected face, normalized face) over approximately two weeks, with the environ-mental and lighting conditions varying in each session. 
* There is a problem with this dataset in that its images are divided into individuals.We have to rearrange those images into video IDs.

![](https://i.imgur.com/uPXtkyd.png)

Link for dataset: [NUAA](http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html)

**RoseYoutu (training)**
* The  dataset  being  used  for  training  in  this  project  is  Rose-Youtu, which contains the real videos and their corresponding fake videos on many types of cameras.
* We use printed paper attacks and video replay attacks on the videos cap-tured from the iPhone 5s camera. Thus, we have prepared a set of 17243 bonafide and 29740 fake images.

Link for dataset: [RoseYoutu](https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/)
## Implementation
* SimpleNet

![](https://i.imgur.com/pxSPXcO.png)

* Architecture

![](https://i.imgur.com/Nou3BBS.png)

* We decided not to train raw RGB images and substitute them with artificial modalities which include more helpful information for the spoofing detection task.

* We select 20 images uniformly from the sequence of frames in the video.Then we transform those images into artificial modalities:
    * Optical Flow:
        * the first and fifth images
        * the first and tenth images
        * the first and fifteenth images
        * the first and last images
    * RankPooling of the sequence of selected images

## Models

**Optical Flow model (OF)**,

* The motivation behind this is to show that flows for the real track will change if we select images at a different time, while the flow for the fake track should remain approximately the same. 
* This model detects spoof faces using face areas so that we use Dlib to crop faces.

**Optical FLow with PRNet model (OF-PRNet)**

* We use PRNet to subtract the background from the face frames. After getting the depth map of the faces from PRNet, we used it as a mask to get face pixels.

![](https://i.imgur.com/n3VenIo.png)

**Optical Flow with Fixed Bounding box (OF-FBB)** 
* This model has two versions, one is trained on printed and replay attack videos, and another (OF-FBB#) is trained on printed attack data, and there is no data preprocessing. 
* ***The following models are also trained on printed attack data. The reason why we will explain in the result***.
* Instead of calculating the Optical Flow with the face bounding box, we calculate the Optical Flow with the Fixed Bounding box by detecting the face in the first images to create the box, which is later used to crop images in the sequences.

![](https://i.imgur.com/Ypn3e6Q.png)

**Optical Flow with Fixed Bounding box and MSRCR model (OF-FBB-MSRCR)**

* Multi-Scale Retinex with Color Restoration (MSRCR) is an algorithm that enhances image quality taken under a wide range of nonlinear illumination conditions to the level that a user would have perceived it in real-time. 
* The same as the model above, but the difference is that we convert images to MSRCR images before calculating the Optical Flow

* With the help of MSRCR, we found a considerable difference between fake and real images.

![](https://i.imgur.com/soe4cLP.png)

**Optical Flow with RankPooling and PRNet Model (OF-RP-PRNet)** 

* We use RankPooling to encode action, which is the face movement in the videos. We use the same data preprocessing in Optical Flow with the PRNet model.

* TO calculate RankPooling we use Support Vector Regression 
    * C = 1: still contains some face features
    * C = 1000: losing face features but capturing the changes in the video

* In this model, we calculate different artificial modalities: 
    * RankPooling(C=1), (C=1000) on sequence of 20 selected images. 
    * Optical FLow of the first and tenth images, the first and last images

* We also apply VisionLab architecture to our model to make comparisons.

![](https://i.imgur.com/boPafxv.png)

## Result

![](https://i.imgur.com/Jh4O832.png)

* We have already implemented 11 models. For convenience, we denote that the models using VisionLab architecture will have an asterisk (*) in their name.  The first six models are trained on both printed and replay attack videos,  and the other models are trained on printed attack videos. 
* Each method produces a different number of modalities.
* In the testing step, we applythe same pipeline to the NUAA dataset.
* After training and testing, we draw some conclusions:
    * Firstly, our goal is to try to detect printed attacks. However, models training on replay attack videos make a lousy effort in predicting printed attacks. Those models have lower accuracy than other models trained only on printed attack data on testing dataset NUAA. The testing accuracyon  NUAA  of  the  **OF-FFB#**  model  is  higher  than  the  **OF-FFB**  model,  **71.55%**  compared  to **69.95%**.
    * Secondly, Optical Flow calculating on faces bounding boxing misses essential features such as the face movement compared to the background. Consequently, **Optical Flow with fixed bounding box** (model *3, 6, 7*) has better performance than **Optical FLow with face bounding box** (model *1, 2, 4, 5*).
    * Finally, our architecture brings better performance than VisionLab architecture. The best model is **OF-FFB-MSRCR**, with an accuracy of **74.73%** on NUAA.
    * Moreover, we found that artificial modalities have a drawback - when we selected optical flow, we implied that real subjects will change their mimics through time, which is not always true.

Link for useful files and pretrained model: [pretrained model](https://drive.google.com/drive/folders/1CZUj2Kb_g-lA9B3nllcdu7TuXKoGzAFo?usp=sharing)
## References

* Ana Belén Petro, Catalina Sbert, and Jean-Michel Morel, Multiscale Retinex, Image Processing On Line,  (2014), pp. 71–88.
* Aleksandr Parkin and Oleg Grinchuk.Creating Artificial Modalities to Solve RGB Liveness.arXiv,2006.16028, 2020
* Basura Fernando, Efstratios Gavves, Jose Oramas, Amir Ghodrati, and Tinne Tuytelaars. Rank pooling for action recognition. TPAMI, 39(4):773787, 2017.
* Yao Feng, Fan Wu, Xiaohu Shao, Yanfeng Wang, and XiZhou. Joint 3d face reconstruction and dense alignment withposition map regression network. In ECCV, 2018




