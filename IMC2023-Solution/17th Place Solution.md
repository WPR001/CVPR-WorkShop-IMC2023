# 17th Place Solution

First, we would like to say thank you to the organizers and Kaggle staff for setting up this challenge, it has been an amazing experience for us.

## **Our solution**

For our final submission, we used the an emsemble of SuperPoint, KeyNet/AffNet/HardNet and SOSNet as our feature detectors/descriptors, and used SuperGlue and Adalam for feature matching. We used Colmap for reconstruction and camera localization. We also applied [hloc](https://github.com/cvg/Hierarchical-Localization) to speed up the pipeline and make it more scalable for validation and testing.

### **Image Retrieval**

We used [NetVLAD ](https://openaccess.thecvf.com/content_cvpr_2016/papers/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.pdf)as implemented in [hloc](https://github.com/cvg/Hierarchical-Localization) as the global feature descriptor for image retrieval

### **Feature Matching**

The first thing we noticed in the dataset was that some scenes contain a lot of rotated images, and we tried to tackle this problem with 2 approaches:

1) use rotation invariant feature matchers (e.g., KeyNet/AffNet/HardNet, SOSNet).
2) use a [lightweight orientation detector](https://github.com/ternaus/check_orientation) to detect the rotation angles and rotate the image pairs accordingly so that both images have a similar orientation (for simplicity, we only set the rotation angles to 90, 180 and 270 degrees).

We proceeded with both approaches and found out that both approaches achieve a similar improvement on the Heritage dataset, however, by ensembling more feature matchers, we observe some extra improvements on Urban and Haiper datasets, so we finally took this approach, and this ensemble achieved the best results for us within the time limit of 9h: **SuperGlue + KeyNet/AffNet/HardNet (with Adalam) + SOSNet (with Adalam)**. Using orientation compensation on the ensembled model does not bring any extra improvements.

### **Things that did not work**:

1) We first tried [this SOTA orientation detector](https://github.com/pidahbus/deep-image-orientation-angle-detection), however it consumes too much memory and could not be integrated into our pipeline on Kaggle
2) We found **KeyNet/AffNet/HardNet + Adalam** in the baseline the **best single feature matcher** without any preprocessing -- We could achieve 0.455/0.433 (equal to 47th place) by only tuning its parameters, however, when we integrated them into our pipeline using hloc, its performance dropped significantly to 0.334/0.277 (locally as well, mainly on urban), we tried to investigate but still do not know why.
3) We experimented on a lot of recent feature matchers and ensembles, including DKMv3, DISK, LoFTR, SiLK, DAC, and they either do not perform as well or are too slow when integrated into the pipeline. In general, we found that end-to-end dense matchers not well suited for this multiview challenge despite their success in last year's two view challenge, their speed is too slow and the scores they achieve are also not as good. Here are some local validation results:

1. SiLK (on ~800x600):
   urban: 0.125
   haiper: 0.165
2. DKMv3 (on ~800x600 and it's still very slow):
   heritage: 0.185
   haiper: 0.510
3. DISK (on ~1600x1200):
   urban: 0.461
   heritage: 0.292 (0.452 with rotation compensation)
   haiper: 0.433
4. SOSNet with Adalam (on ~1600x1200):
   urban: 0.031
   heritage: 0.460 (same with rotation compensation)
   haiper: 0.653
5. Sift / Rootsift with Adalam (on ~1600x1200):
   urban: 0.02
   heritage: 0.396
   haiper: 0.635
6. DAC: the results are very bad

### **Reconstruction**

After merging all the match points from the ensemble, we apply [geometric verification](https://github.com/colmap/pycolmap/blob/743a4ac305183f96d2a4cfce7c7f6418b31b8598/pipeline/match_features.cc#L76) in Colmap before reconstructing the model, which speeds up the reconstruction.

### **Things that did not work**:

1) We tried using Pixel-Perfect SFM, we set it up locally and it gave descent results visually comparable to our pipeline, but since we could not get it up running on Kaggle we did not proceed further.
2) We tried using MAGSAC++ to replace the default RANSAC function Colmap uses to remove bad matching points before reconstructing the model, but we did not see a significant difference in the final scores.