### 1st Place Solution: Sparse + Dense matching, confidence-based merge, SfM, and then iterative refinement

# 0. Introduction

We are delighted to be participating in the image matching challenging 2023. Thanks to the organizers, sponsors, and Kaggle staff for their efforts, and congrats to all the participants. We learn a lot from this competition and other participants.

Our team members include Xingyi He, Dongli Tan, Sida Peng, Jiaming Sun, and Prof. Xiaowei Zhou. We are affiliated with the State Key Lab. of CAD&CG, Zhejiang University. I would like to express my gratitude to my teammates for their hard work and dedication.

# 1. Overview and Motivation

![Fig.1](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14597895%2F2b2b6c045d8a2dfa0a536090f025db02%2Fmain_fig.png?generation=1686841291013288&alt=media)
We proposed a coarse-to-fine SfM framework to draw benefits from the recent success of detector-free matchers, while solving the multi-view inconsistency issue of detector-free matchers.
Due to the time limitation in the competition, we also incorporate the "light-weight" sparse feature detection and matching methods to determine image rotation and final overlap region between pairs, where the detector-free matcher will be performed upon.

However, caused by the multi-view inconsistency of detector-free matchers, directly using matches for SfM will lead to a significant number of 2D and 3D points. It is hard to construct feature tracks, and the incremental mapping phase will be extremely slow.

Our coarse-to-fine framework solves this issue by first quantizing matches with a confidence-guided merge approach, improving consistency while sacrificing the matching accuracy. We use the merged matches to reconstruct a coarse SfM model.
Then, we refine the coarse SfM model by a novel iterative refinement pipeline, which iterates between an attention-based multi-view matching module to refine feature tracks and a geometry refinement module to improve the reconstruction accuracy.

# 2. Method

## 2.1 Image Pair Construction

For each image, we select k relevant images using image retrieval method. Here we haven't found significant differences among different retrieval methods. This could potentially be attributed to the relatively small number of images or scenes in the evaluation dataset.

## 2.2 Matching

### 2.2.1 Rotation Detection

There are some scenes within the competition datasets which contain rotated images. Since many popular learning-based matching methods can not handle this case effectively, Our approach, similar to that of many other participants, involves rotating one of the query images several times[0, π/2, π, 3π/2] and matching it with the target image, respectively. This helps to mitigate the drastic reduction in the number of matching points caused by image rotations.

### 2.2.2 Overlap Detection

Like last year's solution, estimating the overlap region is a commonly employed technique. We use the first round of matching to obtain the overlap region and then perform the second round of matching within them. According to the area ratio, we resize the smaller region in one image and align it with the larger region. We find a sparse matcher is capable of balancing efficiency and effectiveness.

### 2.2.3 Matching

We find the ensemble of multiple methods tends to outperform any individual method. Due to time constraints, we choose the combination of one sparse method (SPSG) and one dense method (LoFTR). We also find that substitute LoFTR by DKMv3 performs better in this competition.

## 2.3 Multi-view inconsistency problem

![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14597895%2Fab3bc20584b40131a4fe28b55d7d1530%2Fnon_repeatable_problem.png?generation=1686841396817545&alt=media)

As shown in Fig.2, the resulting feature locations of detector-free matchers (e.g., LoFTR) in an image depend on the other image. This pair-dependent nature leads to fragmentary feature tracks when running pair-wise matching over multiple views, which makes detector-free matchers not directly applicable to existing SfM systems (e.g., COLMAP).
Moreover, as for the sparse detection and matching part, since the cropped image overlap regions are also relevant to the other image, re-detecting keypoints on the cropped images for matching also shares the same multi-view inconsistency issue.
This issue is solved by the following coarse-to-fine SfM framework.

## 2.4 Coarse SfM

In this phase, we first strive for consistency by merging to reconstruct an initial coarse SfM model, which will be further refined for higher pose accuracy in the refinement phase.

### 2.4.1 Confidence-guided Merge

After the matching, we merge matches on each image based on confidence to improve the consistency (repeatability) of matches for SfM. For each image, we first aggregate all its matches with other images and then perform NMS with a window size of 5 to merge matches into points with the local highest confidence, as depicted in Fig.1(2). After the NMS, the number of 2D points can be significantly reduced, and the top 10000 points are selected for each image by sorting the confidence if the total point is still larger than the threshold.

### 2.4.2 Mapping

Based on the merged matches, we perform the coarse SfM by COLMAP. Note that the geometry verification is skipped since RANSAC is performed in the matching phase. For the reconstruction of the scene with a large number of images (~250 in this competition), we enable the parallelized bundle adjustment (PBA) in COLMAP. Specifically, since PBA uses a PCG solver, which is an inexact solution to the BA problem and unlike the exact solution of Levenberg-Marquardt (LM) solver used by default in Ceres, we enable the PBA only after a large number of images are registered (i.e., >40). This is based on the intuition that the beginning of reconstruction is of critical importance, and the inexact solution of PBA may lead to a poor initialization of the scene.

## 2.5 Iterative Refinement

We proceed to refine the initial SfM model to obtain improved camera poses and point clouds. To this end, we propose an iterative refinement pipeline. Within each iteration, we first enhance the accuracy of feature tracks with a transformer-based multi-view refinement matching module.
These refined feature tracks are then fed into a geometry refinement phase which optimizes camera poses and point clouds jointly. The geometry refinement iterates between the geometric-BA and track topology adjustment (including complete tracks, merge tracks, and filter observations). The refinement process can be performed multiple times for higher accuracy.
Our feature track refinement matching module is trained on the MegaDepth, and more details are in our paper which is soon available on arXiv.

| Method            | score(private) |
| ----------------- | -------------- |
| spsg              | 0.482          |
| spsg+LoFTR        | 0.526          |
| spsg+LoFTR+refine | 0.570          |
| spsg+DKM^+refine  | 0.594          |

^only replace LoFTR in the Haiper dataset

# 3. Ideas tried but not worked

## 3.1 Other retrieval modules

Other than NetVLad, we have also tried the Cosplace, as well as using SIFT+NN as a lightweight detector and matcher for retrieval. However, there is no noticeable improvement, even performs slightly worse than NetVLad in our framework. We think this may be because the pair construction is at the very beginning of the overall pipeline, and our framework is pretty robust to the image pair variance.

## 3.2 Other sparse detectors and matchers

Other than Superpoint + Superglue, we have also tried Silk + NN, which performs worse than Superpoint + Superglue. I think it may be because we did not successfully tune it to work in our framework.

## 3.3 Other detector-free matchers

Other than LoFTR, we also tried Matchformer and AspanFormer in our framework. We find Matcherform performs on par with LoFTR but slower, which will lead to running out of time. AspanFormer performs worse than LoFTR when used in our framework in this challenge.

## 3.4 Visual localization

We observe that there may image not successfully registered during mapping. Our idea is to "focus" on these images and regard them as a visual localization problem by trying to register them into the existing SfM model. We use a specifically trained version of LoFTR for localization, which can bring ~3% improvement on the provided training dataset. However, we did not have a spare running time quota in submission and, therefore, did not successfully evaluate visual localization in the final submission.

# 4. Some insights

## 4.1 About the randomness

We observe that the ransac performed with matching, the ransac PnP during mapping, and the bundle adjustment multi-threading in COLMAP may contain randomness.
After a careful evaluation, we find the ransac randomness seed in both matching and mapping is fixed. The randomness can be dispelled by setting the number of threads to 1 in COLMAP.
Therefore, our submission can achieve exactly the same results after multiple rerunning, which helps us to evaluate the performance of our framework.

## 4.2 About the workload of the evaluation machine

Given that the randomness problem of our framework is fixed, we observe that the submission during the last week before the DDL is slower than (~20min) the previous submission with the same configuration.
Our final submission before the DDL using the DKM as a detector-free matcher has run out of time, which we believe may bring improvements, and we decided to choose it as one of our final submissions.
We rerun this submission version after the DDL, and it can be successfully finished within the time limit, which achieves 59.4 finally.
![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14597895%2F93c7812782a9455aaeafabdffec752b9%2Ffinal_shot_2.png?generation=1686842141320946&alt=media)

# 5. Acknowledgment

The members of our team have participated in the IMC for three consecutive years(IMC 2021, 2022, and 2023), and we are glad to see there are more and more participants in this competition, and the number of submissions achieves a new high this year. We really enjoyed the competition this year since one of the most applications of feature matching is SfM. The organizers remove the limitation of the only matching submission as in IMC2021 but limit the running time and computation resources (a machine with only 2 CPU cores and 1 GPU is provided), which makes the competition more interesting, challenging, and flexible. Thanks to the organizers, sponsors, and Kaggle staff again!

# 6. Suggestions

We also have some suggestions that we notice the scenes in this year's competition are mainly outdoor datasets. We think more types of scenes, such as indoor and object-level scenes with severe texture-poor regions, can be added to the competition in the future. In our recent research, we also collected a texture-poor SfM dataset which is object-centric with ground-truth annotations. We think it may be helpful for the future IMC competition, and we are glad to share it with the organizers if needed.

Special thanks to the authors of the following open-source software and papers: COLMAP, SuperPoint, SuperGlue, LoFTR, DKM, HLoc, pycolmap, Cosplace, NetVlad.