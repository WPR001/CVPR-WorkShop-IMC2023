# 4th place solution

Firstly, I'd like to express my gratitude to the hosts and Kaggle staff for conducting the IMC 2023 competition. The task was both exciting and challenging, which made it a pleasure to engage with over the two months.

## Overview

SuperPoint/SuperGlue proved to be exceptionally accurate and quick.
My code is partly a combination of the baseline provided by the host and [the notebook by ](https://www.kaggle.com/code/chankhavu/loftr-superglue-dkm-with-inspiration)[@chankhavu](https://www.kaggle.com/chankhavu) at IMC 2022. I extend my appreciation to both for providing the codes.

## Main pipeline

### Screening process based on the number of matches

Considering the large number of image combinations, an effective screening method was necessary. I noticed that the number of matches made by SG is significantly low (<10) when an image pair is unsuitable for stereo matching. Consequently, I decided to bypass the process if the number of matches achieved by SG (longside = 1200) fell below a certain threshold (in this case, 30 matches). This strategy significantly reduced processing time, allowing for more pair trials within the given timeframe, leading to a noticeable improvement (LB: +0.08).

### Rotation during the screening process

Procuring meaningful matches from pairs with unsorted image orientations, such as those found in Cyprus, proved to be challenging. Therefore, I incorporated a rotation process into the screening procedure, resulting in further improvement (LB: +0.04).

### Image splitting

Each image was divided into four sections, each generating its own set of keypoints, followed by the execution of matchings across all pair combinations (4x4 = 16 pairs) with a batched process for SP/SG (longside = 1400). This method proved to be more effective and time-efficient than traditional TTA in my case (LB: +0.01~0.02).

### Ensembling with DKM

After comparing various models, DKM v3 emerged as a relatively lightweight and effective choice when used in conjunction with SG (LB: +0.01).

### Parallel execution of separate matching and mapping processes

Both matching and mapping/reconstruction are time-intensive tasks. However, the former utilizes both the GPU and a single CPU, while the latter only requires CPU resources. Therefore, implementing parallel processing using the queue library improved time efficiency by approximately 20~30%. This concept was inspired by the first-place solution at IMC 2022.
Remember to set `mapper_options.num_threads = 1`, which can also help avoid OOM during reconstruction.

## The final score:

urban / kyiv-puppet-theater (26 images, 325 pairs) -> mAA=0.921538, mAA_q=0.987385, mAA_t=0.921538
urban -> mAA=0.921538

heritage / dioscuri (174 images, 15051 pairs) -> mAA=0.594950, mAA_q=0.689662, mAA_t=0.602279
heritage / cyprus (30 images, 435 pairs) -> mAA=0.706437, mAA_q=0.727126, mAA_t=0.724828
heritage / wall (43 images, 903 pairs) -> mAA=0.805980, mAA_q=0.935105, mAA_t=0.824917
heritage -> mAA=0.702456

haiper / bike (15 images, 105 pairs) -> mAA=0.933333, mAA_q=0.999048, mAA_t=0.933333
haiper / chairs (16 images, 120 pairs) -> mAA=0.981667, mAA_q=0.999167, mAA_t=0.981667
haiper / fountain (23 images, 253 pairs) -> mAA=0.999605, mAA_q=1.000000, mAA_t=0.999605
haiper -> mAA=0.971535

**Final metric -> mAA=0.865176**

**Public LB: 0.471**
**Private LB: 0.534**

It should be noted that the submission showed best the highest local and public score, and also resulted in the best private score among my submissions. While I struggled with the randomness of Colmap, I now recognize that the dataset was useful and served as a valuable reference in aiming for the correct goal.

## Ideas that did not work well

- Other models such as LoFTR, SE2-LoFTR, disk, QuadTreeAttention, OpenGlue, and SILK were tested. It was revealed that the combination of SP/SG and DKM consistently outperformed them in terms of both speed and performance.
  Employing USAC_MAGSAC prior to reconstruction occasionally shortened the reconstruction duration, but the effect was minor with the parallel execution, where the matching process is rate-determining. Also, it never improved the score in my case.
- Implementing CLAHE rendered my pipeline unstable and less robust. While it proved effective in some scenes, it often deteriorated the accuracy in others, overall often leading to a decrease in score.
- Other forms of TTA (resolution, flip, 10deg-rotation, crop) provided only minimal improvement while consuming significant time. It appeared more beneficial to experiment with numerous pairs than to utilize TTA.
- I attempted to determine the R and T for each image that could not be registered with Colmap, employing the same methodology as IMC2022. However, this approach failed to improve the score. The reason seems straightforward: with either method, if matches cannot be identified, there is little that can be done. (On the other hand, adding the R and T values obtained from reconstructions other than best_idx to the submission also slightly improved the score.)