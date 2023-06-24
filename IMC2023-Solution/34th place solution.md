# 34th place solution

Thank you to the organizers of the IMC 2023 Challenge and the Kaggle officials for their efforts. This challenge has been very helpful.

# Overview

Our code started with the submission-example baseline provided by the host. The pipeline has the following architecture:
![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8362662%2F6c55c1a4b581a462acef473aac6f8bff%2FMicrosoftTeams-image%20(7).png?generation=1686732896225750&alt=media)

# keep point

### Get image pair shortlist

we started looking for a model provided by the timm library in 'get_image_pair_shortlist'. We determined that this problem was similar to a classification problem, so we used the model 'tf_efficientnet_b8'. It was experimentally better than the default 'tf_efficientnet_b7'. And we conducted experiments by changing the sim_th parameter. We found that the performance was the best when sim_th was set to 0.6.

### Keypoint detect / matching

#### Model select

We only used 'KeyNetAffNetHardNet'. We spent a lot of time in the model selection process. We experimented with model ensembling and parameter tuning for 'LoFTR', 'DISK', 'KeyNetAffNetHardNet', 'KeyNetAffNetSoSNet', 'DKM', 'Silk', and 'RootSIFT'. However, in our personal experiments, 'KeyNetAffNetHardNet' performed the best.

#### Ensemble for multi-resolution

We examined datasets/scenes with various resolutions and conducted experiments with various resolutions. Among them, the optimal resolutions were 1696 and 1536. We also experimented with them as single-resolutions, but the performance was not good. We improved the performance by ensembling the results of applying 1696 resolution and 1536 resolution.

#### Find the optimal parameters

As we used 'KeyNetAffNetHardNet', we conducted experiments by adjusting numerous related parameters.
The final submitted parameters are as follows.

##### detect_features

***\*num_feats = 20000\****
***\*matching_alg = 'adalam' # smnn, adalam\****
***\*min_matches = 10\****

##### matche_features

***\*ransac_iters = 128\****
***\*search_expansion = 16\****

# The final score:

![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8362662%2Fea0805dba29a4ee979b9804bed6c9af2%2F15361696keynet.png?generation=1686734928396262&alt=media)

***\*Public LB: 0.451\****

***\*Private LB: 0.471\****

# Other things I tried

The pipeline has the following architecture:
![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8362662%2F829cc787a3904f3701fab10edffffbfe%2FMicrosoftTeams-image%20(6).png?generation=1686732818842259&alt=media)

# keep point

### Get image pair shortlist

we used the top-ranked model 'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384'. It was experimentally better than the efficientnet_b8.

### Keypoint detect / matching

#### High-resolution input image

First, we examined datasets/scenes with various resolutions. Among them, we confirmed that Cyprus and Wall have ultra-high-resolution images. We determined that if we reduce the image size too much during the resize stage, a lot of information would be lost in these ultra-high-resolution images. Therefore, we checked the GPU memory provided by Kaggle and resized the images to a manageable resolution. (We selected 1920, 1696, and 1848 depending on the experiment.)

#### Model select

We only used 'KeyNetAffNetHardNet'.

#### Ensemble for multi-resolution / single-resolution

Due to various factors, we designed the pipeline to distinguish architecture and applied multi-resolution for datasets/scenes containing input images with a resolution of 4K or higher to perform more keypoint detection and matches. On the other hand, we applied single-resolution for datasets/scenes with input images of FHD or lower because forcing upsampling of input images causes noise. Since applying multi-resolution can detect many noisy keypoints, we applied single-resolution.

Additionally, when we previously applied multi-resolution for all datasets/scenes and submitted, we were able to avoid run-timeout issues.

### Colmap / reconstruction

When performing submission with the above options, run-timeout often occurred. We thought it was due to the resolution being too high to resize. However, we thought it would be better to improve other parts such as feature matching accuracy and reconstruction for better results.
This idea was inspired by this discussion (https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/413551). For the train set, I believe that registering new images has a greater impact on performance than accuracy through GBA, so I tried to reduce the execution time by reducing the number of GBA iterations. Therefore, I set the options by doubling or tripling the GBA-related options.
`Mapper.ba_global_images_ratio = 1.1 * 3 (default: 1.1)`
`Mapper.ba_global_points_ratio = 1.1 * 3 (default: 1.1)`
`Mapper.ba_global_images_freq = 500 * 3 (default: 500)`
`Mapper.ba_global_points_freq = 250000 * 3 (default: 250000)`

# The final score:

urban / kyiv-puppet-theater (26 images, 325 pairs) -> mAA=0.896615, mAA_q=0.954154, mAA_t=0.896615

urban -> mAA=0.896615

heritage / dioscuri (174 images, 15051 pairs) -> mAA=0.609727, mAA_q=0.692964, mAA_t=0.614750

heritage / cyprus (30 images, 435 pairs) -> mAA=0.926207, mAA_q=0.976092, mAA_t=0.931494

heritage / wall (43 images, 903 pairs) -> mAA=0.767331, mAA_q=0.954153, mAA_t=0.771539

heritage -> mAA=0.767755

haiper / bike (15 images, 105 pairs) -> mAA=0.929524, mAA_q=1.000000, mAA_t=0.929524

haiper / chairs (16 images, 120 pairs) -> mAA=0.980000, mAA_q=1.000000, mAA_t=0.980000

haiper / fountain (23 images, 253 pairs) -> mAA=1.000000, mAA_q=1.000000, mAA_t=1.000000

haiper -> mAA=0.969841

**Final metric -> mAA=0.878070**

**Public LB: 0.448**

**Private LB: 0.461**

# Sad story

I won't say much. I didn't select the right submission, and ended up receiving a low score for the funny situation. Haha!!!
It's okay. This is also part of my skill.
![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8362662%2Fab00053f1ee12803e39d35388cd6b6de%2FUntitled%20(1).png?generation=1686728295436104&alt=media)

The difference between these two is as follows:

***\*mAA : 0.504\****

get_image_pair_shorlist model : ***\*convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384\****

***\*mAA : 0.471\****

get_image_pair_shorlist model : ***\*tf_efficientnet_b8\****