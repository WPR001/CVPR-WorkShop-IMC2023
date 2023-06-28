# [CVPR-Workshop-Image Matching Challenge 2023(Introduction and Sumarry)](https://image-matching-workshop.github.io/)

------

[TOC]

## CVPR2023

https://cvpr2023.thecvf.com/

## Workshop

https://image-matching-workshop.github.io/

### About

Matching two or more images across wide baselines is a core computer vision problem, with applications to stereo, 3D reconstruction, re-localization, SLAM, and retrieval, among many others. Until recently one of the last bastions of traditional handcrafted methods, they too have begun to be replaced with learned alternatives. Interestingly, these new solutions still rely heavily on design intuitions behind handcrafted methods. In short, we are clearly in a transition stage, and our workshop, held every year at CVPR since 2019, aims to address this, bringing together researchers across academia and industry to assess the true state of the field. We aim to establish what works, what doesn’t, what’s missing, and which research directions are most promising, while focusing on experimental validation.

Towards this end, every workshop edition has included an open challenge on local feature matching. Its results support our statement, as solutions have evolved from carefully tuned traditional baselines (e.g. SIFT keypoints with learned patch descriptors) to more modern solutions (e.g. transformers). Local features might have an expiration date, but true end-to-end solutions still seem far away. More importantly, the results of the Image Matching Challenges have shown that comprehensive benchmarking with downstream metrics is crucial to figure out how novel techniques compare with their traditional counterparts. Our ultimate goal is to understand the performance of algorithms in real-world scenarios, their failure modes, and how to address them, and to find out problems that emerge in practical settings but are sometimes ignored by academia. We believe that this effort provides a valuable feedback loop to the community.

Topics include (but are not limited to):

- Formulations of keypoint extraction and matching pipelines with deep networks.
- Application of geometric constraints into the training of deep networks.
- Leveraging additional cues such as semantics and mono-depth estimates.
- Methods addressing adversarial conditions where current methods fail (weather changes, day versus night, etc.).
- Attention mechanisms to match salient image regions.
- Integration of differentiable components into 3D reconstruction frameworks.
- Connecting local descriptors/image matching with global descriptors/image retrieval.
- Matching across different data modalities such as aerial versus ground.
- Large-scale evaluation of classical and modern methods for image matching, by means of our open challenge.
- New perception devices such as event-based cameras.
- Other topics related to image matching, structure from motion, mapping, and re-localization, such as privacy-preserving representations.



## Background & Basic knowledge

### Goal of the Competition

#### IMC2022

##### Description

For most of us, our best camera is part of the phone in our pocket. We may take a snap of a landmark, like the Trevi Fountain in Rome, and share it with friends. By itself, that photo is two-dimensional and only includes the perspective of our shooting location. Of course, a lot of people have taken photos of that fountain. Together, we may be able to create a more complete, three-dimensional view. What if machine learning could help better capture the richness of the world using the vast amounts of unstructured image collections freely available on the internet?

The process to reconstruct 3D objects and buildings from images is called Structure-from-Motion (SfM). Typically, these images are captured by skilled operators under controlled conditions, ensuring homogeneous, high-quality data. It is much more difficult to build 3D models from assorted images, given a wide variety of viewpoints, lighting and weather conditions, occlusions from people and vehicles, and even user-applied filters.

![img](https://storage.googleapis.com/kaggle-media/competitions/google-image-matching/trevi-canvas-licensed-nonoderivs.jpg)

The first part of the problem is to identify which parts of two images capture the same physical points of a scene, such as the corners of a window. This is typically achieved with local features (key locations in an image that can be reliably identified across different views). Local features contain short description vectors that capture the appearance around the point of interest. By comparing these descriptors, likely correspondences can be established between the pixel coordinates of image locations across two or more images. This “image registration” makes it possible to recover the 3D location of the point by triangulation.

![img](https://storage.googleapis.com/kaggle-media/competitions/google-image-matching/image3.gif)

Google employs Structure-from-Motion techniques across many Google Maps services, such as the 3D models created from StreetView and aerial imagery. In order to accelerate research into this topic, and better leverage the volume of data already publicly available, Google presents this competition in collaboration with the University of British Columbia and Czech Technical University.

In this code competition, you’ll create a machine learning algorithm that registers two images from different viewpoints. With access to a dataset of thousands of images to train and test your model, top-scoring notebooks will do so with the most accuracy.

If successful, you'll help solve this well-known problem in computer vision, making it possible to map the world with unstructured image collections. Your solutions will have applications in photography and cultural heritage preservation, along with Google Maps. Winners will also be invited to give a presentation as part of the Image Matching: Local Features and Beyond workshop at the Conference on Computer Vision and Pattern Recognition (CVPR) in June.



##### Evaluation

###### Evaluation metric

Participants are asked to estimate the relative pose of one image with respect to another. Submissions are evaluated on the **mean Average Accuracy (mAA)** of the estimated poses. Given a fundamental matrix and the hidden ground truth, we compute the error in terms of rotation (**𝜖<sub>𝑅</sub>**, in degrees) and translation (**𝜖<sub>𝑇</sub>**, in meters). Given one threshold over each, we *classify* a pose as accurate if it meets both thresholds. We do this over ten pairs of thresholds, one pair at a time (e.g. at 1<sup>𝑜</sup>  and 20 cm at the finest level, and 10<sup>𝑜</sup>  and 5 m at the coarsest level):

```python
thresholds_r = np.linspace(1, 10, 10)  # In degrees.
thresholds_t = np.geomspace(0.2, 5, 10)  # In meters.
```

We then calculate the percentage of image pairs that meet every pair of thresholds, and average the results over all thresholds, which rewards more accurate poses. As the dataset contains multiple scenes, which may have a different number of pairs, we compute this metric separately for each scene and average it afterwards. A python implementation of this metric is available on [this notebook](https://www.kaggle.com/code/eduardtrulls/imc2022-tutorial-load-and-evaluate-training-data).

###### Submission File

For each ID in the test set, you must predict the fundamental matrix between the two views. The file should contain a header and have the following format:

```python
sample_id,fundamental_matrix
a;b;c-d,0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
a;b;e-f,0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
a;b;g-h,0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
etc
```

Note that `fundamental_matrix` is a 3×33×3 matrix, flattened into a vector in row-major order.



##### Explore Data Analysis (EDA)

Data files [[link](https://www.kaggle.com/competitions/image-matching-challenge-2022/data)]

###### Dataset Description

Aligning photographs of the same scene is a problem of longstanding interest to computer vision researchers. Your challenge in this competition is to generate mappings between pairs of photos from various cities.

This competition uses a hidden test. When your submitted notebook is scored, the actual test data (including a sample submission) will be made available to your notebook.

###### Files

**train/\*/calibration.csv**

- `image_id`: The image filename.
- `camera_intrinsics`: The 3×3 calibration matrix **𝐊** for this image, flattened into a vector by row-major indexing.
- `rotation_matrix`: The 3×3 rotation matrix **𝐑** for this image, flattened into a vector by row-major indexing.
- `translation_vector`: The translation vector **𝐓**.

**train/\*/pair_covisibility.csv**

- `pair`: A string identifying a pair of images, encoded as two image filenames (without the extension) separated by a hyphen, as `key1-key2`, where `key1` > `key2`.
- `covisibility`: An estimate of the overlap between the two images. Higher numbers indicate greater overlap. We recommend using all pairs with a covisibility estimate of 0.1 or above. The procedure used to derive this number is described in Section 3.2 and Figure 5 of [this paper](https://arxiv.org/pdf/2003.01587.pdf).
- `fundamental_matrix`: The target column as derived from the calibration files. Please see the [problem definition page](https://www.kaggle.com/competitions/image-matching-challenge-2022/overview/problem-definition) for more details.

**train/scaling_factors.csv** The poses for each scene where reconstructed via [Structure-from-Motion](https://en.wikipedia.org/wiki/Structure_from_motion), and are only accurate up to a scaling factor. This file contains a scalar for each scene which can be used to convert them to meters. For code examples, please refer to [this notebook](https://www.kaggle.com/eduardtrulls/imc2022-tutorial-load-and-evaluate-training-data).

**train/\*/images/** A batch of images all taken near the same location.

**train/LICENSE.txt** Records of the specific source of and license for each image.

**sample_submission.csv** A valid sample submission.

- `sample_id`: The unique identifier for the image pair.
- `fundamental_matrix`: The target column. Please see the [problem definition page](https://www.kaggle.com/competitions/image-matching-challenge-2022/overview/problem-definition) for more details. The default values are randomly generated.

**test.csv** Expect to see roughly 10,000 pairs of images in the hidden test set.

- `sample_id`: The unique identifier for the image pair.
- `batch_id`: The batch ID.
- `image_[1/2]_id`: The filenames of each image in the pair.

**test_images** The test set. The test data comes from a different source than the train data and contains photos of mostly urban scenes with variable degrees of overlap. The two images forming a pair may have been collected months or years apart, but never less than 24 hours. Bridging this domain gap is part of the competition. The images have been resized so that the longest edge is around 800 pixels, may have different aspect ratios (including portrait and landscape), and are upright.



#### IMC2023

##### Description

The goal of this competition is to reconstruct accurate 3D maps. Last year's [Image Matching Challenge](https://www.kaggle.com/competitions/image-matching-challenge-2022/overview) focused on two-view matching. This year you will take one step further: your task will be to reconstruct the 3D scene from many different views.

Your work could be the key to unlocking mapping the world from assorted and noisy data sources, such as images uploaded by users to services like Google Maps.



##### Evaluation

###### Evaluation metric

Participants are asked to estimate the pose for each image in a set with 𝑁 images. Each camera pose is parameterized with a rotation matrix **𝐑** and a translation vector **𝐓**, from an arbitrary frame of reference.

Submissions are evaluated on the **mean Average Accuracy (mAA)** of the estimated poses. Given a set of cameras, parameterized by their rotation matrices and translation vectors, and the hidden ground truth, we compute the relative error in terms of rotation (**𝜖<sub>𝑅</sub>**, in degrees) and translation (**𝜖<sub>𝑇</sub>**, in meters) for every possible pair of images in 𝑁, that is, (𝑁 2) pairs.

We then threshold each of this poses by its accuracy in terms of both rotation, and translation. We do this over ten pairs of thresholds: e.g. at 1 degree and 20 cm at the finest level, and 10 degrees and 5 m at the coarsest level. The actual thresholds vary for each dataset, but they look like this:

```python
thresholds_r = np.linspace(1, 10, 10)  # In degrees.
thresholds_t = np.geomspace(0.2, 5, 10)  # In meters.
```

We then calculate the percentage of accurate samples (pairs of poses) at every thresholding level, and average the results over all thresholds. This rewards more accurate poses. Note that while you submit 𝑁, the metric will process all samples in (𝑁 2).

Finally, we compute this metric separately for each scene and then average it to compute its **mAA**. These values are then averaged over datasets, which contain a variable number of scenes, to obtain the final **mAA** metric.

###### Submission File

For each image ID in the test set, you must predict its pose. The file should contain a header and have the following format:

```python
image_path,dataset,scene,rotation_matrix,translation_vector
da1/sc1/images/im1.png,da1,sc1,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
da1/sc2/images/im2.png,da1,sc1,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
etc
```

The `rotation_matrix` (a 3×33×3 matrix) and `translation_vector` (a 3-D vector) are written as `;`-separated vectors. Matrices are flattened into vectors in row-major order. Note that this metric does not require the intrinsics (the calibration matrix **𝐊**), usually estimated along with **𝐑** and **𝐓** during the 3D reconstruction process.



##### Explore Data Analysis (EDA)

Data files [[link](https://www.kaggle.com/competitions/image-matching-challenge-2023/data)]

###### Dataset Description

Building a 3D model of a scene given an unstructured collection of images taken around it is a longstanding problem in computer vision research. Your challenge in this competition is to generate 3D reconstructions from image sets showing different types of scenes and accurately pose those images.

This competition uses a hidden test. When your submitted notebook is scored, the actual test data (including a sample submission) will be made available to your notebook. Expect to find roughly 1,100 images in the hidden test set. The number of images in a scene may vary from <10 to ~250.

Parts of the dataset (the Haiper subset) were created with the Captur3 app and the Haiper Research team from Haiper AI.

###### Files

**sample_submission.csv** A valid, randomly-generated sample submission with the following fields:

- `image_path`: The image filename, including the path.
- `dataset`: The unique identifier for the dataset.
- `scene`: The unique identifier for the scene.
- `rotation_matrix`: The first target column. A 3×3 matrix, flattened into a vector in row-major convection, with values separated by `;`.
- `translation_vector`: The second target column. A 3-D dimensional vector, with values separated by `;`.

**[train/test]/\*/\*/images** A batch of images all taken near the same location. Some of training datasets may also contain a folder named **images_full** with additional images.

**train/\*/\*/sfm** A 3D reconstruction for this batch of images, which can be opened with [colmap](https://colmap.github.io/), the 3D structure-from-motion library bundled with this competition.

**train/\*/\*/LICENSE.txt** The license for this dataset.

**train/train_labels.csv** A list of images in these datasets, with ground truth.

- `dataset`: The unique identifier for the dataset.
- `scene`: The unique identifier for the scene.
- `image_path`: The image filename, including the path.
- rotation_matrix: The first target column. A 3×33×3 matrix, flattened into a vector in row-major convection, with values separated by `;`.
- translation_vector: The second target column. A 3-D dimensional vector, with values separated by `;`.



### Basic knowledge

#### 3D computer vision 











![image-20220611155358501](https://github.com/WPR001/CVPR-WorkShop-IMC2023/assets/77914093/6702cd2e-be1d-4454-a683-b9cd6d74fbfe)





<img width="477" alt="截屏2023-06-24 13 41 47" src="https://github.com/WPR001/CVPR-WorkShop-IMC2023/assets/77914093/e5c12e88-250b-485d-89b2-74c0733c6f5a">
















## IMC2022 gold medal  solution

Our proposal for this competition is based on the fusion of matching results from four different models. **LoFTR, SuperGlue, DKM and QuadTreeAttention**. They all use publicly available pre-training weights and have not been trained or fine-tuned because we thought it would be less effective due to the different distribution of training and test data for this competition. After that, we enlarged the original image size to fit different models, and the resolution of each model in the final scheme was kept between 600 and 1600. Finally, we concat the keypoints of all model outputs and use cv2.USAC_MAGSAC to compute the F-matrix. In order to complete the four models in the 9 hours specified by kaggle, we used a ThreadPoolExecutor, multi-threaded inference keypoints and calculation of the F-matrix. In addition, we also tried Matchformerm model, different level flip data enhancement experiments, but did not work.

#### Model Ensemble

```python
def inf_models_ensemble(image_fpath_1, image_fpath_2):
    '''
    Input : image1 和 image2, 进行inference ensemble
    Return: match keypoints (匹配关键点)
    '''
    # loftr
    k_mkpts1, k_mkpts2 = loftr_inference(image_fpath_1, image_fpath_2, kornia_max_image_size) # kornia_max_image_size = 1024
    mkpts1_merge = k_mkpts1
    mkpts2_merge = k_mkpts2
    
    #  QuadTreeAttention
    qta_mkps1,qta_mkps2 = qta_inference(image_fpath_1, image_fpath_2, max_image_size=qta_max_img_size, divide_coef=32) # qta_max_img_size = 1024
    mkpts1_merge = np.concatenate((mkpts1_merge, qta_mkps1), axis=0)
    mkpts2_merge = np.concatenate((mkpts2_merge, qta_mkps2), axis=0)
    
    # SUPERGLUE
    sg_mkpts1, sg_mkpts2 = superglue_inference(image_fpath_1, image_fpath_2)
    mkpts1_merge = np.concatenate((mkpts1_merge, sg_mkpts1), axis=0)
    mkpts2_merge = np.concatenate((mkpts2_merge, sg_mkpts2), axis=0)


    # DKM
    dkm_mkpts1, dkm_mkpts2, _ = dkm_inference(image_fpath_1, image_fpath_2)
    if len(dkm_mkpts1) > 0:
        mkpts1_merge = np.concatenate((mkpts1_merge, dkm_mkpts1), axis=0)
        mkpts2_merge = np.concatenate((mkpts2_merge, dkm_mkpts2), axis=0)


    return mkpts1_merge, mkpts2_merge # get match keypoints
```

 

#### Calculating F matrix

```python
def calc_F_matrix(item):
    '''
    Input : sample_id, mkps1, mkps2 # 一对匹配关键点
    Return: F-matrix
    '''
    sample_id, mkps1, mkps2 = item 


    if len(mkps1) > 7: # 如果有足够的点, 则进行F-matrix计算
        F, _ = cv2.findFundamentalMat(mkps1, mkps2, cv2.USAC_MAGSAC, ransacReprojThreshold=0.25, confidence=0.999999, maxIters=120_000)  # 计算F-matrix
    else:   # 如果没有足够的点, 则返回空的F-matrix
        F = None   


    del mkps1, mkps2 # clean up
    return sample_id, F
```

 

#### History of the score 

1. Direct inference using LoFTR model, Public LB 0.726;
2. Parameter adjustment of LoFTR (resolution/number of keypoints /threshold, etc.), Public LB 0.767;
3. Use SuperGlue and adjust parameters. Public LB: 0.710;
4. Use QTA and adjust parameters, Public LB: 0.799;
5. Use DKM and adjust parameters, Public LB: 0.600;
6. The results of the four models were integrated, Public LB: 0.830+;
7. Conduct adjustment to cv2. findalmat. Public LB: 0.840+;

 

## Code, data set

- Code
  - IMC_Inference.ipynb
- Dataset
  - datasets.txt
  - othermodels/
  - pywheels/
  - super-glue-pretrained-network/



## IMC2023 bronze medal solution(public baseline)

### Public baseline:

https://www.kaggle.com/code/qi7axu/imc-2023-submission-example

### Ajust the hyper-parameters (bronze medal zone): 

**Public LB: 0.289**

**Private: LB: 0.351**

https://www.kaggle.com/code/allenwpr/public-baseline-private-lb0-351

### Code:

```python
if LOCAL_FEATURE != 'LoFTR':
                detect_features(img_fnames, 
                                8192,
                                feature_dir=feature_dir,
                                upright=True,
                                device=device,
                                resize_small_edge_to=800
                               )
```



## IMC2023 silver medal solution (my solution)
### Target

Calculate the rotation matrix **R** and the translation vector **T** for each camera (each image)

### Model

**Superpoint & Superglue**

We used the Superpoint to 

#### Superpoint

[Paper](https://arxiv.org/abs/1712.07629)

[Code](https://github.com/magicleap/SuperPointPretrainedNetwork)

#### Superglue

[Paper](https://arxiv.org/abs/1911.11763)

[Code](https://github.com/magicleap/SuperGluePretrainedNetwork)

### 3D Reconstruct (Colmap)





### LB change

Using this solution you can get the **private rank: 19 public rank: 14**

Hyper-parameter

```python
# public LB: 0.407
# private LB: 0.467

image_sizes = [1088, 1280]


number_of_exh = 8
thre_exh = 6

USE_ROI = False
ROI_SIZE = 1024

matching_config = {
    'superpoint': {
        'nms_radius': 6, 
        'keypoint_threshold': 0.008, 
        'max_keypoints': -1,
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 10, 
        'match_threshold': 0.27,
    }
}


# public LB: 0.408
# private LB: 0.476

image_sizes = [1088, 1280, 1312]

USE_ROI = False
ROI_SIZE = 1024

sim_th = None
n_matches = 100

num_exhaustives = 10
thresh_exhaustives = 6

matching_config = {
    'superpoint': {
        'nms_radius': 5,
        'keypoint_threshold': 0.008, 
        'max_keypoints': -1,
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 14,
        'match_threshold': 0.23,
    }
}


# public LB: 0.443
# private LB: 0.49

image_sizes = [1088, 1280, 1376]

USE_ROI = False
ROI_SIZE = 1024

sim_th = None
n_matches = 100

num_exhaustives = 10
thresh_exhaustives = 6

matching_config = {
    'superpoint': {
        'nms_radius': 5,
        'keypoint_threshold': 0.008, 
        'max_keypoints': -1,
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 14,
        'match_threshold': 0.23,
    }
}


# public LB: 0.448
# private LB: 0.492

image_sizes = [1088, 1280, 1376]

USE_ROI = False
ROI_SIZE = 1024

sim_th = None
n_matches = 100

num_exhaustives = 10
thresh_exhaustives = 6

matching_config = {
    'superpoint': {
        'nms_radius': 5,
        'keypoint_threshold': 0.008, 
        'max_keypoints': -1,
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.23,
    }
}


# public LB: 0.453
# private LB: 0.494

image_sizes = [1088, 1280, 1376]

USE_ROI = False
ROI_SIZE = 1024

sim_th = None
n_matches = 100

num_exhaustives = 10
thresh_exhaustives = 6

matching_config = {
    'superpoint': {
        'nms_radius': 5,
        'keypoint_threshold': 0.008, 
        'max_keypoints': -1,
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 25,
        'match_threshold': 0.23,
    }
}
```



### Local Validation

```python
# 我的solution大致mAA指标
## public LB: 0.407/0.408
## private LB: 0.476
*** METRICS ***
urban / kyiv-puppet-theater (26 images, 325 pairs) -> mAA=0.712000, mAA_q=0.854154, mAA_t=0.764000
urban -> mAA=0.712000

heritage / dioscuri (174 images, 15051 pairs) -> mAA=0.565976, mAA_q=0.655206, mAA_t=0.571384
heritage / cyprus (30 images, 435 pairs) -> mAA=0.021839, mAA_q=0.022989, mAA_t=0.027126
heritage / wall (43 images, 903 pairs) -> mAA=0.486379, mAA_q=0.878738, mAA_t=0.500775
heritage -> mAA=0.358065

haiper / bike (15 images, 105 pairs) -> mAA=0.941905, mAA_q=0.999048, mAA_t=0.941905
haiper / chairs (16 images, 120 pairs) -> mAA=0.817500, mAA_q=0.857500, mAA_t=0.825000
haiper / fountain (23 images, 253 pairs) -> mAA=0.999605, mAA_q=1.000000, mAA_t=0.999605
haiper -> mAA=0.919670

Final metric -> mAA=0.663245 (t: 2.6792633533477783 sec.)


## Public LB: 0.453
## private LB: 0.494
*** METRICS ***
urban / kyiv-puppet-theater (26 images, 325 pairs) -> mAA=0.720923, mAA_q=0.873538, mAA_t=0.771385
urban -> mAA=0.720923

heritage / dioscuri (174 images, 15051 pairs) -> mAA=0.562268, mAA_q=0.653432, mAA_t=0.567776
heritage / cyprus (30 images, 435 pairs) -> mAA=0.021839, mAA_q=0.022989, mAA_t=0.026897
heritage / wall (43 images, 903 pairs) -> mAA=0.432115, mAA_q=0.764120, mAA_t=0.437874
heritage -> mAA=0.338741

haiper / bike (15 images, 105 pairs) -> mAA=0.942857, mAA_q=0.999048, mAA_t=0.942857
haiper / chairs (16 images, 120 pairs) -> mAA=0.809167, mAA_q=0.856667, mAA_t=0.816667
haiper / fountain (23 images, 253 pairs) -> mAA=0.999605, mAA_q=1.000000, mAA_t=0.999605
haiper -> mAA=0.917210

Final metric -> mAA=0.658958 (t: 3.8252220153808594 sec.)
```



## IMC2023 gold medal solution

### [1st solution](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417407)

**Sparse + Dense matching, confidence-based merge, SfM, and then iterative refinement**

#### 0. Introduction

We are delighted to be participating in the image matching challenging 2023. Thanks to the organizers, sponsors, and Kaggle staff for their efforts, and congrats to all the participants. We learn a lot from this competition and other participants.

Our team members include Xingyi He, Dongli Tan, Sida Peng, Jiaming Sun, and Prof. Xiaowei Zhou. We are affiliated with the State Key Lab. of CAD&CG, Zhejiang University. I would like to express my gratitude to my teammates for their hard work and dedication.

#### 1. Overview and Motivation

![Fig.1](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14597895%2F2b2b6c045d8a2dfa0a536090f025db02%2Fmain_fig.png?generation=1686841291013288&alt=media)
We proposed a coarse-to-fine SfM framework to draw benefits from the recent success of detector-free matchers, while solving the multi-view inconsistency issue of detector-free matchers.
Due to the time limitation in the competition, we also incorporate the "light-weight" sparse feature detection and matching methods to determine image rotation and final overlap region between pairs, where the detector-free matcher will be performed upon.

However, caused by the multi-view inconsistency of detector-free matchers, directly using matches for SfM will lead to a significant number of 2D and 3D points. It is hard to construct feature tracks, and the incremental mapping phase will be extremely slow.

Our coarse-to-fine framework solves this issue by first quantizing matches with a confidence-guided merge approach, improving consistency while sacrificing the matching accuracy. We use the merged matches to reconstruct a coarse SfM model.
Then, we refine the coarse SfM model by a novel iterative refinement pipeline, which iterates between an attention-based multi-view matching module to refine feature tracks and a geometry refinement module to improve the reconstruction accuracy.

#### 2. Method

##### 2.1 Image Pair Construction

For each image, we select k relevant images using image retrieval method. Here we haven't found significant differences among different retrieval methods. This could potentially be attributed to the relatively small number of images or scenes in the evaluation dataset.

##### 2.2 Matching

###### 2.2.1 Rotation Detection

There are some scenes within the competition datasets which contain rotated images. Since many popular learning-based matching methods can not handle this case effectively, Our approach, similar to that of many other participants, involves rotating one of the query images several times[0, π/2, π, 3π/2] and matching it with the target image, respectively. This helps to mitigate the drastic reduction in the number of matching points caused by image rotations.

###### 2.2.2 Overlap Detection

Like last year's solution, estimating the overlap region is a commonly employed technique. We use the first round of matching to obtain the overlap region and then perform the second round of matching within them. According to the area ratio, we resize the smaller region in one image and align it with the larger region. We find a sparse matcher is capable of balancing efficiency and effectiveness.

###### 2.2.3 Matching

We find the ensemble of multiple methods tends to outperform any individual method. Due to time constraints, we choose the combination of one sparse method (SPSG) and one dense method (LoFTR). We also find that substitute LoFTR by DKMv3 performs better in this competition.

##### 2.3 Multi-view inconsistency problem

![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14597895%2Fab3bc20584b40131a4fe28b55d7d1530%2Fnon_repeatable_problem.png?generation=1686841396817545&alt=media)

As shown in Fig.2, the resulting feature locations of detector-free matchers (e.g., LoFTR) in an image depend on the other image. This pair-dependent nature leads to fragmentary feature tracks when running pair-wise matching over multiple views, which makes detector-free matchers not directly applicable to existing SfM systems (e.g., COLMAP).
Moreover, as for the sparse detection and matching part, since the cropped image overlap regions are also relevant to the other image, re-detecting keypoints on the cropped images for matching also shares the same multi-view inconsistency issue.
This issue is solved by the following coarse-to-fine SfM framework.

##### 2.4 Coarse SfM

In this phase, we first strive for consistency by merging to reconstruct an initial coarse SfM model, which will be further refined for higher pose accuracy in the refinement phase.

###### 2.4.1 Confidence-guided Merge

After the matching, we merge matches on each image based on confidence to improve the consistency (repeatability) of matches for SfM. For each image, we first aggregate all its matches with other images and then perform NMS with a window size of 5 to merge matches into points with the local highest confidence, as depicted in Fig.1(2). After the NMS, the number of 2D points can be significantly reduced, and the top 10000 points are selected for each image by sorting the confidence if the total point is still larger than the threshold.

###### 2.4.2 Mapping

Based on the merged matches, we perform the coarse SfM by COLMAP. Note that the geometry verification is skipped since RANSAC is performed in the matching phase. For the reconstruction of the scene with a large number of images (~250 in this competition), we enable the parallelized bundle adjustment (PBA) in COLMAP. Specifically, since PBA uses a PCG solver, which is an inexact solution to the BA problem and unlike the exact solution of Levenberg-Marquardt (LM) solver used by default in Ceres, we enable the PBA only after a large number of images are registered (i.e., >40). This is based on the intuition that the beginning of reconstruction is of critical importance, and the inexact solution of PBA may lead to a poor initialization of the scene.

##### 2.5 Iterative Refinement

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

#### 3. Ideas tried but not worked

##### 3.1 Other retrieval modules

Other than NetVLad, we have also tried the Cosplace, as well as using SIFT+NN as a lightweight detector and matcher for retrieval. However, there is no noticeable improvement, even performs slightly worse than NetVLad in our framework. We think this may be because the pair construction is at the very beginning of the overall pipeline, and our framework is pretty robust to the image pair variance.

##### 3.2 Other sparse detectors and matchers

Other than Superpoint + Superglue, we have also tried Silk + NN, which performs worse than Superpoint + Superglue. I think it may be because we did not successfully tune it to work in our framework.

##### 3.3 Other detector-free matchers

Other than LoFTR, we also tried Matchformer and AspanFormer in our framework. We find Matcherform performs on par with LoFTR but slower, which will lead to running out of time. AspanFormer performs worse than LoFTR when used in our framework in this challenge.

##### 3.4 Visual localization

We observe that there may image not successfully registered during mapping. Our idea is to "focus" on these images and regard them as a visual localization problem by trying to register them into the existing SfM model. We use a specifically trained version of LoFTR for localization, which can bring ~3% improvement on the provided training dataset. However, we did not have a spare running time quota in submission and, therefore, did not successfully evaluate visual localization in the final submission.

#### 4. Some insights

##### 4.1 About the randomness

We observe that the ransac performed with matching, the ransac PnP during mapping, and the bundle adjustment multi-threading in COLMAP may contain randomness.
After a careful evaluation, we find the ransac randomness seed in both matching and mapping is fixed. The randomness can be dispelled by setting the number of threads to 1 in COLMAP.
Therefore, our submission can achieve exactly the same results after multiple rerunning, which helps us to evaluate the performance of our framework.

##### 4.2 About the workload of the evaluation machine

Given that the randomness problem of our framework is fixed, we observe that the submission during the last week before the DDL is slower than (~20min) the previous submission with the same configuration.
Our final submission before the DDL using the DKM as a detector-free matcher has run out of time, which we believe may bring improvements, and we decided to choose it as one of our final submissions.
We rerun this submission version after the DDL, and it can be successfully finished within the time limit, which achieves 59.4 finally.
![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14597895%2F93c7812782a9455aaeafabdffec752b9%2Ffinal_shot_2.png?generation=1686842141320946&alt=media)

#### 5. Acknowledgment

The members of our team have participated in the IMC for three consecutive years(IMC 2021, 2022, and 2023), and we are glad to see there are more and more participants in this competition, and the number of submissions achieves a new high this year. We really enjoyed the competition this year since one of the most applications of feature matching is SfM. The organizers remove the limitation of the only matching submission as in IMC2021 but limit the running time and computation resources (a machine with only 2 CPU cores and 1 GPU is provided), which makes the competition more interesting, challenging, and flexible. Thanks to the organizers, sponsors, and Kaggle staff again!

#### 6. Suggestions

We also have some suggestions that we notice the scenes in this year's competition are mainly outdoor datasets. We think more types of scenes, such as indoor and object-level scenes with severe texture-poor regions, can be added to the competition in the future. In our recent research, we also collected a texture-poor SfM dataset which is object-centric with ground-truth annotations. We think it may be helpful for the future IMC competition, and we are glad to share it with the organizers if needed.

Special thanks to the authors of the following open-source software and papers: COLMAP, SuperPoint, SuperGlue, LoFTR, DKM, HLoc, pycolmap, Cosplace, NetVlad.

## Conclusion
