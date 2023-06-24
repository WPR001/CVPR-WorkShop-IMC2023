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





## IMC2023 bronze medal solution(public baseline)



## IMC2023 silver medal solution



## IMC2023 gold medal solution



## Conclusion
