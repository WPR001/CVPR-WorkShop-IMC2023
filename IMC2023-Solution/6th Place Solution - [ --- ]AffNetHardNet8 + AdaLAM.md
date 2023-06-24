# 6th Place Solution - [ --- ]AffNetHardNet8 + AdaLAM

I would like to express my thanks to the competition organizers and Kaggle staff for hosting this amazing competition. I would also like to thank the competition hosts (@oldufo , [@eduardtrulls](https://www.kaggle.com/eduardtrulls)) for providing helpful materials and a great example submission, which allowed me to quickly get up to speed with the competition.
Participating in this competition and IMC2022 has taught me a lot about image matching, Kornia, and SfM.

# 1. Overview

My final solution was based on the submission example, but I pushed the limits of KeyNetAffNetHardNet + AdaLAM matcher with Colmap.

- I implemented four local feature detectors that are similar to KeynetAffnetHardnet ([Example from Kornia](https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.GFTTAffNetHardNet)), and each detector extracted local features for all the images (with limited size) in a scene. See more info about other non-learning based keypoint detectors [here](https://kornia.readthedocs.io/en/latest/feature.html).
- Using HardNet8 rather than HardNet although I didn't see a performance difference.
- AdaLAM matcher was used to match all the pairs and get the matches for the scene. Only pair with an average matching distance < 0.5 were kept. 'force_seed_mnn' was set to True and 'ransac_iters' was increased to 256 compared with the submission example.
- The matches from the four detectors were then combined.
- I used USAC_MAGSAC to get the fundamental matrix for each pair and only kept the inlier matches. The results are written to the database as two-view geometry. `incremental_mapping` is then performed without `match_exhaustive`.

![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3964695%2Fbb53c3f1e7ebad9394a673a421b47d11%2Fdetector.png?generation=1686695228576401&alt=media)
![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3964695%2F051e9413d0ce9cd8357723ba7efab1be%2Ffull_pipeline.png?generation=1686695522836099&alt=media)

## 1.1. How I reached this solution

After running the submission example a few times, I thought that the key to finding a good solution was to identify a good shortlist of image pairs and apply similar solutions from IMC2022. I noticed that the KeyNetAffNetHardNet solution only needed to run the slower local feature detection once for each image, and the matching using AdaLAM matcher was fast enough to match each pair. Therefore, I decided to use KeyNetAffNetHardNet and the matching distance of AdaLAM to find the matching/pairing shortlist. However, with some optimizations, it ended up being the final solution in the last week.

## 1.2. KeyNetAffNetHardNet

I was surprised by the performance of KeyNetAffNetHardNet after using AdaLAM to match all possible pairs. With an increased number of features to 8000 and a maximum longer edge of 1600, I was able to achieve a score of 0.414/0.457 (Public/Private).

During my experimentation, I discovered that the matching distance can be used to determine whether two images have overlapping areas or not. See my [Test Notebook](https://www.kaggle.com/code/maxchen303/imc2023-test-notes) for some experiments using KeyNetAffNetHardNet + Adalam for pairing.
Disabling the Upright option (enabling OriNet) can make the Adalam matching more robust in handling rotated images.

# 2. Implementation details with Colmap

## 2.1. Using focal length

In the submission example, there is a section that extracts the focal length from image exif. However, the "FocalLengthIn35mmFilm" property may not be found using the `image.get_exif()` method. In some cases, the focal length information may exist in the `exif_ifd` (as described in this [Stack Overflow post](https://stackoverflow.com/questions/68033479/how-to-show-all-the-metadata-about-images)).
To extract the focal length from the `exif_ifd`, I used the following code:

```python
exif = image.getexif()
exif_ifd = exif.get_ifd(0x8769)
exif.update(exif_ifd)
```

If the focal length is found in the exif, I also set the "prior_focal_length" flag to true when adding the camera to the database. This improved the mAA for some scenes (mAA of urban / kyiv-puppet-theater 0.764 -> 0.812).

According to the [Colmap tutorial](https://colmap.github.io/tutorial.html#database-management):

> By setting the prior_focal_length flag to 0 or 1, you can give a hint whether the reconstruction algorithm should trust the focal length value.

## 2.2. Handling Randomness: Bypass match_exhaustive()

After testing my solution many times on the training dataset, I noticed that the evaluation results were random even when using the same code. I found that the source of this randomness was the `match_exhaustive()` function, which runs RANSAC on the matches and generates [two-view geometry](https://github.com/colmap/colmap/blob/dev/src/colmap/estimators/two_view_geometry.h) for each image pair in the database.

To address this issue, I decided to bypass the `match_exhaustive` function and use USAC_MAGSAC from OpenCV to find the fundamental matrix for each pair. I then wrote the two_view_geometry directly into the database. With this change, I was able to get deterministic mAAs when running the same code. By tuning the USAC_MAGSAC parameters, I was able to ensure the best score for reconstructing the same matches.

## 2.3. Multiprocessing

I discovered that adding matches from other AffNetHardNet detectors could increase the overall mAA score. To fit more similar detectors into the process, I used multiprocessing to ensure that the second CPU core was fully utilized. The final solution takes about 7.5 hours to run using four detectors with edges no longer than 1600.

To optimize the process, I sorted all the scenes by the number of images before processing, from greater to smaller. For each scene, there were two stages: matching to generate the database and reconstruction from the database. The reconstruction of different scenes is allowed to run in parallel with the matchings. Ideally, the pipeline should look like the following:
![img](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3964695%2Fc9cbe4d2f30c6874e597dd5949c2b62e%2Fmultiprocessing.png?generation=1686696885412256&alt=media)

# 3. Other things I tried

- After discovering that KeyNetAffNetHardNet was good at finding matching/pairing shortlists, I spent a lot of time exploring pair-wise matching such as LoFTR, SE-LoFTR, and DKMv3. However, these methods were slow even just running on the selected pairs and did not improve the mAAs in my implementations.
- Use AdaLAM matcher with other keypoint detectors such as DISK, SiLK, and ALIKE: I also used OriNet and AffNet to convert the keypoints from these detectors to Lafs and get the HardNet8 descriptors. I then tried matching on the native descriptors, HardNet8 descriptors, or concatenated descriptors. This approach seemed to perform better than matching on the native descriptors without Lafs. If I had more time, I would like to explore this direction further.
- Running local feature detection without resizing by splitting large images into smaller images that could fit into the GPU memory. However, this approach was extremely slow and did not improve performance. I found that the mAAs did not improve beyond a certain resolution. Allowing the longer edge to be 1600 or 2048 yielded similar scores.
- Tuning Colmap mapping options: Too many options and difficult to evaluate the outcomes.

# 4. Local Validation Score

```python
urban / kyiv-puppet-theater (26 images, 325 pairs) -> mAA=0.904923, mAA_q=0.931077, mAA_t=0.914154
urban -> mAA=0.904923

heritage / dioscuri (174 images, 15051 pairs) -> mAA=0.906996, mAA_q=0.982559, mAA_t=0.910856
heritage / cyprus (30 images, 435 pairs) -> mAA=0.855172, mAA_q=0.868966, mAA_t=0.865977
heritage / wall (43 images, 903 pairs) -> mAA=0.484496, mAA_q=0.820377, mAA_t=0.499336
heritage -> mAA=0.748888

haiper / bike (15 images, 105 pairs) -> mAA=0.921905, mAA_q=0.999048, mAA_t=0.921905
haiper / chairs (16 images, 120 pairs) -> mAA=0.979167, mAA_q=0.999167, mAA_t=0.979167
haiper / fountain (23 images, 253 pairs) -> mAA=0.999605, mAA_q=1.000000, mAA_t=0.999605
haiper -> mAA=0.966892

Final metric -> mAA=0.873568 (t: 1.7704436779022217 sec.)
```