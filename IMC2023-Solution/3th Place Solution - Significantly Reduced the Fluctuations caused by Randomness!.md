# 3th Place Solution - Significantly Reduced the Fluctuations caused by Randomness!

We are delighted to participate in this competition and would like to express gratitude to all the Kaggle staffs and Sponsors. Congratulations to all the participants.
The team members include 陈鹏、陈建国、阮志伟 and 李伟. I would like to express my sincere gratitude to everyone for the excellent teamwork over the past month. I have thoroughly enjoyed working with all of you, and I am delighted to be a part of this team.

# 1 Overview

![pipeline](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3163774%2Fc8338666681788720689a7ead39a92ab%2F.png?generation=1686757921566293&alt=media)

# 2 Main pipeline

## 2.1 SP/SG

### 2.1.1 Rotation

Rotating the image has a significant effect, since SG is lack of the rotation invariance. Therefore, for each image pair A-B, we fixed image A and rotated image B four times (0, 90, 180, 270). After performing four times SG matching, we selected the rotation angle with the most matches for the next stage.
![Rotation](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3163774%2F33aa1271502732f84ca6420949ed19a6%2F.png?generation=1686758077155244&alt=media)

### 2.1.2 Resize image

In the early stage of the competition, we used the original image for extracting keypoints and matching. And we noticed that the mAA score in the heritage cyprus scene was only 0.1. After experiments, we found that if we scaled the images in cyprus scene to 1920 on the longer side, the mAA score was improved to 0.6. Specifically, we didn't rescale the image size after SP keypoints extraction. Finally, in our application, we scaled the image size if the longer side was larger than 1920, otherwise kept the original image size for the following SP + SG inference.

### 2.1.3 SP/SG setting

We increased the NMS value of SP from 3 to 8 and set the maximum number of keypoints to 4000.

## 2.2 GeoVerification（RANSAC）

We used USAC_MAGSAC for geometric verification with the following configuration.
cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 2, 0.99999, 100000)

## 2.3 Setting Camera Params（Randomness）

In our experiments, we wanted to eliminate the randomness effect in our final mAA, since we observed that the same notebook could result in fluctuations of approximately 0.03 in the LB. We found that the randomness come mainly from the Ceres (https://github.com/colmap/colmap/issues/404)) optimization.
After analysis, the initial values of camera parameters, such as the focal length, had a significant impact on the stability of optimization results.
![randomness](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3163774%2F9411c508f380417b835bb253b06eae7b%2Fwall_random.png?generation=1686758232496900&alt=media)
Since the camera focal length information could be extracted from the image EXIF data, we provided the prior camera focal length to the camera and shared the same camera settings among images captured by the same device. Although there were still some fluctuations in the metrics, the majority of the experimental results were consistent.
The figure below shows four submissions of the same final notebook. **On the public LB, our metric fluctuates by no more than 0.004.** Moreover, compared to many other teams, **we do not have significant metric fluctuations between public LB and private LB.** Our public LB score was amazingly close to private score. This might indicated that our method **Significantly Reduced the Fluctuations caused by Randomness!**
![mAA](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3163774%2Ffd18321289d3745de78691742d0ca28d%2Fonline_maa.png?generation=1686758332640650&alt=media)

## 2.4 Mapper

In the COLMAP reconstruction process, we revised some default parameters in incremental mapping. After first trail, if the best model register ratio was below 1, we relaxed the mapper configuration (abs_pose_min_num_inliers=15) and re-run incremental mapping.

# 3 Conclusion

![Conclusion](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3163774%2F47522f20ba1cf5b608a1806f0cf761f4%2Fconclution.png?generation=1686760502421586&alt=media)

# 4 Not fully tested

## 4.1 Select image pairs

Our final solution took nearly 9 hours, since we generated the image matching pairs by exhaustive method. In order to speed up our solution, we tried Efficientnet_b7, Convnextv2_huge, and DINOv2 to extract image features and generate image matching pairs by feature similarity. In offline experiments, we selected the top N/2 most similar images (N being the total number of images) for each image to form image pairs. Compared to other methods, DINOv2 performed the best. By incorporating DINOv2 into the image pairs selection, we could control the processing time to 7 hours, and achieved 0.485 in LB (compared with exhaustive 0.507).
![dinov2](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3163774%2Fbfdb729a111a6633b27e573788e09263%2Fdinov2.png?generation=1686760519524067&alt=media)

## 4.2 Pixel-Perfect Structure-from-Motion

PixSFM had a good performance improvement in our local validation. However, during online testing, it consistently times out even we run it on scenes less than 50 images. Moreover, it took a lot time for us to install the environment and run it on kaggle. Times out Sad!

# 5 Ideas that did not work well

## 5.1 Different detectors and matchers

We tested DKMv3, GlueStick, and SILK, but neither was able to surpass SPSG. In our experiments, we observed that DKMv3 performed better than SPSG in challenging scenes, such as those with large viewpoint differences or rotations.
![dkm](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3163774%2Ffae472d5266e521f18f667592eaf3454%2Fdkm.png?generation=1686758763195086&alt=media)
However, the overall metrics of DKMv3 were not as good as SPSG, which may be due to the operation of sampling dense matches we performed.

## 5.2 Crop image

Just like last year's winning solution, we took a crop after completing the first stage of matching. In the second stage, we performed matching again on the crop and merge all the matching results together.

## 5.3 Half precision

It confused our team a lot. In the kaggle notebook, we found the half-precision improved the results, but on the public LB, it returned the lower results.

## 5.4 Merge matches in four directions

In our methods, we rotated the image (0, 90, 180, 270) and performed four times matching. Instead of keeping all matches from four directions, we only keep the best angle matches, because keeping all matches didn't lead to any improvement in the results.

## 5.5 findHomography

When feature points lie on the same plane (e.g., in a wall scene) or when the camera undergoes pure rotation, the fundamental matrix degenerates. Therefore, in RANSAC, we simultaneously used the findFundamentalMat and the findHomography to calculate the inliers. However, this approach didn't lead to an improvement in the metrics.

## 5.6 3D Model Refinement

After the first trail of mapping, we tried to filter nosiy 3D points with large projection error or short track length, and then re-bundle adjust the model. Experically, this could help export better poses, but.. that's life.