# 40th place solution - SuperGlue + SIFT

## Overview

We used SuperGlue or SIFT on different scenes based on a heuristic and you can see our scores below.

Notebook: https://www.kaggle.com/code/gunesevitan/image-matching-challenge-2023-inference
Code: https://github.com/gunesevitan/image-matching-challenge-2023

### Scene Scores

|                     | mAA    | mAA Rotation | mAA Translation |
| ------------------- | ------ | ------------ | --------------- |
| bike                | 0.9228 | 0.9904       | 0.9228          |
| chairs              | 0.9775 | 0.9916       | 0.9775          |
| fountain            | 1.0    | 1.0          | 1.0             |
| dioscuri            | 0.5062 | 0.5220       | 0.5236          |
| cyprus              | 0.6523 | 0.7887       | 0.6586          |
| wall                | 0.8150 | 0.9359       | 0.8317          |
| kyiv-puppet-theater | 0.7704 | 0.8756       | 0.7895          |

### Dataset Scores

|          | mAA    | mAA Rotation | mAA Translation |
| -------- | ------ | ------------ | --------------- |
| haiper   | 0.9667 | 0.9994       | 0.9667          |
| heritage | 0.6578 | 0.7489       | 0.6713          |
| urban    | 0.7704 | 0.8756       | 0.7895          |

### Global Scores

| mAA    | mAA Rotation | mAA Translation |
| ------ | ------------ | --------------- |
| 0.7983 | 0.8746       | 0.8092          |

### LB Scores

Public LB Score: **0.415**
Private LB Score: **0.465**

## SuperPoint & SuperGlue

SuperPoint and SuperGlue models are used with almost default parameters except `keypoint_threshold` is set to 0.01. We found that SuperGlue works better with raw sizes but some of the scenes had very large images that didn't fit into GPU memory. We resized images to 2560 (maximum longest edge that can be used safely on Kaggle) longest edge if any of the edges exceed that number. Otherwise, raw sizes are used.

## SIFT

We initially started with COLMAP's SIFT implementation and it was working pretty good as a baseline. It was performing better on some scenes with very large images and strong rotations compared to deep models. There was a score trade-off between cyprus and wall while enabling `estimate_affine_shape` and `upright` and we ended up disabling both of them.

```python
sift_extraction_options.max_image_size = 1400
sift_extraction_options.max_num_features = 8192
sift_extraction_options.estimate_affine_shape = False
sift_extraction_options.upright = False
sift_extraction_options.normalization = 'L2'
```

## Model Selection

We noticed that large images with EXIF metadata has very high memory consumption and those are the images that have 90 degree rotations because of DSLR camera orientation. We add a simple if block that was checking the mean memory consumption of each scene. If that was greater than 16 megabytes, we used SIFT. Otherwise, we used SuperGlue on that scene.

## Incremental Mapper

We used COLMAP's incremental mapper for reconstruction with almost default parameters except `min_model_size` is set to 3. Best reconstruction is selected based on registered image count and unregistered images are filled with scene mean rotation matrix and translation vector.

## Thing that didn't work

- OpenCV SIFT (COLMAP's SIFT implementation was working way better for some reason)
- DSP-SIFT (domain size pooling was boosting my validation score on local but it was throwing an error on Kaggle)
- LoFTR (too slow)
- KeyNet, AffNet, HardNet (bad score)
- DISK (bad score)
- SiLK (too slow)
- ASpanFormer (too slow)
- Rotation Correction (we probably didn't make it correctly within the limited timeframe)
- Two stage matching (it was boosting my validation score but didn't have enough time to fit it into pipeline)