# 31st place solution

Congratulations to everyone for the journey we had throughout the competition. Also, thanks to the organizers for bringing the Image Matching Challenge to Kaggle again.
I will give a brief of my solution to this challenge.

## Architecture

I came to the competition with very limited time and only a little experience from IMC 2022. So, it is likely the first time I walked through the flow of 3D reconstruction.
I strictly followed the host pipeline, which I split into 3 modules:

- Global descriptors
- Local descriptors (matching)
- Reconstruction

My main work was focused on improving their efficiency separately.

## Global descriptors

From my point of view, well-trained models on a **landmark** dataset should give better descriptors than ImageNet pre-trained backbones.
As a result, I utilized some of the models that I had trained to compete in the [Google Landmark 2021](https://www.kaggle.com/competitions/landmark-recognition-2021), and then concatenate them to a global descriptor:

```
EfficientNetV2-M \
EfficientNetV2-L  \
.                  |-->[concat]--> [fc 2048]
ResNeSt-200       /  
ResNeSt-269      /
```

## Local descriptors

This year, competitors are required to perform matching in a strict time interval.
I focused on **detector-based** (2-stage) methods only, because thought I could save time on the points detector part (for example, to match *(image_i, image_j)* and *(image_i, image_k)*, semi-dense and dense methods will have to "extract" *image_i* two times). When I read other top team solutions, it seemed to be a wrong decision I had made, since such an amount of good matching models are omitted 😭. However, here is the list of methods I tried:
**Detector**: SuperPoint, SuperPoint + FeatureBooster, KeyNetAffNetHardNet, DISK, SiLK, ALIKE.
**Matcher**: SuperGlue, GlueStick, SGMNet, AdaLAM.
With the detector, I found that **SuperPoint** gave superior results than others.
With the matcher, **SuperGlue** showed the best performance in accuracy and efficiency. **GlueStick** is quite good but slower. **SGMNet** is quite fast but lower. I then ensemble keypoints and matches from their predictions and filter out duplicates.

## Reconstruction

I didn't think I could improve much on this, so I only played with **colmap parameters** a bit to find out a (maybe) better combination than the default one.
Some parameters I changed:

```
max_num_trials
ba_global_images_freq
ba_global_max_num_iterations
ba_global_points_freq
ba_local_max_num_iterations
init_num_trials
max_num_models
min_model_size
```

I could save a little time with a "lighter" combination of parameters while still keeping the accuracy.

## Final thoughts

I guess mine is quite a simple solution, but still give me a silver :D.
However, the knowledge I gained from the competition may be the best I could achieve.
Thank you for your reading and happy Kaggling!