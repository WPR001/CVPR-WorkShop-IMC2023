# 2th place solution – Win over COLMAP Randomness?!

## **Intro**

Our team would like to deeply appreciate the Kaggle staff, Google Research, and Haiper for hosting the continuation of this exciting image matching challenge, as well as everyone here who compete and shared great discussions. My congratulations to all participants!

The work we describe here is truly a joint effort of [@yamsam](https://www.kaggle.com/yamsam), [@remekkinas](https://www.kaggle.com/remekkinas), and [@vostankovich](https://www.kaggle.com/vostankovich). I’m grateful of being a part of this hardworking, cohesive, and skilled team. Thanks a lot, guys! I learned a lot from you.

## **We enjoyed this competition!**

The best submission that was used for final scoring in private LB finished on the last day of the competition. We weren’t completely sure about this submission because it was not clear how much randomness was in it. We had a practice to re-run the same notebook code multiple times to see what scores we can get. We discussed the solution which was implemented on the last day, trusted it and it worked out. The other interesting fact is that our 2nd selected submission for final evaluation scored **0.497/0.542** that also allows us to take 2nd place. This selected second submission is the same as the 1st one except the “Run reconstruction multiple times” trick, that is described below. Anyway, the difference between the best submission (**0.562**) and the 2nd one is noticable.

## **Overview**

In general, throughout our code submissions every time we fight with the randomness coming from COLMAP responsible for scene reconstruction. Our final solution is based on the use of COLMAP and pretrained SuperPoint/SuperGlue models running on different resolutions for every image in the scene. We apply a bunch of different tricks aimed at different parts of COLMAP-based pipeline in order to stabilize our solution and reach the final score. Here are the key steps of our solution:

- Initially, use all **possible unique image pairs** generated from the scene set. Remove the model and logic used for finding and ranking similar images in the scene. A threshold of 100 defines a minimum number of matches that we expect each image pair to have. If it is less, we discard that pair.
- **SP/SG** settings: unlimited N of keypoints, the keypoint threshold is **0.005**, match threshold **0.2**, and the number of sinkhorn iteratiors is **20**.
- **Half precision** for SP/SG helped to reduce occupied memory without sacrificing noticable accuracy. Another great performance trick is to cache keypoints/descriptors, generated by SP for each image, and, then cache SG matches for every image pair. It allowed to reduce the running time a lot.
- **TTA**. Ensemble of matches extracted from images at different scales. In our local experiments, the best results are achieved with a combination of **[1088, 1280, 1376]**. We used np.concatenate to join matches from different models that was pretty common for the last IMC22 competition.
- Apply **rotate detector** to un-rotate images in the scene if necessary. We discovered that some scenes in the train dataset (cyprus, dioscuri) have many 90/270 rotated images. Some of the images have EXIF meta information. Unfortunately, looking into the train dataset, we did not find any specifics about the orientation the image was captured at. To address this rotation issue, we re-rotate the image to its natural orientation by employing this [solution](https://github.com/ternaus/check_orientation). We use it w/o any threshold and look at the number of rotations that we need to apply to an image. After applying rotation, the score for cyprus scene jumped up significantly from **~0.02** up to **~0.55**. RotNet implementation did not work out for us.
- **Set-up initial image** for COLMAP reconstruction explicitly. For each image we store the number of pairs in which it seen and the number of matches these pairs produce. Then we pick the one with the highest number of pairs. If multiple images satisfy this criterion, we pick the one with the highest number of matches. It helped to boost the score.
- To reduce randomness in the score, we decided to do something like **averaging multiple match_exhaustive** calls. The idea is to run match_exhaustive for N times on the original database of matches. Then we take only those matches that that appear in 8/10 cases, other matches are neglected. It was done in a rude way with database copies write/read, etc.
- **Run reconstruction multiple times** from scratch with different matchers threshold e.g. **[100, 125, 75, 100]**. By looking at N of registered images and number of 3D cloud points, we select the best reconstruction. This trick not only allows to find the better reconstruction by finding the better threshold for matches, but also decrease the randomness effect and acts as a countermeasure against a shake up. Due to its running time complexity, we used this strategy only for scenes having less than 40-45 images. This is the last step in our solution that helped us to boost score from **0.497/0.542** to **0.506/0.562**. We also experimented with pycolmap.incremental_mapping employing similar idea but that scenario did not work out.

## **Ideas that did not work out or not fully tested:**

**TTA multi crop**, no success. The idea was to split image into multiple crops and extract matches in order to find similar images in the scene and determine the best image pairs.
• **Square-sized images.**
• **Bigger image size** (e.g., 1600) for SP/SG.
• **Manual RANSAC** instead of using COLMAP internal implementation. We run experiments by disabling geometric verification, but the score was not good.
• **NMS filtering** to reduce number of points by using [ANMS](https://github.com/BAILOOL/ANMS-Codes).
• Filter least significant image pairs by number of outliers instead of relying on raw matches. It was quite important to look for certain number of matches for an image pair. Run experiments using SP/SG and Loftr. We got higher mAA with Loftr, but, probably, more effort needed here to make it work properly, not enough time.
• **Downscale scene images** before passing them to COLMAP.
• **Pixel-Perfect Structure-from-Motion**. It was a promising method to evaluate as we got a good boost locally with [PixSfm](https://github.com/cvg/pixel-perfect-sfm), using a single image size of 1280, and it boosted our score from **0.71727** to **0.76253**. Then, we managed to install this framework successfully and run it in Kaggle environment, but could not beat our best score at that moment. It is a heavyweight framework taking too much RAM, and we could run it only for scenes having at most ~30 images. A bit upset because we spent tons of hours compiling all this stuff.
• **Adaptive image sizes**. Say, if the longest image side >= 1536 for most images in the scene, we use higher image resolution for matchers ensemble e.g. [1280, 1408, 1536]. Otherwise, a default one is applied [1280,1088,1376]. Did not have enough time to test this idea. It worked locally for cyprus and wall that have big resolution. One of our last submissions implementing this idea crashed with internal error.
• Different **detectors, matchers**. We tested Loftr, QuadreeAttention, AspanFormer, DKM v3, GlueStick (keypoints + lines), Patch2Pix, KeyNetAffNetHardNet, DISK, PatchNetVLAD. We also experimented with the confidence matching thresholds and number of matches, but no boost here. Eventually, SP/SG was the best choice for us. Probably, the reason why many dense-based methods did not work out for us is because of the low performance of the “repeatability” metric and high noise in the matches.
• Different **CNNs to find the most similar images** in the scene and generate corresponding image pairs (NetVLAD, different pretrained timm-based backbones, CosPlace etc). We even specifically trained a model to find similar images in the scene, but no success here. Later, we gave up using this strategy at all.
• Different **keypoint/matching refinement** methods (e.g., recently published [AdaMatcher](https://github.com/TencentYoutuResearch/AdaMatcher), Patch2Pix ), but did not have enough time. AdaMatcher seems a promising idea to try, a quote from their paper “as a refinement network for SP/SG we observe a noticeable improvement in AUC”

**The final score is 0.506/0.562 in Public/Private.**

As a reference, this is one of our latest metric reports using train dataset:

urban / kyiv-puppet-theater (26 images, 325 pairs) -> mAA=0.921538, mAA_q=0.991077, mAA_t=0.921846
urban -> mAA=0.921538

heritage / cyprus (30 images, 435 pairs) -> mAA=0.514713, mAA_q=0.525287, mAA_t=0.543678
heritage / wall (43 images, 903 pairs) -> mAA=0.495792, mAA_q=0.875637, mAA_t=0.509302
heritage -> mAA=0.505252

haiper / bike (15 images, 105 pairs) -> mAA=0.940952, mAA_q=0.999048, mAA_t=0.940952
haiper / chairs (16 images, 120 pairs) -> mAA=0.834167, mAA_q=0.863333, mAA_t=0.839167
haiper / fountain (23 images, 253 pairs) -> mAA=0.999605, mAA_q=1.000000, mAA_t=0.999605
haiper -> mAA=0.924908

**Final metric -> mAA=0.783900**

Finally, we had two submissions running on the last day. One of them succeeded and allowed us to get 2nd place, but the other one did not fit the time limit, unexpectedly. Sometimes it is stressful to make a submission on the last day of the competition.