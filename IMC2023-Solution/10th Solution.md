# 10th Solution

Great thanks to the organizers and Kaggle staff for this amazing competition.
Our solution shares architectural similarities with the baseline provided by the organizers.
Firstly, we addressed the issue of rotation in-variance by implementing a rotation model to standardize the orientation of input images. Next, we employed a neural network for the retrieval task, enabling us to extract matching pairs for the feature extraction and matching process. This process generated a database which served as the input for the incremental mapping in Colmap.

## Orientation Model

We would like to express our gratitude to [@iglovikov](https://www.kaggle.com/iglovikov) for developing this great [model](https://github.com/ternaus/check_orientation) , which was trained on a large dataset and exhibited good performance in both (LB) and (CV) evaluations. We made a slight modification to this model, which we refer to as self-ensembling.
In our proposed adjustment, we utilized the model iteratively on the query image and its different rotated versions. Subsequently, we tuned the threshold by conducting validation tests on more than 1500 images with different rotation angles. Through experimentation, we discovered that using a threshold between 0.8 and 0.87 might result in incorrect orientation predictions. To address this issue, we applied self-ensembling by further rotating the image and checking if the predicted class probability fell within this range of thresholds. We repeated this last step twice to ensure accuracy.

## Retrieval Method

### Retrieval Model

In order to address the challenges posed by the cost inefficiency of exhaustive matching and the limitations of sequential matching for 3D reconstruction, we sought alternative methods for image retrieval.
After careful evaluation, we chose to utilize NetVlad as our chosen method due to its superior performance compared to openibl and cosplace. To further enhance the results, we employed various techniques, including:

1) We passed the original image and its horizontally flipped version through the model. By summing the descriptors obtained from both passes, we achieved improved performance, particularly for highly similar images. This technique is especially effective in cases where the scene exhibits symmetry. By redefining a new point in the n-dimensional descriptive space, which effectively increase the similarity distance between two distinct parts of the scene.
2) Re-ranking: After calculating the similarity scores, we performed re-ranking by selecting the top 1 match. We then re-queried the retrieval process using these two images instead of just one. The similarity scores of the resulting matches were summed together after raising them to a specific power "m". This manipulation of probabilities ensures that if the best match for one of the query images is found, it will be favored over an image that is similar to both where the sum of the similarities on both scenarios is equal.

We repeated this procedure twice using NetVlad with Test Time Augmentation (TTA) on the image size. The results were nearly perfect, and the approach even enabled the correct ordering of The Wall scene pairs, which can be best matched by doing sequential matching.

### Number of Matches

Determining the number of image pairs to select is a critical factor that directly affects both validation and leaderboard performance. This becomes particularly important for large scenes where there may be no or very few common images between subsets of the scene.

To address this challenge, we devised a strategy based on a complete graph representation. In this graph, each edge represents the similarity between two images. The goal is to choose a subset of edges where each image has an equal number of connected nodes.

We employed a Binary Search approach to determine the number of matches for each image, with a check function to verify if the resulting graph is connected or not. The lower bound of the binary search was set to half the number of images, ensuring that we consider common matches and prevent incomplete 3D model reconstruction. Additionally, we made sure that the approach remains exhaustive for small scenes containing less than 40 images.

By employing this method, we aimed to strike a balance between capturing sufficient matching pairs for accurate 3D reconstruction while avoiding redundant or disconnected image subsets.

### Feature Extraction and Matching

In our selected submissions, we have utilized the SuperPoint and SuperGlue algorithms followed by MagSac filtering. Unfortunately SuperGlue is not licensed for commercial use. However, we have also achieved highly promising results using GlueStick. We have made modifications to the GlueStick architecture to integrate it with SuperPoint, and we achieved a score of approximately 0.450 on the public leaderboard and 0.513 on private leaderboard without employing our best tuning parameters. It is worth noting that this modified architecture is permitted for commercial use and offers improved processing speed. We anticipate that further tuning can yield even better results with GlueStick, but didn't choose it as our last submissions.

### Refinement

Although not included in our selected submissions, we would like to mention an approach that significantly improved validation results across various scenes. We employed Pixel-Perfect-SFM in conjunction with sd2net for dense matching, but it didn't improved the results on public leaderboard.

### Registering Unregistered Images

While not part of our selected submissions, we made attempts to register unregistered images using various techniques. However, these attempts did not yield significant improvements on the leaderboard. We explored the following strategies:

- Utilizing different orientations and attempting registration.
- Applying different extractor-matchers, such as LoFTR and R2D2, for the unregistered images.
- Adjusting parameters for SuperPoint and SuperGlue to optimize the registration process.

### Tried but not worked

- Pixel-Perfect-SFM
- Semantic Segmentation masks
- Illumenation enhancement
- PnP localization for unregistered Images
- LoFTR/ SILK / DAC / Wireframe
- CosPlace / OpenIBL
- Large Image splitting into Parts
- Grid-based Point Sampling (equally spatial replacement of points in an image).
- Rotation self-ensemble averaging
- Filtering the extracted pairs using a threshold from a binarysearch function or a fixed threshold.

### Important Notes:

1) Most participants were using colmap reconstruction, which is nondeterministic. We were able to modify the code so that it became deterministic but working only one CPU thread, which helped us observe our improvement and avoid randomness.

2) We found out that OpenCV and PIL libs use EXIF Information when reading images, unless providing flags to prevent it. By that, we mean that if the orientation in EXIF will rotate the image automatically before processing. This was confusing for us, as there are missing information about how the GT were collected for rotation part (with or without considering them), that's why one of our chosen solutions included this correction to overcome such issue, plus we lost a lot of submission to check the effect of this issue on leaderboard. It would have been more helpful if there was better explanation about how the GT calculated.

3) Our validation scored 0.485 on public and 0.514 on private, with local score of 0.87.

4) The fact that validation and leaderboard were not correlated made things more difficult and random, as it was clear there will be a shake up due to the fact that for some specific scene some solutions might fail which will drastically impact the mAA score.

   