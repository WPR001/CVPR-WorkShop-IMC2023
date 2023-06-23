#!/usr/bin/env python
# coding: utf-8

# # ***Ensemble of the following works***
# - SuperGlue
'''
SuperGlue是一种用于计算机视觉任务的神经网络模型，旨在解决多个任务之间的对齐问题。它是由Facebook AI Research开发的，可以解决目标跟踪、语义分割、立体视觉和场景重建等任务。

SuperGlue算法的核心思想是将两个图像之间的相似性转化为一个连续的函数，然后使用深度神经网络来计算这个函数。这个函数被称为匹配分数，它给出了两个图像之间每个点的相似程度。通过计算匹配分数，SuperGlue算法可以将两个图像中的相同物体对齐，以便进行进一步的分析和处理。

相比传统的计算机视觉算法，SuperGlue算法可以更准确地对齐不同的图像，并且具有更好的鲁棒性和可扩展性。它已经在多个计算机视觉任务中取得了很好的效果，并且正在被广泛应用于实际场景中。
'''
#     https://www.kaggle.com/code/losveria/superglue-baseline/notebook

# - LoFTR
'''
LoFTR是一种基于深度学习的图像匹配算法，它可以用于图像检索、三维重构等应用。它可以在不同的图像之间找到相似的点，从而实现图像间的匹配。
'''
#     https://www.kaggle.com/code/cbeaud/imc-2022-kornia-score-0-725/notebook

'''
DKM算法是一种基于密度峰值的聚类算法，全称为Density Peak-based clustering Method。它通过确定数据点的局部密度和距离最近的更高密度点的距离来确定聚类中心，
从而将数据点划分为不同的簇。该算法的优点是不需要预先指定簇的数量，且对噪声数据具有鲁棒性。但是，该算法对于高维数据和大规模数据集的处理效率较低。
'''
# - DKM
#     https://www.kaggle.com/code/radac98/public-baseline-dkm-0-667
# 

# In[ ]:


dry_run = False


# # *Import dependencies and install Libs*

# In[ ]:


import logging
import time
import os
import numpy as np
import cv2
import csv
from glob import glob
import torch
import matplotlib.pyplot as plt
import gc
import pandas as pd

# the following dependencies are for superglue
import random
from collections import namedtuple
import sys
sys.path.append("../input/super-glue-pretrained-network")
from models.matching import Matching
from models.utils import frame2tensor

# the following dependencies are for loftr
get_ipython().system('pip install ../input/kornia-loftr/kornia-0.6.4-py2.py3-none-any.whl')
get_ipython().system('pip install ../input/kornia-loftr/kornia_moons-0.1.9-py3-none-any.whl')
import kornia
from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF


# the following dependencies are for DKM
from PIL import Image
sys.path.append('/kaggle/input/dkm-dependecies/DKM/')
get_ipython().system('mkdir -p pretrained/checkpoints')
get_ipython().system('cp /kaggle/input/dkm-dependecies/pretrained/dkm.pth pretrained/checkpoints/dkm_base_v11.pth')
get_ipython().system('pip install -f /kaggle/input/dkm-dependecies/wheels --no-index einops')
get_ipython().system('cp -r /kaggle/input/dkm-dependecies/DKM/ /kaggle/working/DKM/')
get_ipython().system('cd /kaggle/working/DKM/; pip install -f /kaggle/input/dkm-dependecies/wheels -e .')
torch.hub.set_dir('/kaggle/working/pretrained/')
from dkm import dkm_base


# # *Settings*

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src = '/kaggle/input/image-matching-challenge-2022/'


# ## *Utils*

# In[ ]:


test_samples = []
with open(f'{src}/test.csv') as f:
    #将csv中每一行中的每一列元素用','进行分隔
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        # Skip header.
        if i == 0:
            continue
        test_samples += [row]


def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


def resize_img_loftr(img, max_len, enlarge_scale, variant_scale, device):
    if max_len == -1:
        scale = 1
    else:
        scale = max(max_len, max(img.shape[0], img.shape[1]) * enlarge_scale) / max(img.shape[0], img.shape[1])
    w = int(round(img.shape[1] * scale) / 8) * 8
    h = int(round(img.shape[0] * scale) / 8) * 8
    
    isResized = False
    if w >= h:
        if int(h * variant_scale) <= w:
            isResized = True
            h = int(h * variant_scale / 8) * 8
    else:
        if int(w * variant_scale) <= h:
            isResized = True
            w = int(w * variant_scale / 8) * 8
    img_resize = cv2.resize(img, (w, h)) 
    img_resize = K.image_to_tensor(img_resize, False).float() / 255.
    
    return img_resize.to(device), (w / img.shape[1], h / img.shape[0]), isResized


def resize_img_superglue(img, max_len, enlarge_scale, variant_scale):
    if max_len == -1:
        scale = 1
    else:
        scale = max(max_len, max(img.shape[0], img.shape[1]) * enlarge_scale) / max(img.shape[0], img.shape[1])
    w = int(round(img.shape[1] * scale))
    h = int(round(img.shape[0] * scale))
    
    isResized = False
    if w >= h:
        if int(h * variant_scale) <= w:
            isResized = True
            h = int(h * variant_scale) 
    else:
        if int(w * variant_scale) <= h:
            isResized = True
            w = int(w * variant_scale)
    img_resize = cv2.resize(img, (w, h)) 
    return img_resize, (w / img.shape[1], h / img.shape[0]), isResized


# # *Matchers*

# In[ ]:


# ===========================
#          SuperGlue
# ===========================
resize = [-1, ]
resize_float = True
config = {
    "superpoint": {
        "nms_radius": 4, 
        "keypoint_threshold": 0.005,
        "max_keypoints": 2048
    },
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 10,
        "match_threshold": 0.2,
    }
}
matching_superglue = Matching(config).eval().to(device)

# ===========================
#          LoFTR
# ===========================
matcher_loftr = KF.LoFTR(pretrained=None)
matcher_loftr.load_state_dict(torch.load("../input/kornia-loftr/loftr_outdoor.ckpt")['state_dict'])
matcher_loftr = matcher_loftr.to(device).eval()

# ===========================
#          DKM
# ===========================
model = dkm_base(pretrained=True, version="v11")


# # Inference

# In[ ]:


F_dict = {}

scales_lens_superglue = [[1.2, 1200, 1.0], [1.2, 1600, 1.6], [0.8, 2000, 2], [1, 2800, 3]]

scales_lens_loftr = [[1.1, 1000, 1.0], [1, 1200, 1.3], [0.9, 1400, 1.6]]

w_h_muts_dkm = [[680 * 510, 1]]

np.random.seed(42)

# DEBUG = True
DEBUG = False

if DEBUG == True:
    import time
    st = time.time()

    
with torch.no_grad():
    for i, row in enumerate(test_samples):
        sample_id, batch_id, image_0_id, image_1_id = row    
        
        image_0_BGR = cv2.imread(f'{src}/test_images/{batch_id}/{image_0_id}.png') 
        image_1_BGR = cv2.imread(f'{src}/test_images/{batch_id}/{image_1_id}.png')
        
        image_0_GRAY = cv2.cvtColor(image_0_BGR, cv2.COLOR_BGR2GRAY)
        image_1_GRAY = cv2.cvtColor(image_1_BGR, cv2.COLOR_BGR2GRAY)
        
        # ===========================
        #           LoFTR
        # ===========================
        mkpts0_loftr_all = []
        mkpts1_loftr_all = []
        for variant_scale, max_len, enlarge_scale in scales_lens_loftr:
            
            image_0_resize, scale_0, isResized_0 = resize_img_loftr(image_0_GRAY, max_len, enlarge_scale, variant_scale, device)
            image_1_resize, scale_1, isResized_1 = resize_img_loftr(image_1_GRAY, max_len, enlarge_scale, variant_scale, device)
            
            if isResized_0 == False or isResized_1 == False: continue
            
            input_dict = {"image0": image_0_resize, 
                      "image1": image_1_resize}
            correspondences = matcher_loftr(input_dict)
            confidence = correspondences['confidence'].cpu().numpy()
            
            if len(confidence) < 1: continue

            confidence_quantile = np.quantile(confidence, 0.6)
            idx = np.where(confidence >= confidence_quantile)
            
            mkpts0_loftr = correspondences['keypoints0'].cpu().numpy()[idx]
            mkpts1_loftr = correspondences['keypoints1'].cpu().numpy()[idx]
            
            if DEBUG == True:
                print("loftr scale_0", scale_0)
                print("loftr scale_1", scale_1)

            mkpts0_loftr = mkpts0_loftr / scale_0
            mkpts1_loftr = mkpts1_loftr / scale_1

            mkpts0_loftr_all.append(mkpts0_loftr)
            mkpts1_loftr_all.append(mkpts1_loftr)
        
        mkpts0_loftr_all = np.concatenate(mkpts0_loftr_all, axis=0)
        mkpts1_loftr_all = np.concatenate(mkpts1_loftr_all, axis=0) 
    
        
        # ===========================
        #          SuperGlue
        # ===========================
        mkpts0_superglue_all = []
        mkpts1_superglue_all = []
        
        for variant_scale, max_len, enlarge_scale in scales_lens_superglue:
            image_0, scale_0, isResized_0 = resize_img_superglue(image_0_GRAY, max_len, enlarge_scale, variant_scale)
            image_1, scale_1, isResized_1 = resize_img_superglue(image_1_GRAY, max_len, enlarge_scale, variant_scale)
            
            if isResized_0 == False or isResized_1 == False: break 
            
            image_0 = frame2tensor(image_0, device)
            image_1 = frame2tensor(image_1, device)

            pred = matching_superglue({"image0": image_0, "image1": image_1})
            pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            matches, conf = pred["matches0"], pred["matching_scores0"]
            
            valid = matches > -1
            mkpts0_superglue = kpts0[valid]
            mkpts1_superglue = kpts1[matches[valid]]

            if DEBUG == True:
                print("superglue scale_0", scale_0)
                print("superglue scale_1", scale_1)

            mkpts0_superglue /= scale_0
            mkpts1_superglue /= scale_1

            mkpts0_superglue_all.append(mkpts0_superglue)
            mkpts1_superglue_all.append(mkpts1_superglue)
            
        if len(mkpts0_superglue_all) > 0:
            mkpts0_superglue_all = np.concatenate(mkpts0_superglue_all, axis=0)
            mkpts1_superglue_all = np.concatenate(mkpts1_superglue_all, axis=0) 
        
        
        # ===========================
        #            DKM
        # ===========================
        img0PIL = Image.fromarray(cv2.cvtColor(image_0_BGR, cv2.COLOR_BGR2RGB))
        img1PIL = Image.fromarray(cv2.cvtColor(image_1_BGR, cv2.COLOR_BGR2RGB))
        
        mkpts0_dkm_all = []
        mkpts1_dkm_all = []
        
        for w_h_mut, param in w_h_muts_dkm:
            
            ratio = (image_0_BGR.shape[0] + image_1_BGR.shape[0]) / (image_0_BGR.shape[1] + image_1_BGR.shape[1]) * param # 根据图0的高宽比确定计算参数
            
            model.w_resized = int(np.sqrt(w_h_mut / ratio))
            model.h_resized = int(ratio * model.w_resized)
            
            dense_matches, dense_certainty = model.match(img0PIL, img1PIL, do_pred_in_og_res=True)
            dense_certainty = dense_certainty.pow(0.6)
            
            sparse_matches, sparse_certainty = model.sample(dense_matches, dense_certainty, max(min(500, (len(mkpts0_loftr_all) + len(mkpts0_superglue_all)) // int(4 * len(w_h_muts_dkm))), 100), 0.01)
            mkpts0_dkm = sparse_matches[:, :2]
            mkpts1_dkm = sparse_matches[:, 2:]
            h, w, c = image_0_BGR.shape
            mkpts0_dkm[:, 0] = ((mkpts0_dkm[:, 0] + 1) / 2) * w
            mkpts0_dkm[:, 1] = ((mkpts0_dkm[:, 1] + 1) / 2) * h
            h, w, c = image_1_BGR.shape
            mkpts1_dkm[:, 0] = ((mkpts1_dkm[:, 0] + 1) / 2) * w
            mkpts1_dkm[:, 1] = ((mkpts1_dkm[:, 1] + 1) / 2) * h

            mkpts0_dkm_all.append(mkpts0_dkm)
            mkpts1_dkm_all.append(mkpts1_dkm)

        mkpts0_dkm_all = np.concatenate(mkpts0_dkm_all, axis=0)
        mkpts1_dkm_all = np.concatenate(mkpts1_dkm_all, axis=0)
        
        # ensemble of all mkpts 
        mkpts0 = []
        mkpts1 = []
        
        if len(mkpts0_loftr_all) > 0:
            mkpts0.append(mkpts0_loftr_all)
            mkpts1.append(mkpts1_loftr_all)
        
        if len(mkpts0_superglue_all) > 0:
            mkpts0.append(mkpts0_superglue_all)
            mkpts1.append(mkpts1_superglue_all)
        
        mkpts0.append(mkpts0_dkm_all)
        mkpts1.append(mkpts1_dkm_all)
        
        mkpts0 = np.concatenate(mkpts0, axis=0)
        mkpts1 = np.concatenate(mkpts1, axis=0)  
       
        if len(mkpts0) > 8:
            F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.15, 0.9999, 20000)
            F_dict[sample_id] = F
        else:
            F_dict[sample_id] = np.zeros((3, 3))
            continue
        
        if DEBUG == True: 
            print("the number of loftr keypoints: ", len(mkpts0_loftr_all))
            print("the number of superglue keypoints: ", len(mkpts0_superglue_all))
            print("the number of dkm keypoints: ", len(mkpts0_dkm_all))
            print("the number of all keypoints: ", len(mkpts0))
            print("Fundamental matrix: ")
            print(F_dict[sample_id])
        
        gc.collect()

if DEBUG == True:
    ed = time.time()
    print(f"spend {ed - st:.2f}s")
    
with open('submission.csv', 'w') as f:
    f.write('sample_id,fundamental_matrix\n')
    for sample_id, F in F_dict.items():
        f.write(f'{sample_id},{FlattenMatrix(F)}\n')

