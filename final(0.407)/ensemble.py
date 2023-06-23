# 5
import os
import cv2
import sys
import h5py
import torch
import shutil
import sqlite3
import warnings
import pycolmap
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from PIL import Image, ExifTags
import torch.nn.functional as F
from collections import defaultdict
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

sys.path.append('/kaggle/input')
sys.path.append('/tmp')
from sglib.models.matching import Matching

INPUT_ROOT = '/kaggle/input/image-matching-challenge-2023'
DATA_ROOT = '/kaggle/data'
OUTPUT_ROOT = '/kaggle/working'

DEBUG = False


datasets_scenes = []
sample_submission_df = pd.read_csv(f"{INPUT_ROOT}/sample_submission.csv")
for _, r in sample_submission_df[['dataset', 'scene']].iterrows():
    ds = f"{r.dataset}/{r.scene}"
    if ds not in datasets_scenes:
        datasets_scenes.append(ds)


matching_name = 'SuperGlue'
image_sizes = [1024, 1280] 
# image_sizes = [1280,1472]
# image_sizes = [1152,1472]
number_of_exh = 8
thre_exh = 6

USE_ROI = False
ROI_SIZE = 1024

sim_th = None
n_matches = 100

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

matching_model = Matching(matching_config).cuda().half().eval()




# loftr

import kornia
from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF

# LoFTR
sys.path.append('/kaggle/input/othermodels/github_LoFTR-master') 

# Kornia, a dual-softmax operator
# models https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp

kornia_max_image_size = 600 # max image size for LoFTR 表示 LoFTR 算法处理的图像的最大尺寸是 1120 像素；
kornia_at_least_matches = 180 # at least matches for LoFTR 表示在 LoFTR 算法中至少需要匹配 280 个特征点，才能被认为是有效的匹配；
kornia_thrs_conf_match = 0.3 # threshold for confidence match for LoFTR 表示在 LoFTR 算法中特征点匹配的置信度阈值为 0.3；
kornia_max_matches = 1000 # -1 -> use all 表示在 LoFTR 算法中最多匹配 1000 个特征点。如果设置为 -1，则使用所有的特征点进行匹配。

kf_loftr_out_device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda') # use cpu or cuda
kf_loftr_out_matcher = KF.LoFTR(pretrained=None) # load model
kf_loftr_out_matcher.load_state_dict(torch.load("/kaggle/input/kornia-loftr/loftr_outdoor.ckpt")['state_dict']) # load weights
kf_loftr_out_matcher = kf_loftr_out_matcher.to(kf_loftr_out_device).eval() # set to eval mode



def load_torch_image(fname, device, max_image_size):
    '''
    Load image from file and convert to torch tensor. 最长边 扩大至指定尺寸
    '''
    img = cv2.imread(fname)
    scale = max_image_size / max(img.shape[0], img.shape[1])  # 扩大倍数
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device), scale


def filter_conf_matches(mkpts0, mkpts1, mconf, thrs_conf_match=0.2, max_matches=-1):
    '''
    Filter matches by confidence.
    '''
    # sort points by confidence descending
    mkps1_sorted = [x for (y,x) in sorted(zip(mconf,mkpts0), key=lambda pair: pair[0], reverse=True)] # 根据置信度降序排序image1的点
    mkps2_sorted = [x for (y,x) in sorted(zip(mconf,mkpts1), key=lambda pair: pair[0], reverse=True)] # 根据置信度降序排序image2的点
    
    # overwrite
    mkps1 = np.array(mkps1_sorted) 
    mkps2 = np.array(mkps2_sorted)

    num_thrsh_greater = (mconf >= thrs_conf_match).sum() # 置信度大于阈值的点的数量

    take_first_el = min(max_matches, num_thrsh_greater) # matches 数量限制
    if take_first_el > 0 and len(mkps1) > take_first_el:
        mkps1 = mkps1[:take_first_el] # image1 取前take_first_el个点
        mkps2 = mkps2[:take_first_el] # image2 取前take_first_el个点
    return mkps1, mkps2


def loftr_inference(fname1, fname2, max_image_size):    
    '''
    Infer matches using LoFTR.
    '''
    image_1, scale1 = load_torch_image(fname1, kf_loftr_out_device, max_image_size) # image1 最长边 扩大至指定尺寸
    image_2, scale2 = load_torch_image(fname2, kf_loftr_out_device, max_image_size) # image2 最长边 扩大至指定尺寸
    
    # 颜色格式
    input_dict = {"image0": K.color.rgb_to_grayscale(image_1),
                  "image1": K.color.rgb_to_grayscale(image_2)
              }

    with torch.no_grad():
        correspondences = kf_loftr_out_matcher(input_dict) # LoFTR模型 推理两张图片
        
    # mkpts0, mkpts1, confidence
    mkpts0 = correspondences['keypoints0'].cpu().numpy() # keypoints0
    mkpts1 = correspondences['keypoints1'].cpu().numpy() # keypoints1
    
    # 将坐标缩放到原图
    mkpts0 = mkpts0 / scale1 
    mkpts1 = mkpts1 / scale2     
    
    return mkpts0, mkpts1












def resize_img(img, max_image_size):
    if max_image_size == -1:
        return img, 1.0 # no resize
    scale = max_image_size / max(img.shape[0], img.shape[1]) 
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    return img, scale


def get_crop_img(img_path, mkpts):
    if len(mkpts) < 10:  # sanity check
        return None, None
    img = cv2.imread(img_path)
    im_h, im_w, _ = img.shape

    min_x, min_y = np.amin(mkpts[:, 0]), np.amin(mkpts[:, 1])
    max_x, max_y = np.amax(mkpts[:, 0]), np.amax(mkpts[:, 1])
    left, top, right, bottom = min_x, min_y, max_x, max_y

    pad = 4
    x = max(0, int(left - pad))
    xr = min(im_w-1, math.ceil(right + pad))
    y = max(0, int(top - pad))
    yb = min(im_h-1, math.ceil(bottom + pad))
    crop_img = img[y:yb, x:xr]

    h_crop, w_crop = crop_img.shape[:2]
    if min(h_crop, w_crop) < 10:
        return None, None
    shift_xy = (x, y)
    return crop_img, shift_xy

def db_scan_new(mkpts, min_samples=5, max_dst=40):
    # min_samples = 6  # round(len(mkpt1) * 0.8)
    # max_dst = 40  # maximum distance between two samples
    db = DBSCAN(eps=max_dst, min_samples=min_samples).fit(mkpts)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    if n_clusters_ < 1:
        return None


    filtered_labels = [x for x in labels if x != -1]
    unique, counts = np.unique(filtered_labels, return_counts=True)

    T = 0.2
    all_idxs = []
    for lbl_idx in np.argsort(counts)[::-1]:
        if counts[lbl_idx] / counts.max() >= T:
            idxs = np.argwhere(filtered_labels == lbl_idx).flatten()
            all_idxs.extend(idxs)
    all_idxs = np.array(sorted(all_idxs))
    dense_mkpts = mkpts[all_idxs]
    return dense_mkpts

def extract_crops_via_cluster(im1_path, im2_path, mkpts0, mkpts1):
    #dense_mkpts0, dense_mkpts1 = db_scan(mkpts0, mkpts1)

    im_h, im_w, _ = cv2.imread(im1_path).shape
    px_scale_factor = max(1.0, max(im_h, im_w) / 1024)
    px_scaled_dst = int(40 * px_scale_factor)

    dense_mkpts0 = db_scan_new(mkpts0, min_samples=5, max_dst=px_scaled_dst)
    dense_mkpts1 = db_scan_new(mkpts1, min_samples=5, max_dst=px_scaled_dst)
    if dense_mkpts0 is None or dense_mkpts1 is None:
        return None, None, None, None


    # crop by dense matches
    cropped_img1, shift_xy1 = get_crop_img(im1_path, dense_mkpts0)
    cropped_img2, shift_xy2 = get_crop_img(im2_path, dense_mkpts1)
    return cropped_img1, cropped_img2, shift_xy1, shift_xy2


def get_img_pairs_all(fnames):
    index_pairs, h_w_exif = [], {}
    for i in range(len(fnames)):
        img = Image.open(fnames[i])
        w, h = img.size
        h_w_exif[fnames[i].split('/')[-1]] = {'h': h, 'w': w, 'exif': img._getexif()}
        for j in range(i+1, len(fnames)):
            index_pairs.append((i,j))
    return index_pairs, h_w_exif


def get_global_desc(model, fnames):
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    global_descs_convnext, h_w_exif = [], {}
    for fname in tqdm(fnames, desc='Get global features using similarity model'):
        img = Image.open(fname)
        w, h = img.size
        h_w_exif[fname.split('/')[-1]] = {'h': h, 'w': w, 'exif': img._getexif()}
        timg = transform(img.convert('RGB')).unsqueeze(0).cuda().half()
        with torch.no_grad():
            desc = model.forward_features(timg.cuda().half()).mean(dim=(-1,2))
            desc = desc.view(1, -1)
            desc_norm = F.normalize(desc, dim=1, p=2)
        global_descs_convnext.append(desc_norm.detach().cpu())
    global_descs_all = torch.cat(global_descs_convnext, dim=0)
    return global_descs_all.type(torch.FloatTensor), h_w_exif


def get_image_pairs_filtered(model, fnames, sim_th=0.5, min_pairs=20, all_if_less=20):

    num_imgs = len(fnames)

    if num_imgs <= all_if_less:
        matching_list, h_w_exif = get_img_pairs_all(fnames)
        return matching_list, h_w_exif

    descs, h_w_exif = get_global_desc(model, fnames)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()

    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    for st_idx in range(num_imgs-1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]  
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total+=1
    matching_list = sorted(list(set(matching_list)))

    return matching_list, h_w_exif


MAX_IMAGE_ID = 2**31 - 1


CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""


CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""


CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)


CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""


CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""


CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""


CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"


CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def array_to_blob(array):
    return array.tostring()


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))

    def add_two_view_geometry_new(self, pair_id, matches,
                                  F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))
        
        
def get_focal(height, width, exif):

    max_size = max(height, width)

    focal = None
    if exif is not None:
        focal_35mm = None

        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break

        if focal_35mm is not None:
            focal = focal_35mm / 35. * max_size
    
    if focal is None:
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal


def create_camera(db, height, width, exif, camera_model):
    focal = get_focal(height, width, exif)
    if camera_model == 'simple-pinhole':
        model = 0 # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    if camera_model == 'pinhole':
        model = 1 # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == 'simple-radial':
        model = 2 # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == 'radial':
        model = 3 # radial
        param_arr = np.array([focal, width / 2, height / 2, 0., 0.])
    elif camera_model == 'opencv':
        model = 4 # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])
    return db.add_camera(model, width, height, param_arr)


def add_keypoints(db, feature_dir, h_w_exif, camera_model, single_camera=False):
    keypoint_f = h5py.File(os.path.join(feature_dir, 'keypoints.h5'), 'r')
    camera_id = None
    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename][()]
        if camera_id is None or not single_camera:
            height = h_w_exif[filename]['h']
            width = h_w_exif[filename]['w']
            exif = h_w_exif[filename]['exif']
            camera_id = create_camera(db, height, width, exif, camera_model)
        image_id = db.add_image(filename, camera_id)
        fname_to_id[filename] = image_id
        db.add_keypoints(image_id, keypoints)
    return fname_to_id


def add_matches(db, feature_dir, fname_to_id):

    match_file = h5py.File(os.path.join(feature_dir, 'matches.h5'), 'r')
    added = set()
    for key_1 in match_file.keys():
        group = match_file[key_1]
        for key_2 in group.keys():
            id_1 = fname_to_id[key_1]
            id_2 = fname_to_id[key_2]

            pair_id = (id_1, id_2)
            if pair_id in added:
                warnings.warn(f'Pair {pair_id} ({id_1}, {id_2}) already added')
                continue
            added.add(pair_id)

            matches = group[key_2][()]
            db.add_matches(id_1, id_2, matches)

            
def import_into_colmap(feature_dir, h_w_exif):
    db = COLMAPDatabase.connect(f"{feature_dir}/colmap.db")
    db.create_tables()
    fname_to_id = add_keypoints(db, feature_dir, h_w_exif, camera_model='simple-radial', single_camera=False)
    add_matches(db, feature_dir, fname_to_id)
    db.commit()
    db.close()


def get_unique_idxs(A, dim=0):
    _, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0],device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices


def resize(image, image_size):
    h, w = image.shape[:2]
    aspect_ratio = h/w
    smaller_side_size = int(image_size/max(aspect_ratio, 1/aspect_ratio))
    if aspect_ratio > 1: # H > W
        new_size = (image_size, smaller_side_size)
    else: # H <= W
        new_size = (smaller_side_size, image_size)
    image = cv2.resize(image, new_size[::-1])
    return image, new_size



def extract_superpoint_features(model, cache, filename, img_size):
    """Extract SuperPoint features if not in cache."""
    if 'keypoints' not in cache[filename][img_size]:
        with torch.no_grad():
            prediction = model.superpoint({'image': cache[filename][img_size]['img']})

        cache[filename][img_size] = {**cache[filename][img_size], **{
            'keypoints': torch.stack(prediction['keypoints']),
            'scores': torch.stack(prediction['scores']),
            'descriptors': torch.stack(prediction['descriptors'])
        }}

def run_main_model(model, cache, fname1, fname2, img_size):
    extract_superpoint_features(model, cache, fname1, img_size)
    extract_superpoint_features(model, cache, fname2, img_size)

    data = {
        'image0': cache[fname1][img_size]['img'],
        'image1': cache[fname2][img_size]['img'],
        'keypoints0': cache[fname1][img_size]['keypoints'],
        'keypoints1': cache[fname2][img_size]['keypoints'],
        'scores0': cache[fname1][img_size]['scores'],
        'scores1': cache[fname2][img_size]['scores'],
        'descriptors0': cache[fname1][img_size]['descriptors'],
        'descriptors1': cache[fname2][img_size]['descriptors']
    }

    # Run SuperGlue
    with torch.no_grad():
        pred = model.superglue(data)

    kpts1, kpts2 = cache[fname1][img_size]['keypoints'][0].cpu().numpy(), cache[fname2][img_size]['keypoints'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    valid_matches = matches > -1
    matched_kpts1 = kpts1[valid_matches].astype(np.float32)
    matched_kpts2 = kpts2[matches[valid_matches]].astype(np.float32)

    return matched_kpts1, matched_kpts2



def perform_matching_inference(model, filename1, filename2, cache=None):
    """
    Perform matching inference on two image files.
    
    Args:
        model: The model to use for inference.
        filename1: Path to the first image.
        filename2: Path to the second image.
        cache: Cache dictionary to store image data.
        
    Returns:
        tuple: Matched keypoints in both images and the number of superglue matches.
    """
    # Load and preprocess images
    for filename in [filename1, filename2]:
        if filename not in cache:
            image = cv2.imread(filename, 0)
            height, width = image.shape[:2]
            cache[filename] = {}
            
            for image_size in image_sizes:
                if max(height, width) != image_size:
                    resized_image, (resized_height, resized_width) = resize(image, image_size)
                else:
                    resized_image = image.copy()
                    resized_height, resized_width = resized_image.shape[:2]
                    
                normalized_image = torch.from_numpy(resized_image.astype(np.float32)/255.0).cuda()[None, None].half()
                cache[filename][image_size] = {'img': normalized_image, 'h': height, 'w': width, 'h_r': resized_height, 'w_r': resized_width}

    # Initialize matched keypoints
    matched_keypoints1, matched_keypoints2 = np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    
    # Perform matching inference
    for idx, image_size in enumerate(image_sizes):
        mkpts1_, mkpts2_ = run_main_model(model, cache, filename1, filename2, image_size)
        
        if idx == 0:
            num_superglue_matches = len(mkpts1_)
            
        # Scale keypoints if necessary
        max_height1, max_width1 = cache[filename1][image_size]['h'], cache[filename1][image_size]['w']
        max_height2, max_width2 = cache[filename2][image_size]['h'], cache[filename2][image_size]['w']
        
        if max(max_height1, max_width1) != image_size:
            mkpts1_[:, 0] *= max_width1 / cache[filename1][image_size]['w_r']
            mkpts1_[:, 1] *= max_height1 / cache[filename1][image_size]['h_r']
        if max(max_height2, max_width2) != image_size:
            mkpts2_[:, 0] *= max_width2 / cache[filename2][image_size]['w_r']
            mkpts2_[:, 1] *= max_height2 / cache[filename2][image_size]['h_r']
            
        matched_keypoints1, matched_keypoints2 = np.vstack([matched_keypoints1, mkpts1_]), np.vstack([matched_keypoints2, mkpts2_])
        
        # Return early if no extra matches are needed
        if num_superglue_matches < n_matches:
            return matched_keypoints1, matched_keypoints2, num_superglue_matches

    # ensemble loftr superglue
    


    # Process region of interest if necessary
    if USE_ROI:
        cropped_img1, cropped_img2, shift_xy1, shift_xy2 = extract_crops_via_cluster(filename1, filename2, matched_keypoints1, matched_keypoints2)
        mkpts_crop1, mkpts_crop2 = run_main_model(model, cache, filename1, filename2, image_size)
        
        x1_shift, y1_shift = shift_xy1
        x2_shift, y2_shift = shift_xy2
        
        mkpts_crop1[:, 0] += x1_shift
        mkpts_crop1[:, 1] += y1_shift
        mkpts_crop2[:, 0] += x2_shift
        mkpts_crop2[:, 1] += y2_shift
        
        matched_keypoints1, matched_keypoints2 = np.vstack([matched_keypoints1, mkpts_crop1]), np.vstack([matched_keypoints2, mkpts_crop2])
        
    return matched_keypoints1, matched_keypoints2, num_superglue_matches



def matching_keypoint(model, file_names, pairs, dir):

    keypoint_cache = {}
    match_file = os.path.join(dir, f" matches_{model}.h5")

    # Extract matched keypoints
    with h5py.File(match_file, mode='w') as match_handle:
        for pair_index in tqdm(pairs, desc='Extracting keypoints'):
            file_name1, file_name2 = file_names[pair_index[0]], file_names[pair_index[1]]
            key1, key2 = os.path.basename(file_name1), os.path.basename(file_name2)


            '''
            ensemble loftr + superglue
            '''
            # keypoint1, keypoint2, num_matches = perform_matching_inference(model, file_name1, file_name2, keypoint_cache)
            L_mkpts1, L_mkpts2 = loftr_inference(file_name1, file_name2, 1120) # kornia_max_image_size = 1120
            s_mkpts1, s_mkpts2, num_matches = perform_matching_inference(model, file_name1, file_name2, keypoint_cache)
            keypoint1 = s_mkpts1
            keypoint2 = s_mkpts2
            keypoint1 = np.concatenate((keypoint1, L_mkpts1), axis=0)
            keypoint2 = np.concatenate((keypoint2, L_mkpts2), axis=0)
            num_matches_loftr = len(L_mkpts1)
            num_matches = num_matches + num_matches_loftr
            





            if num_matches >= n_matches:
                keypoint_group = match_handle.require_group(key1)
                data = np.concatenate([keypoint1, keypoint2], axis=1)
                keypoint_group.create_dataset(key2, data=data)

    # Combine keypoints
    keypoints, total_keypoints, match_indices = defaultdict(list), defaultdict(int), defaultdict(dict)
    with h5py.File(match_file, mode='r') as match_handle:
        for key1 in match_handle.keys():
            for key2 in match_handle[key1].keys():
                match_data = match_handle[key1][key2][...]
                keypoints[key1].append(match_data[:, :2])
                keypoints[key2].append(match_data[:, 2:])

                current_indices = torch.arange(len(match_data)).reshape(-1, 1).repeat(1, 2)
                current_indices[:, 0] += total_keypoints[key1]
                current_indices[:, 1] += total_keypoints[key2]
                total_keypoints[key1] += len(match_data)
                total_keypoints[key2] += len(match_data)
                match_indices[key1][key2] = current_indices

    for key in keypoints.keys():
        keypoints[key] = np.round(np.concatenate(keypoints[key], axis=0))

    # Get unique keypoints and match indices
    unique_keypoints, unique_indices = {}, {}
    for key, points in keypoints.items():
        uniq_points, uniq_indices = torch.unique(torch.from_numpy(points.astype(np.float32)), dim=0, return_inverse=True)
        unique_indices[key] = uniq_indices
        unique_keypoints[key] = uniq_points.numpy()

    # Write keypoints to file
    with h5py.File(os.path.join(dir, 'keypoints.h5'), mode='w') as keypoint_handle:
        for key, points in unique_keypoints.items():
            keypoint_handle[key] = points

    # Create output match data
    out_matches = defaultdict(dict)
    for key1, match_group in match_indices.items():
        for key2, match_index in match_group.items():
            match_index_copy = deepcopy(match_index)
            match_index_copy[:, 0] = unique_indices[key1][match_index_copy[:, 0]]
            match_index_copy[:, 1] = unique_indices[key2][match_index_copy[:, 1]]
            matched_keypoints = np.concatenate([unique_keypoints[key1][match_index_copy[:, 0]], unique_keypoints[key2][match_index_copy[:, 1]]], axis=1)
            unique_indices_current = get_unique_idxs(torch.from_numpy(matched_keypoints), dim=0)
            semiclean_match_index = match_index_copy[unique_indices_current]
            semiclean_match_index = semiclean_match_index[get_unique_idxs(semiclean_match_index[:, 0], dim=0)]
            semiclean_match_index = semiclean_match_index[get_unique_idxs(semiclean_match_index[:, 1], dim=0)]
            out_matches[key1][key2] = semiclean_match_index.numpy()

    # Write matches to file
    with h5py.File(match_file, mode='w') as match_handle:
        for key1, match_group in out_matches.items():
            group = match_handle.require_group(key1)
            for key2, match in match_group.items():
                group[key2] = match



def colmap_reconstruction_pipeline(image_directory, feature_directory, image_dimensions_exif,
                                   number_of_exh=number_of_exh, thre_exh=6, matching_name='SuperGlue'):

    # Import features into COLMAP
    import_into_colmap(feature_directory=feature_directory, h_w_exif=image_dimensions_exif)

    # Initialize the main database
    database_path = os.path.join(feature_directory, 'colmap.db')

    # Store matches from all iterations
    all_matches = {}

    # Run exhaustive matching and store results in a temporary database
    for iteration in range(number_of_exh):
        temp_database_path = os.path.join(feature_directory, f'colmap_{iteration}.db')
        shutil.copyfile(database_path, temp_database_path)
        pycolmap.match_exhaustive(temp_database_path)

        db = COLMAPDatabase.connect(temp_database_path)
        matches = np.array(db.execute("SELECT * from two_view_geometries").fetchall())
        for row_index, match in enumerate(matches):
            match_data = np.frombuffer(match[3], dtype=np.uint32).reshape(-1, 2) if match[3] else None
            all_matches.setdefault(row_index, {})[iteration] = match_data
        db.close()

    # Process and add two-view geometries to the main database
    main_db = COLMAPDatabase.connect(database_path)
    for index in all_matches:
        pair_counts = {}
        for iteration in range(number_of_exh):
            if all_matches[index][iteration] is None:
                continue
            for match_ids in all_matches[index][iteration]:
                pair = (match_ids[0], match_ids[1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        filtered_pairs = np.array([pair for pair, count in pair_counts.items() if count >= thre_exh])
        matches_indices = filtered_pairs if filtered_pairs.size > 0 else np.empty((0, 2), dtype=np.uint32)
        main_db.add_two_view_geometry_new(matches[index][0], matches_indices)

    # Commit changes and close the database
    main_db.commit()
    main_db.close()

    # Create output directory for reconstruction results
    output_path = os.path.join(feature_directory, f'colmap_rec_{matching_name}')
    os.makedirs(output_path, exist_ok=True)

    # Run the reconstruction
    mapper_options = pycolmap.IncrementalMapperOptions()
    mapper_options.min_model_size = 3
    reconstruction_results = pycolmap.incremental_mapping(database_path=database_path,
                                                          image_path=image_directory,
                                                          output_path=output_path,
                                                          options=mapper_options)

    return reconstruction_results



def postprocess_images(maps, dataset_name, scene_name):
    """
    Postprocess the images in the given maps, dataset, and scene.
    
    Parameters:
    - maps (dict): A dictionary containing information about images.
    - dataset_name (str): The name of the dataset.
    - scene_name (str): The name of the scene.
    
    Returns:
    dict: A dictionary containing the rotation matrix and translation vector of the post-processed images.
    """
    
    postprocessed_images = {}
    max_images_registered = 0
    best_index = None
    
    # Check if maps is a dictionary, and if so, find the index with the maximum number of registered images
    if isinstance(maps, dict):
        for index, record in maps.items():
            print(index, record.summary())
            
            num_images = len(record.images)
            if num_images > max_images_registered:
                max_images_registered = num_images
                best_index = index

    # If the best index is found, post-process the images at that index
    if best_index is not None:
        print(maps[best_index].summary())
        
        for image in maps[best_index].images.values():
            key = f'{dataset_name}/{scene_name}/images/{image.name}'
            postprocessed_images[key] = {
                "R": image.rotmat(),
                "t": image.tvec
            }

    return postprocessed_images


def arr_to_str(a):
    return ';'.join([str(x) for x in a.reshape(-1)])

import os
results_df = pd.DataFrame(columns=['image_path', 'dataset', 'scene', 'rotation_matrix', 'translation_vector'])


def get_image_directory_paths(input_root, dataset_scene, mode='test'):
    dataset, scene = dataset_scene.split('/')
    image_directory = f"{input_root}/{mode}/{dataset}/{scene}/images"
    return image_directory


def create_feature_directory(data_root, dataset, scene):
    feature_directory = f"{data_root}/featureout/{dataset}/{scene}"
    os.system(f"rm -rf {feature_directory}")
    os.makedirs(feature_directory)
    return feature_directory


def get_image_file_paths(image_directory):
    return sorted(glob(f"{image_directory}/*"))


def execute_matching_pipeline(matching_model, image_file_paths, feature_directory):
    index_pairs, h_w_exif = get_img_pairs_all(fnames=image_file_paths)
    matching_keypoint(matching_model, image_file_paths, index_pairs, feature_directory)
    return h_w_exif


# 对test set进行处理和预测
def process_images(results_df, matching_model, input_root, data_root, datasets_scenes, debug=False):
    for dataset_scene in datasets_scenes:
        image_directory = get_image_directory_paths(input_root, dataset_scene, mode='train' if debug else 'test')
        if not os.path.exists(image_directory):
            continue
        
        feature_directory = create_feature_directory(data_root, *dataset_scene.split('/'))
        image_file_paths = get_image_file_paths(image_directory)
        h_w_exif = execute_matching_pipeline(matching_model, image_file_paths, feature_directory)

        maps = colmap_reconstruction_pipeline(image_directory, feature_directory, h_w_exif)
        results = postprocess_images(maps, *dataset_scene.split('/'))

        # Append results to DataFrame
        for image_path in image_file_paths:
            image_id = '/'.join(image_path.split('/')[-4:])
            if image_id in results:
                R = results[image_id]['R'].reshape(-1)
                T = results[image_id]['t'].reshape(-1)
            else:
                R = np.eye(3).reshape(-1)
                T = np.zeros(3)

            results_df = results_df.append({
                'image_path': image_id,
                'dataset': dataset_scene.split('/')[0],
                'scene': dataset_scene.split('/')[1],
                'rotation_matrix': arr_to_str(R),
                'translation_vector': arr_to_str(T)
            }, ignore_index=True)

    return results_df

results_df = pd.DataFrame(columns=['image_path', 'dataset', 'scene', 'rotation_matrix', 'translation_vector'])
results_df = process_images(results_df, matching_model, INPUT_ROOT, DATA_ROOT, datasets_scenes, DEBUG)


# ## EVAL


EVAL_PART = False


import numpy as np
from dataclasses import dataclass
from time import time

# Evaluation metric.
@dataclass
class Camera:
    rotmat: np.array
    tvec: np.array

def quaternion_from_matrix(matrix):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    # Symmetric matrix K.
    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                  [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                  [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                  [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0

    # Quaternion is eigenvector of K that corresponds to largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)
    return q

def evaluate_R_t(R_gt, t_gt, R, t, eps=1e-15):
    t = t.flatten()
    t_gt = t_gt.flatten()

    q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    GT_SCALE = np.linalg.norm(t_gt)
    t = GT_SCALE * (t / (np.linalg.norm(t) + eps))
    err_t = min(np.linalg.norm(t_gt - t), np.linalg.norm(t_gt + t))
    
    return np.degrees(err_q), err_t



# r = 0.01
# import pandas as pd
#import random
df = results_df
#df_len = len(df)
#asam = random.sample(range(df_len),int(df_len*r))
#for a in asam:
 #   df.loc[a,"rotation_matrix"]= "1.0;0.0;0.0;0.0;1.0;0.0;0.0;0.0;1.0"
  #  df.loc[a,"translation_vector"]= "0.0;0.0;0.0"


df.to_csv(f"{OUTPUT_ROOT}/submission.csv", index=False)