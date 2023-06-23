'''
SuperPoint和SuperGlue都是计算机视觉领域中的神经网络模型，由ETH Zurich的研究团队开发。
SuperPoint是一个特征点检测与描述模型，而SuperGlue则是一个基于特征点匹配的三维重建模型。

SuperPoint是一个基于卷积神经网络（CNN）的模型，可以检测图像中的特征点，并生成每个特征点的描述符。
通过SuperPoint，我们可以在图像中找到最重要的点并将其与其他图像进行匹配。
SuperPoint采用的是一种叫做“分层空间金字塔”的方法，可以处理尺度变化和旋转变化等。SuperPoint的性能非常好，可以在速度和精度之间取得很好的平衡。

SuperGlue则是一个基于SuperPoint的特征点匹配模型，可以将SuperPoint生成的描述符用于计算两幅图像之间的匹配。
通过SuperGlue，我们可以将两幅图像中的点进行一一匹配，并且计算它们的几何关系。这个几何关系可以用于三维重建，即通过两幅图像匹配来推断物体在三维空间中的位置。

综上所述，SuperPoint和SuperGlue是两个相互补充的模型，其中SuperPoint可以用于特征点检测和描述符生成，而SuperGlue可以用于特征点匹配和三维重建。
通过这两个模型，我们可以实现基于图像的三维重建。
'''





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

# 将/kaggle/input目录添加到系统路径中，以便能够导入该目录下的模块。
sys.path.append('/kaggle/input')
# 将/tmp目录添加到系统路径中，以便能够导入该目录下的模块。
sys.path.append('/tmp')
# sglib.models.matching模块中导入Matching类，用于执行图像匹配任务。
from sglib.models.matching import Matching

# 设置输入数据的根目录。
INPUT_ROOT = '/kaggle/input/image-matching-challenge-2023'
# 设置数据处理后的根目录。
DATA_ROOT = '/kaggle/data'
# 设置输出结果的根目录。
OUTPUT_ROOT = '/kaggle/working'

DEBUG = False


# 创建一个空列表，用于存储数据集和场景。
datasets_scenes = []
# 读取样本提交文件，并将其存储为Pandas数据帧。
sample_submission_df = pd.read_csv(f"{INPUT_ROOT}/sample_submission.csv")
# 遍历样本提交数据帧中的每一行。
for _, r in sample_submission_df[['dataset', 'scene']].iterrows():
    # 获取当前行的数据集和场景，并将它们合并为一个字符串。
    ds = f"{r.dataset}/{r.scene}"
    # 如果当前数据集和场景的字符串不在列表中，则执行以下操作。
    if ds not in datasets_scenes:
        # 将当前数据集和场景的字符串添加到列表中。
        datasets_scenes.append(ds)


# 设置图像匹配模型的名称为SuperGlue。
matching_name = 'SuperGlue'
# 设置图像的大小
image_sizes = [1088, 1280]
# image_sizes = [1280,1472]
# image_sizes = [1152,1472]



'''
在这段代码中，num_exhaustives 是指进行多少次匹配。此处的代码通过复制原始数据库 database_path，然后对每个副本进行匹配，最终将所有匹配结果存储在 all_matches 中。
这里的目的是通过多次匹配来提高匹配的精度。

如果想提高匹配的精度，可以考虑增加 num_exhaustives 的值，即增加匹配的次数。但是，需要注意的是，过多的匹配可能会导致运行时间变长，同时也可能会导致匹配结果的质量变差。
因此，需要在匹配次数和匹配质量之间做出平衡。



thresh_exhaustives 是一个阈值，影响着从所有可能的匹配中筛选出符合要求的匹配的数量。
具体来说，如果一个匹配对的得分（即 v）大于等于 thresh_exhaustives，那么这个匹配对就会被保留下来，否则就会被过滤掉。

提高 thresh_exhaustives 的值可以使得从所有可能的匹配中筛选出的匹配对更加稳定、可靠，从而提高图像匹配的精度。
但是，如果将 thresh_exhaustives 的值过高设置，可能会导致一些正确的匹配对被过滤掉，从而降低匹配精度。
'''
number_of_exh = 8
thre_exh = 6

USE_ROI = False
ROI_SIZE = 1024



''''
sim_th是一个相似度阈值，用于筛选匹配的图像对。只有相似度大于等于该阈值的图像对才会被选入候选列表中。
sim_th越大，筛选出来的图像对越严格，匹配精度也越高。如果希望提高匹配精度，可以适当增大sim_th的值。

min_pairs是每个图像选择的最小匹配数。即每个图像至少匹配的图像数量。这个参数可以用于控制候选列表中的图像数量。如果希望更多的候选图像，可以适当增大min_pairs的值。

exhaustive_if_less是一个阈值参数，用于决定是否采用全局匹配。当候选列表中的图像对数量小于该阈值时，会使用全局匹配方法来获取更多的图像对。
如果该值设置得太小，会增加计算量，降低匹配精度；而如果设置得太大，则会减少匹配精度。建议根据实际情况来调整该参数。

device是指定计算设备的参数，默认值为cpu。如果有GPU可用，可以将该参数设置为cuda，以加速计算。  
'''
sim_th = None
n_matches = 100



'''
可以通过以下方式修改 superglue 参数以提高图像匹配精度：

将 'nms_radius' 参数从原来的 4 减小到 2，这将使得关键点更加稠密，提高匹配精度。
将 'keypoint_threshold' 参数从原来的 0.005 增加到 0.02，这将使得筛选出的关键点更加可靠，提高匹配精度。
将 'sinkhorn_iterations' 参数从原来的 20 减小到 5，这将加快 Sinkhorn 迭代算法的收敛速度，提高匹配精度。
将 'match_threshold' 参数从原来的 0.2 减小到 0.05，这将使得匹配更加严格，提高匹配精度。
需要注意的是，修改参数可能会对算法的其他方面产生影响，因此需要根据具体情况进行调整。
'''
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


# 这段代码是一个用于调整图像大小的函数，下面是每行代码的注释：
# 定义函数，输入为图像和最大图像尺寸
def resize_img(img, max_image_size):
    # 如果最大图像尺寸为-1
    if max_image_size == -1:
        return img, 1.0 # no resize  则返回原图和缩放比例为1
    # 计算缩放比例，使图像的较长边缩放到最大图像尺寸大小
    scale = max_image_size / max(img.shape[0], img.shape[1]) 
    # 计算缩放后的图像宽度
    w = int(img.shape[1] * scale)
    # 计算缩放后的图像高度
    h = int(img.shape[0] * scale)
    # 调整图像大小
    img = cv2.resize(img, (w, h))
    return img, scale


# 这段代码定义了一个名为 get_crop_img 的函数，该函数接受两个参数：图像路径和一组关键点（mkpts）。
# 图像裁剪
def get_crop_img(img_path, mkpts):
    # 在函数的开头，进行了一个简单的检查，以确保 mkpts 中至少有10个点，否则函数将返回空值。
    if len(mkpts) < 10:  # sanity check
        return None, None
    # 然后，通过OpenCV的 imread() 函数读取图像，并获取图像的高度和宽度。
    img = cv2.imread(img_path)
    im_h, im_w, _ = img.shape

    # 接下来，找到 mkpts 中最小和最大的 x 和 y 坐标，并将它们分别存储在 min_x，min_y，max_x 和 max_y 变量中。
    min_x, min_y = np.amin(mkpts[:, 0]), np.amin(mkpts[:, 1])
    max_x, max_y = np.amax(mkpts[:, 0]), np.amax(mkpts[:, 1])
    # 然后，这些坐标被用来计算出需要截取的图像的左、上、右和下的像素坐标。
    left, top, right, bottom = min_x, min_y, max_x, max_y

    # 为了确保能截取到关键点周围的上下文信息，pad 像素被添加到左、上、右和下的坐标。
    # 类似深度学习中的padding
    pad = 4
    x = max(0, int(left - pad))
    xr = min(im_w-1, math.ceil(right + pad))
    y = max(0, int(top - pad))
    yb = min(im_h-1, math.ceil(bottom + pad))
    # 使用这些计算出来的坐标，可以通过切片操作来截取需要的图像。
    crop_img = img[y:yb, x:xr]

    # 最后，返回截取的图像和一个元组，该元组包含左上角的偏移量 (x, y)，以便在原始图像中重新定位截取的图像。
    h_crop, w_crop = crop_img.shape[:2]
    if min(h_crop, w_crop) < 10:
        return None, None
    shift_xy = (x, y)
    return crop_img, shift_xy

# 这段代码实现了基于 DBSCAN 算法的密集聚类。
# 定义函数 db_scan_new，该函数接受一个名为 mkpts 的参数，
    # 以及两个可选参数 min_samples 和 max_dst，它们分别代表最小样本数和最大距离阈值。
    # 默认情况下，min_samples 为 5，max_dst 为 40。
def db_scan_new(mkpts, min_samples=5, max_dst=40):
    # min_samples = 6  # round(len(mkpt1) * 0.8)
    # max_dst = 40  # maximum distance between two samples
    # 使用 DBSCAN 算法进行聚类，其中 eps 参数设置为 max_dst，min_samples 参数设置为 min_samples。
    # 将算法应用于 mkpts 数据集，并将结果存储在 db 变量中。
    db = DBSCAN(eps=max_dst, min_samples=min_samples).fit(mkpts)
    # # 获取每个点的聚类标签，存储在 labels 变量中。
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    # 获取聚类数量，但会忽略噪声点（标签为 -1）。
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # 获取噪声点数量。
    n_noise_ = list(labels).count(-1)
    # 如果聚类数量小于 1，即没有聚类，则返回 None。
    if n_clusters_ < 1:
        return None

    # 过滤掉噪声点，得到所有非噪声点的聚类标签。
    filtered_labels = [x for x in labels if x != -1]
    # 统计每个聚类中的点数。
    unique, counts = np.unique(filtered_labels, return_counts=True)

    # 设置一个阈值，用于过滤掉点数过少的聚类。
    T = 0.2
    # 存储所有密集聚类的点的索引。
    all_idxs = []
    for lbl_idx in np.argsort(counts)[::-1]:
        if counts[lbl_idx] / counts.max() >= T:
            # 如果该聚类的点数占所有聚类点数的比例大于等于阈值，则将该聚类的点的索引加入 all_idxs。
            idxs = np.argwhere(filtered_labels == lbl_idx).flatten()
            all_idxs.extend(idxs)
    # 对所有密集聚类的点的索引进行排序。
    all_idxs = np.array(sorted(all_idxs))
    # 根据索引获取所有密集聚类的点。
    dense_mkpts = mkpts[all_idxs]
    # 返回所有密集聚类的点作为结果。
    return dense_mkpts

# 该函数的作用是从两张图像中提取密集匹配点，并基于这些匹配点裁剪原始图像，返回裁剪后的图像和它们的偏移量。
# 传入参数为两张图像的路径和它们的特征点列表。
def extract_crops_via_cluster(im1_path, im2_path, mkpts0, mkpts1):
    #dense_mkpts0, dense_mkpts1 = db_scan(mkpts0, mkpts1)

    # 读取第一张图像的尺寸，计算出图像的缩放比例，和缩放后的距离阈值。
    im_h, im_w, _ = cv2.imread(im1_path).shape
    px_scale_factor = max(1.0, max(im_h, im_w) / 1024)
    px_scaled_dst = int(40 * px_scale_factor)

    # 使用基于密度的聚类算法 DBSCAN 提取每张图像的密集匹配点。
    dense_mkpts0 = db_scan_new(mkpts0, min_samples=5, max_dst=px_scaled_dst)
    dense_mkpts1 = db_scan_new(mkpts1, min_samples=5, max_dst=px_scaled_dst)
    # 如果没有提取到密集匹配点，则返回 None。
    if dense_mkpts0 is None or dense_mkpts1 is None:
        return None, None, None, None


    # crop by dense matches
    # 基于提取到的密集匹配点，使用 get_crop_img 函数来进行图像裁剪，并得到裁剪后的图像和它们的偏移量。
    cropped_img1, shift_xy1 = get_crop_img(im1_path, dense_mkpts0)
    cropped_img2, shift_xy2 = get_crop_img(im2_path, dense_mkpts1)
    return cropped_img1, cropped_img2, shift_xy1, shift_xy2

# 定义函数名为get_img_pairs_all，接收一个参数fnames，表示文件名列表。
def get_img_pairs_all(fnames):
    # 初始化一个空列表index_pairs和一个空字典h_w_exif。
    index_pairs, h_w_exif = [], {}
    # 使用range()函数生成索引，循环遍历每个文件名。
    for i in range(len(fnames)):
        # 使用Python Imaging Library（PIL）的Image.open()函数打开文件名为fnames[i]的图像文件，并将该图像对象赋值给变量img。
        img = Image.open(fnames[i])
        # 使用img.size属性获取图像的宽度和高度，并将它们赋值给变量w和h。
        w, h = img.size
        # 将一个新的字典作为值添加到h_w_exif字典中，字典的键是文件名的最后一个斜杠后面的字符串，值包含图像的高度、宽度和Exif信息。
        h_w_exif[fnames[i].split('/')[-1]] = {'h': h, 'w': w, 'exif': img._getexif()}
        for j in range(i+1, len(fnames)):
            index_pairs.append((i,j))
    return index_pairs, h_w_exif

# 这段代码定义了一个名为get_global_desc的函数，该函数接受两个参数：model和fnames。
# 该函数的作用是使用相似性模型获取全局特征。
def get_global_desc(model, fnames):
    # 调用resolve_data_config函数，该函数返回一个配置字典，用于指定数据的处理方式。
    # 该函数的第一个参数为空字典，第二个参数为model。
    config = resolve_data_config({}, model=model)
    # 调用create_transform函数，该函数返回一个数据转换器，用于将数据转换为模型所需的格式。
    # 该函数的参数是config字典。
    transform = create_transform(**config)
    global_descs_convnext, h_w_exif = [], {}
    # 使用for循环遍历fnames列表中的每个文件名，并使用tqdm库显示进度条。
    # 该循环的描述为“Get global features using similarity model”。
    for fname in tqdm(fnames, desc='Get global features using similarity model'):
        # 将图像转换为RGB格式，并使用transform转换器将其转换为模型所需的格式。
        # 然后将其转换为CUDA张量，并使用half()方法将其转换为半精度浮点数。
        img = Image.open(fname)
        w, h = img.size
        h_w_exif[fname.split('/')[-1]] = {'h': h, 'w': w, 'exif': img._getexif()}
        timg = transform(img.convert('RGB')).unsqueeze(0).cuda().half()
        # 使用torch.no_grad()上下文管理器，禁用梯度计算。
        # 然后使用model.forward_features方法计算当前图像的特征向量，并使用mean方法计算其平均值。
        # 最后使用view方法将其形状改为(1, -1)
        with torch.no_grad():
            desc = model.forward_features(timg.cuda().half()).mean(dim=(-1,2))
            desc = desc.view(1, -1)
            # 使用F.normalize函数对特征向量进行归一化，使其范数为2。
            desc_norm = F.normalize(desc, dim=1, p=2)
        global_descs_convnext.append(desc_norm.detach().cpu())
    # 使用torch.cat函数将所有特征向量连接成一个张量，作为函数的返回值。
    # 同时，将h_w_exif字典也作为返回值返回。
    global_descs_all = torch.cat(global_descs_convnext, dim=0)
    return global_descs_all.type(torch.FloatTensor), h_w_exif


# 这段代码实现一个函数，根据输入的模型、图像文件名、相似度阈值、最小匹配对数以及是否匹配所有图像，返回图像对的列表。
'''
model: 模型
fnames: 图像文件名列表
sim_th：相似度阈值，默认为0.5
min_pairs：最小匹配对数，默认为20
all_if_less：如果图像数小于等于此值，就返回所有图像对
'''
def get_image_pairs_filtered(model, fnames, sim_th=0.5, min_pairs=20, all_if_less=20):

    num_imgs = len(fnames)
    # 如果图像数量小于等于 all_if_less，则直接调用 get_img_pairs_all 函数返回所有的图像对。
    if num_imgs <= all_if_less:
        matching_list, h_w_exif = get_img_pairs_all(fnames)
        return matching_list, h_w_exif

    # 使用给定的模型，提取图像的全局特征描述，然后计算特征描述之间的欧几里得距离（距离矩阵），并将其转换为 numpy 类型。
    descs, h_w_exif = get_global_desc(model, fnames)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()

    '''
    对于每个起始图像索引 st_idx，根据相似度阈值生成一个布尔掩码 mask，
    然后根据掩码生成一个匹配图像列表 to_match，如果匹配图像列表的数量小于最小匹配对数，
    就从当前图像和所有其他图像的距离矩阵 dm 中选择前 min_pairs 个最相似的图像作为匹配图像。
    然后，对于每个匹配图像索引 idx，检查它是否等于起始图像索引 st_idx，如果不是，则将它们作为一个元组添加到匹配列表 matching_list 中，
    最后将匹配列表排序并去重。
    '''
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

    # 返回匹配列表以及每个图像的宽度和高度以及 EXIF 信息。
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


# 用于将两个图像ID转换为一对ID
def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    # 其中，MAX_IMAGE_ID是一个常量，表示图像ID的最大值。
    return image_id1 * MAX_IMAGE_ID + image_id2

# 函数array.tostring()是将一个数组（array）转换为字符串（string）的方法。
# 该方法将数组中的每一个元素都转换为字符串，然后用逗号连接成一个字符串返回。
# 需要注意到，该方法只能用于一维数组，如果数组是多维的，则需要先使用flatten()方法将其转换为一维数组。
def array_to_blob(array):
    return array.tostring()


'''
# 这段代码是一个用于操作COLMAP数据库的脚本，下面对其进行注释和解释：
# 
# COLMAPDatabase 类是一个继承自 sqlite3.Connection 的自定义类，用于操作 COLMAP 数据库。它包含了一系列方法来添加相机、图像、关键点、描述符、匹配等数据到数据库中。
# 
# IS_PYTHON3 变量用于检查是否运行在 Python 3 及以上的版本上。
# 
# MAX_IMAGE_ID 变量定义了图像ID的最大值，用于在生成匹配对ID时进行计算。
# 
# CREATE_CAMERAS_TABLE、CREATE_DESCRIPTORS_TABLE、CREATE_IMAGES_TABLE、CREATE_TWO_VIEW_GEOMETRIES_TABLE、CREATE_KEYPOINTS_TABLE、CREATE_MATCHES_TABLE 和 CREATE_NAME_INDEX 是用于创建数据库表的 SQL 语句。
# 
# image_ids_to_pair_id 函数用于将两个图像ID转换为匹配对ID。
# 
# pair_id_to_image_ids 函数用于将匹配对ID转换为两个图像ID。
# 
# array_to_blob 函数用于将数组转换为二进制数据。
# 
# blob_to_array 函数用于将二进制数据转换为数组。
# 
# add_camera 方法用于向数据库中添加相机信息。
# 
# add_image 方法用于向数据库中添加图像信息。
# 
# add_keypoints 方法用于向数据库中添加关键点信息。
# 
# add_descriptors 方法用于向数据库中添加描述符信息。
# 
# add_matches 方法用于向数据库中添加匹配信息。
# 
# add_two_view_geometry 方法用于向数据库中添加两视图几何信息。
# 
# 该脚本提供了一组用于操作 COLMAP 数据库的方法，用于添加相机、图像、关键点、描述符、匹配等数据到数据库中，并提供了一些辅助函数用于数据转换和ID转换。
'''
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
        

'''
# get_focal 函数用于从图像文件中获取焦距信息。它接受图像文件路径 image_path 和一个可选的 err_on_default 参数，默认为 False。
# 函数打开图像文件，并获取其尺寸信息。然后，尝试从图像的EXIF数据中获取焦距信息。如果成功获取到焦距，根据相对于35mm胶片的焦距进行换算，
# 得到相对于图像尺寸的焦距。如果无法获取焦距信息，根据默认的先验值计算一个焦距。最后，返回计算得到的焦距。
'''
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


'''
#create_camera 函数用于向数据库中添加相机信息。它接受一个数据库对象 db，图像文件路径 image_path 和相机模型 camera_model。
#函数打开图像文件，并获取图像的宽度和高度。然后，调用 get_focal 函数获取图像的焦距。根据相机模型类型，选择相应的相机模型ID和参数数组。
#最后，调用数据库的 add_camera 方法添加相机信息，并返回相机ID。
'''
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


'''
#add_keypoints 函数用于向数据库中添加关键点信息。它接受一个数据库对象 db，HDF5文件路径 h5_path，图像文件路径 image_path，
#图像文件扩展名 img_ext，相机模型 camera_model 和一个可选的 single_camera 参数，默认为 True。

#函数打开关键点的HDF5文件，并遍历文件中的每个关键点数据。对于每个关键点数据，构建关键点文件名（包括扩展名），
#并与图像文件路径进行拼接得到图像文件的完整路径。如果图像文件不存在，抛出异常。

#如果是第一次添加关键点数据或者不使用单一相机模型，调用 create_camera 函数创建相机，并返回相机ID。然后，
#调用数据库的 add_image 方法添加图像信息，并将图像文件名与图像ID进行映射保存。最后，调用数据库的 add_keypoints 方法添加关键点信息。

#返回关键点文件名到图像ID的映射字典 fname_to_id。
'''
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



#add_matches 函数用于向数据库中添加匹配信息。它接受一个数据库对象 db，HDF5文件路径 h5_path 和图像文件名到图像ID的映射字典 fname_to_id。

#函数打开匹配的HDF5文件，并遍历文件中的每对匹配数据。对于每对匹配数据，获取对应的图像ID。使用图像ID计算匹配对ID，
#并检查是否已经添加过该匹配对。如果已经添加过，则发出警告并跳过。

#如果匹配对没有被添加过，则获取匹配数据，并调用数据库的 add_matches 方法添加匹配信息。

#最后，更新进度条并完成匹配添加。

#该代码用于将DISK的关键点和匹配结果添加到COLMAP的数据库中，以便进行后续的三维重建。

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


# import_into_colmap函数用于将特征导入到COLMAP数据库中。
# 它连接到数据库，创建必要的表格，并将特征从特征目录中的HDF5文件导入到数据库中。  
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


# 这段代码实现了对输入图像进行等比例缩放的功能
'''
其中，image是输入的原始图像，image_size是缩放后的目标尺寸。

首先，通过shape方法获取输入图像的高度和宽度，并计算出输入图像的宽高比aspect_ratio。

接着，根据宽高比和目标尺寸计算出较小的一边的尺寸smaller_side_size。

然后，根据输入图像的高宽比选择新的尺寸new_size，如果高大于宽，则将高缩放至目标尺寸，宽按比例缩放；如果宽大于等于高，则将宽缩放至目标尺寸，高按比例缩放。

最后，调用OpenCV的resize函数对输入图像进行缩放，并返回缩放后的图像和新的尺寸。
'''
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


# 这是一个用于从图像中提取SuperPoint特征并存储到缓存中的函数。
def extract_superpoint_features(model, cache, filename, img_size):
    """Extract SuperPoint features if not in cache."""
    # 判断缓存中是否已经存储了该图像的特征信息
    if 'keypoints' not in cache[filename][img_size]:
        with torch.no_grad():
            # 使用输入的模型对象model对指定图像进行特征提取，返回一个包含“关键点”、“分数”和“描述子”的字典。
            prediction = model.superpoint({'image': cache[filename][img_size]['img']})
        
        # 把提取出的“关键点”、“分数”和“描述子”存储在缓存中。
        cache[filename][img_size] = {**cache[filename][img_size], **{
            'keypoints': torch.stack(prediction['keypoints']),
            'scores': torch.stack(prediction['scores']),
            'descriptors': torch.stack(prediction['descriptors'])
        }}


# 这段代码主要是运行一个视觉SLAM算法中使用的模型，并提取两张图像的SuperPoint特征，然后使用SuperGlue算法进行匹配。
# 定义一个函数，该函数接受一个模型、一个缓存、两个文件名以及一个图像大小作为输入参数。
def run_main_model(model, cache, fname1, fname2, img_size):
    # 使用给定的模型从文件fname1,fname2中提取SuperPoint特征，并缓存到cache中。
    extract_superpoint_features(model, cache, fname1, img_size)
    extract_superpoint_features(model, cache, fname2, img_size)

    # 创建一个包含从缓存中读取的两张图像的SuperPoint特征的字典，以及它们之间的匹配。
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
        # 使用SuperGlue算法进行特征匹配，并将结果存储在pred变量中。
        pred = model.superglue(data)

    # 从缓存中获取两张图像的关键点（keypoint）和匹配。
    kpts1, kpts2 = cache[fname1][img_size]['keypoints'][0].cpu().numpy(), cache[fname2][img_size]['keypoints'][0].cpu().numpy()
    # 从预测的matches中获取第一张图像中的匹配关系，并将其存储在匹配变量中。
    matches = pred['matches0'][0].cpu().numpy()
    # 找到具有有效匹配的关键点。
    valid_matches = matches > -1
    # 将匹配到的关键点从第一张图像中提取出来，并将其存储在变量matched_kpts1中。
    matched_kpts1 = kpts1[valid_matches].astype(np.float32)
    # 将与第一张图像中的关键点匹配的第二张图像中的关键点提取出来，并将其存储在变量matched_kpts2中。
    matched_kpts2 = kpts2[matches[valid_matches]].astype(np.float32)

    return matched_kpts1, matched_kpts2



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
def perform_matching_inference(model, filename1, filename2, cache=None):
    # Load and preprocess images
    # 遍历两张图片的文件名
    for filename in [filename1, filename2]:
        # 如果文件名不在缓存中
        if filename not in cache:
            # 读取图片
            image = cv2.imread(filename, 0)
            # 获取图片高度和宽度
            height, width = image.shape[:2]
            # 初始化缓存字典
            cache[filename] = {}
            
            # 遍历参数列表中的所有图像尺寸
            for image_size in image_sizes:
                # 如果图像的最大尺寸不等于当前尺寸
                if max(height, width) != image_size:
                    resized_image, (resized_height, resized_width) = resize(image, image_size)
                else:
                    resized_image = image.copy()
                    resized_height, resized_width = resized_image.shape[:2]
                
                # 将图像归一化并转换为 PyTorch Tensor 格式
                normalized_image = torch.from_numpy(resized_image.astype(np.float32)/255.0).cuda()[None, None].half()
                # 将图像存储到字典中
                cache[filename][image_size] = {'img': normalized_image, 'h': height, 'w': width, 'h_r': resized_height, 'w_r': resized_width}

    # Initialize matched keypoints
    # 初始化匹配的关键点为空的 numpy 数组
    matched_keypoints1, matched_keypoints2 = np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    
    # Perform matching inference
    # 遍历所有图像尺寸
    '''
    这段代码中的for循环的作用是对不同尺寸的图像进行匹配。
    由于不同尺寸的图像可能会有不同的特征点分布和数量，因此需要对每个尺寸的图像进行单独的匹配，以获取更好的匹配结果。
    在循环中，每次处理一个尺寸的图像，调用run_main_model函数进行匹配，并将匹配结果添加到matched_keypoints1和matched_keypoints2中。
    最后，如果匹配的关键点数量已经达到了n_matches的数量，就直接返回当前的匹配结果，否则继续处理下一个尺寸的图像。
    '''
    for idx, image_size in enumerate(image_sizes):
        # 运行模型进行匹配
        mkpts1_,mkpts2_ = run_main_model(model, cache, filename1, filename2, image_size)
        
        # 判断当前是否是第一个处理的图像尺寸。如果是，将num_superglue_matches设置为mkpts1_的长度，即第一张图像的关键点数量。
        if idx == 0:
            num_superglue_matches = len(mkpts1_)
            
        # Scale keypoints if necessary
        # 获取第一张图像和第二张图像在当前尺寸下的最大高度和宽度。
        max_height1, max_width1 = cache[filename1][image_size]['h'], cache[filename1][image_size]['w']
        max_height2, max_width2 = cache[filename2][image_size]['h'], cache[filename2][image_size]['w']
        
        # 判断第一张和第二张图像是否需要缩放。如果需要，将mkpts1_，mkpts2_中的关键点坐标按比例缩放。
        if max(max_height1, max_width1) != image_size:
            mkpts1_[:, 0] *= max_width1 / cache[filename1][image_size]['w_r']
            mkpts1_[:, 1] *= max_height1 / cache[filename1][image_size]['h_r']
        if max(max_height2, max_width2) != image_size:
            mkpts2_[:, 0] *= max_width2 / cache[filename2][image_size]['w_r']
            mkpts2_[:, 1] *= max_height2 / cache[filename2][image_size]['h_r']
            
        # 将当前尺寸下的匹配关键点添加到matched_keypoints1和matched_keypoints2中。
        matched_keypoints1, matched_keypoints2 = np.vstack([matched_keypoints1, mkpts1_]), np.vstack([matched_keypoints2, mkpts2_])
        
        # Return early if no extra matches are needed
        if num_superglue_matches < n_matches:
            return matched_keypoints1, matched_keypoints2, num_superglue_matches
            
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

            keypoint1, keypoint2, num_matches = perform_matching_inference(model, file_name1, file_name2, keypoint_cache)

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



    """
    Postprocess the images in the given maps, dataset, and scene.
    
    Parameters:
    - maps (dict): A dictionary containing information about images.
    - dataset_name (str): The name of the dataset.
    - scene_name (str): The name of the scene.
    
    Returns:
    dict: A dictionary containing the rotation matrix and translation vector of the post-processed images.
    """
def postprocess_images(maps, dataset_name, scene_name):
    
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
# 主要功能是针对给定的多个数据集（在datasets_scenes参数中传递）中的图像进行处理，生成它们的特征表示并使用colmap算法进行重建。
# 定义函数process_images，
# 它有5个参数：results_df数据帧，matching_model匹配模型，input_root和data_root是输入和输出数据的根目录，
# datasets_scenes是要处理的数据集列表，debug是一个布尔变量，指示是否在调试模式下运行（默认为False）。
def process_images(results_df, matching_model, input_root, data_root, datasets_scenes, debug=False):
    '''
    循环遍历数据集列表，对每个数据集场景进行处理。
    使用get_image_directory_paths函数来获取该数据集场景中的图像目录路径。
    如果该目录不存在，就跳过该场景的处理，进入下一个场景。
    '''
    for dataset_scene in datasets_scenes:
        image_directory = get_image_directory_paths(input_root, dataset_scene, mode='train' if debug else 'test')
        if not os.path.exists(image_directory):
            continue
        

        '''
        如果图像目录存在，则使用create_feature_directory函数创建该场景的特征目录。
        使用get_image_file_paths函数获取该场景内所有图像文件的路径列表。
        使用execute_matching_pipeline函数计算所有图像的特征向量，并执行匹配操作以计算相机之间的变换矩阵。
        使用colmap_reconstruction_pipeline函数对数据集场景中的所有图像进行三维重建，生成某些信息的maps。
        最后使用postprocess_images函数对maps进行后处理，生成最终的结果
        '''
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




df = results_df
df_len = len(df)
df.to_csv(f"{OUTPUT_ROOT}/submission.csv", index=False)