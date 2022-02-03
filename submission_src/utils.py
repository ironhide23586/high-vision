"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""

import os
from queue import Queue

# import cv2
import numpy as np

MODE = 'test'  # choose between train and val
BATCH_SIZE = 1
SHUFFLE = True
PRINT_LOSS_EVERY_N_STEPS = 50
IDX_FPATH = 'train_val_indices'

FREEZE_BACKBONE = False
FREEZE_DECODER = False

IM_DIM = 512
# SHADOW_GT_DIR = '../../driven-data/cloud-cover'
SHADOW_GT_DIR = '/codeexecution/data'
FINAL_MODEL_DIR = 'final_model'
FINAL_MODEL_NAME = 'flonet-weights.01-0.73-0.97.hdf5'

BATCHES_PER_ASYNC_QUEUE = 10
CONFIDENCE_THRESHOLD = .5
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def split_to_backprop_and_update(upsampler_params):
    upsampler_backprop_vars = [v for v in upsampler_params if 'moving' not in v.name]
    upsampler_moving_stats_vars = [v for v in upsampler_params if 'moving' in v.name]
    return upsampler_backprop_vars, upsampler_moving_stats_vars


def set_gpu_id(id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(id)


def get_center_crop_ends(dim, center_size):
    if dim != center_size:
        start_idx = int((dim - center_size) // 2)
        end_idx = int(start_idx + center_size)
    else:
        start_idx = 0
        end_idx = dim
    return start_idx, end_idx


def get_random_crop_ends(dim, center_size):
    if dim != center_size:
        start_idx = int(np.random.randint(dim - center_size))
        end_idx = int(start_idx + center_size)
    else:
        start_idx = 0
        end_idx = dim
    return start_idx, end_idx


def get_rescaled_dims(w, h, min_dim_sz):
    if h > w:
        scale_factor = 1. * h / w
        new_h = int(min_dim_sz * scale_factor)
        new_w = min_dim_sz
    else:
        scale_factor = 1. * w / h
        new_h = min_dim_sz
        new_w = int(min_dim_sz * scale_factor)
    return new_w, new_h


def overlay_mask(im_in, conf_mask_in, color=(255, 0, 0), alpha=.5, thresh=.5):
    if conf_mask_in.max() > 1.:
        conf_mask = conf_mask_in / 255.
    else:
        conf_mask = conf_mask_in
    im = im_in.copy()
    f = conf_mask > thresh
    p = np.array(color) * np.tile(np.expand_dims(conf_mask[f], 0), [3, 1]).T
    im[f] = alpha * p + (1 - alpha) * im[f]
    im = im.astype(np.uint8)
    return im


# def resize_aspect_ratio_preserved(im, min_dim_sz=720, interp=cv2.INTER_NEAREST):
#     h, w = im.shape[0], im.shape[1]
#     new_w, new_h = get_rescaled_dims(w, h, min_dim_sz=min_dim_sz)
#     im_ret = cv2.resize(im, (new_w, new_h), interpolation=interp)
#     return im_ret


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.tile([np.linalg.norm(vector, axis=1)], [2, 1]).T


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    v = np.sum(v2_u * v1_u, axis=1)
    return np.arccos(np.clip(v, -1.0, 1.0))


def nn_preprocess(im_in_):  # re-implemented what tensorflow was doing internally for NASnet.
    m = im_in_.max()
    if m > 0:
        im = im_in_ / m
    else:
        im = im_in_
    im = im - .5
    im = im * 2.
    return im


def input_infer_preprocess(im_bgr_uint8, side=IM_DIM):
    h, w, _ = im_bgr_uint8.shape
    if min(h, w) != side:
        im = resize_aspect_ratio_preserved(im_bgr_uint8, side, interp=cv2.INTER_LINEAR)
    else:
        im = im_bgr_uint8
    im = nn_preprocess(im)
    im = np.expand_dims(im, 0)
    return im


def force_makedir(dir):
    if not os.path.isdir(dir):
        print('Making folder at -', dir)
        os.makedirs(dir)


def topk_idx(v, k):
    return np.argpartition(v, -k)[-k:]


def bottomk_idx(v, k):
    return np.argpartition(v, k)[:k]
