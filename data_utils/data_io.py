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
from glob import glob

import cv2
# import rioxarray
import numpy as np


from data_utils import async_data_reader
import utils


class SBU:

    def __init__(self, dirpath, shuffle=utils.SHUFFLE, mode=None, train_frac=.8):
        train_dir_map = {'train': '', 'val': 'validate', 'test': 'test'}
        if mode is None:
            mode = utils.MODE
        self.mode = mode
        self.train_frac = train_frac
        self.root_data_dir = dirpath
        print('---> Loading Dataset from', self.root_data_dir)
        self.images_dir_prefix = os.sep.join([self.root_data_dir, train_dir_map['train'] + 'pred'])
        self.labels_dir_prefix = os.sep.join([self.root_data_dir, train_dir_map['train'] + 'gt'])
        dirs = glob(self.labels_dir_prefix + os.sep + '*')
        self.gt_depth_fpaths = np.hstack([glob(d + os.sep + '*') for d in dirs])
        self.pred_depth_fpaths = np.array([p.replace('gt', 'pred').replace('.tif', '_flow2.pfm')
                                           for p in self.gt_depth_fpaths])
        self.im_fpaths = np.array([p.replace('gt', 'pred').replace('.tif', '.jpg') for p in self.gt_depth_fpaths])
        filt = np.array([os.path.exists(self.gt_depth_fpaths[i]) and
                         os.path.exists(self.pred_depth_fpaths[i]) and
                         os.path.exists(self.im_fpaths[i]) for i in range(self.gt_depth_fpaths.shape[0])])
        self.gt_depth_fpaths = self.gt_depth_fpaths[filt]
        self.pred_depth_fpaths = self.pred_depth_fpaths[filt]
        self.im_fpaths = self.im_fpaths[filt]

        self.shuffle = shuffle
        if shuffle:
            if not os.path.exists(utils.IDX_FPATH + '.npy'):
                idx = np.arange(self.im_fpaths.shape[0])
                np.random.shuffle(idx)
                np.save(utils.IDX_FPATH, idx)
            else:
                idx = np.load(utils.IDX_FPATH + '.npy')
            self.im_fpaths = self.im_fpaths[idx]
        n = self.im_fpaths.shape[0]
        if train_dir_map[mode] == 'train':
            self.im_fpaths = self.im_fpaths[:int(train_frac * n)]
        elif train_dir_map[mode] == 'validate':
            self.im_fpaths = self.im_fpaths[int(train_frac * n):]
        # self.im_ids = np.array([fp.split(os.sep)[-1] for fp in self.im_fpaths])
        # self.mask_fpaths = np.array([self.labels_dir_prefix + os.sep + id + '.tif' for id in self.im_ids])
        # legit_idx = [os.path.isfile(fp) for fp in self.mask_fpaths]
        # self.im_fpaths = self.im_fpaths[legit_idx]
        # self.mask_fpaths = self.mask_fpaths[legit_idx]
        # self.im_ids = self.im_ids[legit_idx]
        self.epoch_size = self.im_fpaths.shape[0]

    def get_label(self, idx):
        # mask = rioxarray.open_rasterio(self.mask_fpaths[idx]).data.squeeze()
        gt_depth = cv2.imread(self.gt_depth_fpaths[idx], cv2.IMREAD_ANYDEPTH)
        y = gt_depth / self.depth_max
        return y

    def viz_depth_rb(self, d_):
        d = d_ - d_.min()
        d = d / d.max()
        d = np.tile(np.expand_dims(d, -1), [1, 1, 3])
        od = ([255, 0, 0] * d) + ([0, 0, 255] * (1. - d))
        return od

    def get_image(self, idx):
        im = cv2.imread(self.im_fpaths[idx]) / 255.
        pred_depth = cv2.imread(self.pred_depth_fpaths[idx], cv2.IMREAD_ANYDEPTH)
        self.pred_depth_min = pred_depth.min()
        depth = pred_depth - self.pred_depth_min
        self.depth_max = depth.max()
        depth = depth / self.depth_max
        h, w, _ = im.shape
        x = np.zeros([h, w, 4]).astype(np.float)
        x[:, :, :3] = im
        x[:, :, -1] = depth
        return x

    def get_gt_viz(self, idx):
        im_viz = self.__gen_viz(idx)
        return im_viz

    def v(self, idx):
        im = self.get_gt_viz(idx)
        cv2.imwrite('v.jpg', im)

    def __gen_viz(self, idx):
        im = self.get_image(idx).copy()
        mask = self.get_label(idx)
        try:
            polys, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except:
            _, polys, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        im_viz = cv2.drawContours(im, polys, -1, (255, 255, 255), 3)
        return im_viz


class SegmapIngestion:

    def __init__(self, dataset, h, w, random_crop=False, random_rotate=False, random_color_perturbations=False):
        self.dataset = dataset
        self.h = h
        self.w = w
        self.random_crop = random_crop
        self.random_rotate = random_rotate
        self.random_color_perturbations = random_color_perturbations
        self.mode = 'val'
        if self.random_crop or self.random_rotate or self.random_color_perturbations:
            self.mode = 'train'

    def get_data_train_format(self, idx):
        im = self.dataset.get_image(idx)
        mask = self.dataset.get_label(idx)
        im_ret, mask_ret = self.preprocess(im, mask)

        # dir = 'scratchspace/sample_train'
        # utils.force_makedir(dir)
        # cv2.imwrite(dir + '/' + self.mode + '-' + str(idx) + '-x.png', utils.nn_unpreprocess(im_ret))
        # cv2.imwrite(dir + '/' + self.mode + '-' + str(idx) + '-y.png', mask_ret * 255)
        return im_ret, mask_ret

    def preprocess(self, im_in, mask_in):
        if im_in is None:
            return None, None
        org_h, org_w, _ = im_in.shape
        r = 1. * self.w / self.h
        random_rotate = self.random_rotate
        random_crop = self.random_crop
        random_color_perturbations = self.random_color_perturbations
        if np.random.random() < .3 or self.mode == 'val':
            random_rotate = False
            random_crop = False
            random_color_perturbations = False
        if random_rotate:
            angle = np.random.uniform(0, 359.99)
            m = cv2.getRotationMatrix2D((org_w // 2, org_h // 2), angle, 1.4)
            im = cv2.warpAffine(im_in, m, (org_w, org_h))
            mask = cv2.warpAffine(mask_in, m, (org_w, org_h))
            mask[mask < 128] = 0
            mask[mask >= 128] = 255
        else:
            im = im_in
        if random_crop:
            h = np.random.randint(int(.5 * org_h), org_h)
            w = int(r * h)
            start_w, end_w = utils.get_random_crop_ends(org_w, w)
            start_h, end_h = utils.get_random_crop_ends(org_h, h)
            im = cv2.resize(im[start_h:end_h, start_w:end_w],
                            (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask[start_h:end_h, start_w:end_w].astype(np.float32),
                              (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        else:
            # im = utils.resize_aspect_ratio_preserved(im_in, min(self.h, self.w), interp=cv2.INTER_LINEAR)
            # mask = utils.resize_aspect_ratio_preserved(mask_in, min(1280, 1280), interp=cv2.INTER_NEAREST)
            im = im_in.copy()
            mask = mask_in.copy()
        if random_color_perturbations:
            delta = np.random.randint(-45, 45)
            im = np.clip(im.astype(np.float) + delta, 0, 255).astype(np.uint8)
            color_vec = np.random.randint(-18, 18, size=3)
            im = np.clip(im.astype(np.float) + color_vec, 0, 255).astype(np.uint8)
        # cv2.imwrite('v.jpg', utils.overlay_mask(im, mask))
        im = utils.nn_preprocess(im)
        # mask[mask > 0] = 1.
        return im, mask


class SegmapDataStreamer:

    def __init__(self, h=utils.IM_DIM, w=utils.IM_DIM, shuffle=utils.SHUFFLE, mode=None,
                 batch_size=utils.BATCH_SIZE):
        self.num_streamers = 1
        gt_dir = utils.SHADOW_GT_DIR
        dataset = SBU(gt_dir, shuffle=shuffle, mode=mode)
        rc = False
        rr = False
        rp = False
        if mode != 'train':
            rc = False
            rr = False
            rp = False
        irvis_nn_ingestor = SegmapIngestion(dataset, h=h, w=w, random_crop=rc, random_rotate=rr,
                                            random_color_perturbations=rp)
        # x, y = irvis_nn_ingestor.get_data_train_format(0)
        self.data_feeder = async_data_reader.TrainFeeder(irvis_nn_ingestor, batch_size=batch_size)

    def get_data_batch(self):
        data_batch = self.data_feeder.dequeue()
        return data_batch

    def die(self):
        self.data_feeder.die()


class StreamerContainer:

    def __init__(self, streamers, random_sample=False):
        self.streamers = streamers
        self.random_sample = random_sample
        self.num_streamers = len(self.streamers)
        self.current_streamer_idx = -1
        self.streamer = None

    def get_data_batch(self, streamer_idx=None):
        if self.random_sample:
            self.current_streamer_idx = np.random.randint(self.num_streamers)
        else:
            if streamer_idx is None:
                if self.current_streamer_idx + 1 >= self.num_streamers:
                    self.current_streamer_idx = 0
                else:
                    self.current_streamer_idx += 1
            else:
                self.current_streamer_idx = streamer_idx
        self.streamer = self.streamers[self.current_streamer_idx]
        return self.streamer.get_data_batch()

    def die(self):
        for streamer in self.streamers:
            streamer.die()
