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

# import cv2
import numpy as np
import rioxarray

from data_utils import async_data_reader
import utils


class SBU:

    def __init__(self, mode):
        train_dir_map = {'train': 'train', 'val': 'validate', 'test': 'test'}
        self.root_data_dir = utils.SHADOW_GT_DIR
        print('---> Loading Dataset from', self.root_data_dir)
        self.images_dir_prefix = os.sep.join([self.root_data_dir, train_dir_map[mode] + '_features'])
        self.labels_dir_prefix = os.sep.join([self.root_data_dir, train_dir_map[mode] + '_labels'])
        self.im_fpaths = np.array(glob(self.images_dir_prefix + os.sep + '*'))
        self.im_ids = np.array([fp.split(os.sep)[-1] for fp in self.im_fpaths])
        self.mask_fpaths = np.array([self.labels_dir_prefix + os.sep + id + '.tif' for id in self.im_ids])
        self.epoch_size = self.im_ids.shape[0]

    def get_label(self, idx):
        mask = None
        if os.path.exists(self.mask_fpaths[idx]):
            # mask = cv2.imread(self.mask_fpaths[idx], cv2.IMREAD_ANYDEPTH)
            mask = rioxarray.open_rasterio(self.mask_fpaths[idx]).data.squeeze()
        return mask

    def get_image(self, idx):
        fps = glob(self.im_fpaths[idx] + '/*')
        ims = []
        for fp in fps:
            im = rioxarray.open_rasterio(fp).data.squeeze()
            # im = cv2.imread(fp, cv2.IMREAD_ANYDEPTH)
            if im is None:
                return None
            ims.append(im)
        return np.rollaxis(np.array(ims), 0, 3), self.im_fpaths[idx].split(os.sep)[-1]


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
        im, im_id = self.dataset.get_image(idx)
        mask = self.dataset.get_label(idx)
        im_ret, mask_ret = self.preprocess(im, mask)
        if len(im_ret.shape) == 3:
            im_ret = np.expand_dims(im_ret, 0)
        return im_ret, mask_ret, im_id

    def preprocess(self, im_in, mask_in):
        if im_in is None:
            return None, None
        im = utils.nn_preprocess(im_in)
        return im, mask_in


class SegmapDataStreamer:

    def __init__(self, h=utils.IM_DIM, w=utils.IM_DIM, mode=None):
        self.num_streamers = 1
        dataset = SBU(mode=mode)
        rc = False
        rr = False
        rp = False
        self.ingestor = SegmapIngestion(dataset, h=h, w=w, random_crop=rc, random_rotate=rr,
                                        random_color_perturbations=rp)
        self.epoch_size_total = self.ingestor.dataset.epoch_size
        self.batches_per_epoch = self.epoch_size_total // utils.BATCH_SIZE
        print('Total number of images =', self.epoch_size_total)
        # x, y, id = self.ingestor.get_data_train_format(0)
        # self.data_feeder = async_data_reader.TrainFeeder(ingestor, batch_size=batch_size)

    def get_data(self, idx):
        return self.ingestor.get_data_train_format(idx)


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
