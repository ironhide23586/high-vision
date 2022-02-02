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

    def __init__(self, dirpath, shuffle=utils.SHUFFLE, mode=None):
        # train_dir_map = {'train': 'train', 'val': 'validate', 'test': '/codeexecution/data/test'}
        train_dir_map = {'train': 'train', 'val': 'validate', 'test': 'train'}
        if mode is None:
            mode = utils.MODE
        self.mode = mode
        self.root_data_dir = dirpath
        print('---> Loading Dataset from', self.root_data_dir)
        self.images_dir_prefix = os.sep.join([self.root_data_dir, train_dir_map[mode] + '_features'])
        self.labels_dir_prefix = os.sep.join([self.root_data_dir, train_dir_map[mode] + '_labels'])
        self.im_fpaths = np.array(glob(self.images_dir_prefix + os.sep + '*'))
        if mode != 'train':
            shuffle = False
        self.shuffle = shuffle
        if shuffle:
            if not os.path.exists(utils.IDX_FPATH + '.npy'):
                idx = np.arange(self.im_fpaths.shape[0])
                np.random.shuffle(idx)
                np.save(utils.IDX_FPATH, idx)
            else:
                idx = np.load(utils.IDX_FPATH + '.npy')
            self.im_fpaths = self.im_fpaths[idx]
        # n = self.im_fpaths.shape[0]
        self.im_ids = np.array([fp.split(os.sep)[-1] for fp in self.im_fpaths])
        self.mask_fpaths = np.array([self.labels_dir_prefix + os.sep + id + '.tif' for id in self.im_ids])
        # legit_idx = [os.path.isfile(fp) for fp in self.mask_fpaths]
        # self.im_fpaths = self.im_fpaths[legit_idx]
        # self.mask_fpaths = self.mask_fpaths[legit_idx]
        # self.im_ids = self.im_ids[legit_idx]
        self.epoch_size = self.im_ids.shape[0]

    def get_label(self, idx):
        mask = rioxarray.open_rasterio(self.mask_fpaths[idx]).data.squeeze()
        return mask

    def get_image(self, idx):
        fps = glob(self.im_fpaths[idx] + '/*')
        # c = 0
        ims = []
        for fp in fps:
            im = rioxarray.open_rasterio(fp).data.squeeze()
            # im = cv2.imread(fp, cv2.IMREAD_ANYDEPTH)
            if im is None:
                return None
            ims.append(im)
            # cv2.imwrite(str(c) + '.png', (im / im.max()) * 255)
            # c += 1
        return np.rollaxis(np.array(ims), 0, 3)

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
        # mask = self.dataset.get_label(idx)
        mask = None
        im_ret, mask_ret = self.preprocess(im, mask)

        # dir = 'scratchspace/sample_train'
        # utils.force_makedir(dir)
        # cv2.imwrite(dir + '/' + self.mode + '-' + str(idx) + '-x.png', utils.nn_unpreprocess(im_ret))
        # cv2.imwrite(dir + '/' + self.mode + '-' + str(idx) + '-y.png', mask_ret * 255)
        return im_ret, mask_ret

    def preprocess(self, im_in, mask_in):
        if im_in is None:
            return None, None
        im = utils.nn_preprocess(im_in)
        return im, mask_in


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
        x, y = irvis_nn_ingestor.get_data_train_format(0)
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
