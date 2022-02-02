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


import cv2
import numpy as np


from nets_v2.nn import IrvisNN
from data_utils.data_io import SegmapDataStreamer
import utils

utils.MODE = 'test'


if __name__ == '__main__':
    segmap_data_streamer_test = SegmapDataStreamer(mode='test')
    irvis_nn = IrvisNN(mode='infer')

    for i in range(segmap_data_streamer_test.data_feeder.batches_per_epoch):
        x, y_gt = segmap_data_streamer_test.get_data_batch()

        cv2.imwrite('../scratchspace/x_' + str(i) + '.png', (np.squeeze(x)[:, :, :3] + 1) * 128)
        cv2.imwrite('../scratchspace/gt_' + str(i) + '.png', y_gt[0] * 255)

        y = irvis_nn.infer_cloud_cov(x)

        y = np.squeeze(y)
        y_ = y**6
        # y_[y_ < .89] = 0
        cv2.imwrite('../scratchspace/pred2_' + str(i) + '.png', y_ * 255)


        k = 0





