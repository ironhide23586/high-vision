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


from PIL import Image

import numpy as np
from tqdm import tqdm

from nets_v2.nn import IrvisNN
from data_utils.data_io import SegmapDataStreamer
import utils

utils.MODE = 'test'
# OUT_DIR = '/codeexecution/predictions'
OUT_DIR = '../scratchspace/preds'


if __name__ == '__main__':
    segmap_data_streamer_test = SegmapDataStreamer(mode='test')
    irvis_nn = IrvisNN(mode='infer')
    utils.force_makedir(OUT_DIR)
    print('Inferring test images...')
    for i in tqdm(range(segmap_data_streamer_test.data_feeder.batches_per_epoch)):
        x, y_gt, ids = segmap_data_streamer_test.get_data_batch()
        y = np.squeeze(irvis_nn.infer_cloud_cov(x))
        y_ = np.zeros_like(y).astype(np.uint8)
        y_[y >= .5] = 1
        chip_pred_im = Image.fromarray(y_)
        chip_pred_im.save(OUT_DIR + '/' + ids[0] + '.tif')
        k = 0





