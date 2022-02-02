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

from nets_v2.nn import IrvisNN
from data_utils.data_io import SegmapDataStreamer
import utils

utils.MODE = 'test'


if __name__ == '__main__':
    segmap_data_streamer_test = SegmapDataStreamer(mode='test')
    irvis_nn = IrvisNN(mode='infer')

    for i in range(segmap_data_streamer_test.data_feeder.batches_per_epoch):
        x, _ = segmap_data_streamer_test.get_data_batch()
        y = irvis_nn.infer_cloud_cov(x)
        k = 0





