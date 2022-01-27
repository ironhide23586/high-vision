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

import sys

import utils
utils.MODE = 'train'
utils.SHUFFLE = True

from nets_v2.nn import IrvisNN
from data_utils.data_io import SegmapDataStreamer


if __name__ == '__main__':
    if len(sys.argv) > 1:
        utils.set_gpu_id(sys.argv[1])  # GPU ID

    segmap_data_streamer_train = SegmapDataStreamer(mode='train')
    # print('+++++++++++++++')
    segmap_data_streamer_val = SegmapDataStreamer(mode='val')
    # data_streamer = StreamerContainer([segmap_data_streamer])

    irvis_nn = IrvisNN(load_final_model=False, data_feeder=segmap_data_streamer_train,
                       val_data_feeder=segmap_data_streamer_val, init_model=False)
    # irvis_nn.init(run_number=0, step_number=0)
    irvis_nn.init(run_number=16, step_number=92)

    irvis_nn.train()
