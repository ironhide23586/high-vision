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

utils.MODE = 'train'
utils.SHUFFLE = True


if __name__ == '__main__':
    segmap_data_streamer_train = SegmapDataStreamer(mode='train')
    segmap_data_streamer_val = SegmapDataStreamer(mode='val')

    irvis_nn = IrvisNN(load_final_model=True, data_feeder=segmap_data_streamer_train,
                       val_data_feeder=segmap_data_streamer_val, init_model=True)
    # irvis_nn.init(run_number=0, step_number=3)

    irvis_nn.train()

