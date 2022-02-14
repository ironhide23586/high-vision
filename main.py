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


from data_utils.data_io import SegmapDataStreamer
import utils
from nets_v2.nn import IrvisNN

utils.MODE = 'train'
utils.SHUFFLE = True


if __name__ == '__main__':
    segmap_data_streamer_train = SegmapDataStreamer(mode='train')

    x, y = segmap_data_streamer_train.get_data_batch()

    segmap_data_streamer_val = SegmapDataStreamer(mode='val')

    irvis_nn = IrvisNN(load_final_model=False, data_feeder=segmap_data_streamer_train,
                       val_data_feeder=segmap_data_streamer_val, init_model=False)
    # irvis_nn.init(run_number=26, step_number=1)

    irvis_nn.train()

