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

import tensorflow as tf

from keras_unet_collection.models import transunet_2d
import utils


class IrvisNN:

    def __init__(self):
        self.input_tensor = tf.keras.Input(shape=[utils.IM_DIM, utils.IM_DIM, 4], dtype=tf.float32)
        self.n_stack_ups = 2
        self.n_stacks_down = 2
        self.pred_shadow_mask_logits = transunet_2d(self.input_tensor,
                                                    filter_num=[64, 64, 128, 256],
                                                    n_labels=1,
                                                    stack_num_down=self.n_stacks_down,
                                                    stack_num_up=self.n_stack_ups,
                                                    proj_dim=128,
                                                    num_mlp=128,
                                                    num_heads=2,
                                                    num_transformer=1, activation='ReLU',
                                                    mlp_activation='GELU',
                                                    output_activation=tf.nn.relu, batch_norm=True,
                                                    pool='max', unpool=False, name='transunet')
        self.pred_shadow_mask_probs = tf.nn.sigmoid(self.pred_shadow_mask_logits)
        self.model = tf.keras.Model(inputs=self.input_tensor, outputs=self.pred_shadow_mask_probs)
        self.init()

    def init(self):
        load_checkpoint_fpath = os.path.join(utils.FINAL_MODEL_DIR, utils.FINAL_MODEL_NAME)
        self.model.load_weights(load_checkpoint_fpath, by_name=True, skip_mismatch=False)
        print('Loaded model from', load_checkpoint_fpath)

    def __inference_worker(self, ims):
        return self.model(ims, training=False)

    def infer_cloud_cov(self, ims):
        return self.__inference_worker(ims)
