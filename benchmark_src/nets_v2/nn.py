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

import tensorflow as tf


import numpy as np
from keras_unet_collection import models
import utils
from tensorflow.keras.utils import Sequence


class KerasData(Sequence):
    def __init__(self, streamer):
        super().__init__()
        self.streamer = streamer

    def __len__(self):
        return self.streamer.data_feeder.batches_per_epoch

    def __getitem__(self, idx):
        ims, labels = self.streamer.get_data_batch()
        return ims, labels


class InferSamples(tf.keras.callbacks.Callback):

    def __init__(self, writeout_dir, model):
        super().__init__()
        self.writeout_dir = writeout_dir
        self.model = model
        self.im_paths = glob(utils.SAMPLE_IMAGES_DIR + '/*')
        self.infer_step = 0

    def worker(self, suffix=''):
        utils.force_makedir(self.writeout_dir)
        for path in self.im_paths:
            print('Inferring', path)
            im_org = cv2.imread(path)
            ext = '.' + path.split('.')[-1]
            out_viz_fpath = self.writeout_dir + os.sep \
                            + path.split(os.sep)[-1].replace(ext, '_inferred-' + str(self.infer_step) + '.png')
            # im_viz, shadow_mask = self.infer(im_org, viz=True)
            shadow_mask = np.squeeze(self.model.predict(utils.input_infer_preprocess(im_org)))
            # cv2.imwrite(out_viz_fpath, im_viz)
            cv2.imwrite(out_viz_fpath.replace('.png', '-' + suffix + '-mask.png'), shadow_mask * 255)
            print('Output written to', out_viz_fpath)
        self.infer_step += 1

    def on_train_begin(self, logs=None):
        self.worker('train_begin')

    def on_train_end(self, logs=None):
        self.worker('train_end')

    def on_epoch_begin(self, epoch, logs=None):
        self.worker('epoch_begin')

    def on_epoch_end(self, epoch, logs=None):
        self.worker('epoch_end')

    def on_test_begin(self, logs=None):
        self.worker('test_begin')


class IrvisNN:

    def __init__(self, mode='infer'):
        self.mode = mode
        self.update_bn_stats = False
        self.input_tensor = tf.keras.Input(shape=[utils.IM_DIM, utils.IM_DIM, 4], dtype=tf.float32)
        self.n_stack_ups = 2
        self.n_stacks_down = 2
        if mode == 'infer':
            self.load_final_model = True
            self.pred_shadow_mask_logits, self.optimization_vars_and_ops, \
            self.grad_stop_vars_shadow, self.init_ops, \
            self.all_params_tf = models.transunet_2d(self.input_tensor,
                                                     filter_num=[64, 64, 128, 256],
                                                     n_labels=1,
                                                     stack_num_down=self.n_stacks_down, stack_num_up=self.n_stack_ups,
                                                     proj_dim=128,
                                                     num_mlp=128,
                                                     num_heads=2,
                                                     num_transformer=1, activation='ReLU',
                                                     mlp_activation='GELU',
                                                     output_activation=None, batch_norm=True,
                                                     pool='max',
                                                     unpool=False, backbone=None,
                                                     weights=None, freeze_backbone=utils.FREEZE_BACKBONE,
                                                     freeze_batch_norm=not self.update_bn_stats,
                                                     name='transunet')
            self.pred_shadow_mask_probs = tf.nn.sigmoid(self.pred_shadow_mask_logits)
            # self.pred_shadow_mask_probs = self.pred_shadow_mask_logits
            self.model = tf.keras.Model(inputs=self.input_tensor, outputs=self.pred_shadow_mask_probs)
            self.init()

    def init(self):
        load_checkpoint_fpath = os.path.join(utils.FINAL_MODEL_DIR, utils.FINAL_MODEL_NAME)
        self.model.load_weights(load_checkpoint_fpath, by_name=True, skip_mismatch=False)
        print('Loaded model from', load_checkpoint_fpath)

    def __inference_worker(self, ims):
        if ims.shape[0] > 1:
            return self.model.predict_on_batch(ims)
        # return self.model(np.tile(ims, [utils.BATCH_SIZE, 1, 1, 1]), training=False)[:1]
        return self.model(ims, training=False)


    def infer_cloud_cov(self, ims):
        return self.__inference_worker(ims)
