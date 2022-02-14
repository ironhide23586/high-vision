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

import json
import os
import time
from datetime import datetime
from glob import glob

import tensorflow as tf

# if int(tf.__version__.split('.')[0]) > 1:
#     import tensorflow.compat.v1 as tf
#
#     tf.disable_v2_behavior()

# import cv2

from tqdm import tqdm
from keras_unet_collection import models
import utils
from data_utils.data_io import SegmapDataStreamer, StreamerContainer
from tensorflow.keras.utils import Sequence
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


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


# class DistanceError(tf.keras.metrics.Metric):
#
#     def __init__(self, name='distance_error'):
#         super(DistanceError, self).__init__(name=name)
#         # self.input_tensor = in_tensor
#         self.d_err_n = self.add_weight(name='d_error_n', initializer='zeros', dtype=tf.float32)
#         self.d_err = self.add_weight(name='d_error', initializer='zeros', dtype=tf.float32)
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         err = self.distance_error(y_true, y_pred)
#         self.d_err_n.assign_add(1)
#         self.d_err.assign_add(err)
#
#     def reset_state(self):
#         self.d_err = 0
#         self.d_err_n = 0
#
#     def result(self):
#         return self.d_err / self.d_err_n
#
#     def distance_error(self, y_true, y_pred):
#         in_maxes = tf.reduce_max(tf.reduce_max(y_pred[:, :, :, -1], -1), -1)
#         pred_d = tf.pow(y_pred, 0.25) * in_maxes
#         gt_d = tf.pow(y_true, 0.25) * in_maxes
#         err = tf.reduce_mean(tf.linalg.norm(pred_d - gt_d))
#         return err


class IrvisNN:

    def __init__(self, data_feeder=None, val_data_feeder=None, mode=utils.MODE,
                 conf_thresh=utils.CONFIDENCE_THRESHOLD, hard_negative_mining_coeff=utils.HARD_NEGATIVE_MINING_COEFF,
                 update_bn_stats=utils.UPDATE_BATCHNORM_STATS,
                 num_train_steps=utils.NUM_TRAIN_STEPS, base_learn_rate=utils.BASE_LEARN_RATE,
                 lr_exp_decay_power=utils.LEARN_RATE_EXPONENTIAL_DECAY_POWER,
                 model_save_dir_root=utils.MODEL_SAVE_DIR_ROOT, model_name_prefix=utils.MODEL_NAME_PREFIX,
                 load_final_model=False, init_model=False):
        self.mode = mode
        self.update_bn_stats = update_bn_stats
        self.input_tensor = tf.keras.Input(shape=[768, 1024, 5], dtype=tf.float32)
        # self.t = tf.image.resize(tf.identity(self.input_tensor), (1342, 1788))
        self.n_stack_ups = 2
        self.n_stacks_down = 2
        self.model_save_dir_root = model_save_dir_root
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
            self.model = tf.keras.Model(inputs=self.input_tensor, outputs=self.pred_shadow_mask_probs)
            return
        self.data_feeder = data_feeder
        self.keras_feeder = KerasData(data_feeder)
        self.val_data_feeder = None
        self.keras_feeder_val = None
        if val_data_feeder is not None:
            self.val_data_feeder = val_data_feeder
            self.keras_feeder_val = tf.data.Dataset.from_generator(self.get_gen, (tf.float32, tf.float32))
        self.conf_thresh = conf_thresh
        self.hard_negative_mining_coeff = hard_negative_mining_coeff
        self.num_train_steps = num_train_steps
        self.lr_exp_decay_power = lr_exp_decay_power
        self.model_name_prefix = model_name_prefix
        self.load_final_model = load_final_model
        self.num_runs = 0
        if os.path.isdir(self.model_save_dir_root):
            if len(glob(self.model_save_dir_root + '/*')) > 0:
                self.num_runs = max([int(fp.split('-')[-1]) for fp in glob(self.model_save_dir_root + os.sep + '*')]) \
                                + 1
        self.model_save_dirname = 'run-' + str(self.num_runs)
        self.model_save_dirpath = os.sep.join([self.model_save_dir_root, self.model_save_dirname, 'trained_models'])
        self.sample_inference_writeout_dirpath = os.sep.join([self.model_save_dir_root, self.model_save_dirname,
                                                              'sample_inferences'])
        self.__save_suffix = ''
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
                                                 output_activation=tf.nn.relu, batch_norm=True,
                                                 pool='max',
                                                 unpool=False, backbone=None,
                                                 weights=None, freeze_backbone=utils.FREEZE_BACKBONE,
                                                 freeze_batch_norm=not self.update_bn_stats,
                                                 name='transunet')
        # self.pred_shadow_mask_probs = tf.nn.sigmoid(self.pred_shadow_mask_logits)
        # self.pred_shadow_mask_probs = tf.clip_by_value(self.pred_shadow_mask_logits, 0., 1.)
        self.model = tf.keras.Model(inputs=self.input_tensor,
                                    outputs=self.pred_shadow_mask_logits)

        # self.distance_error(self.pred_shadow_mask_logits, self.pred_shadow_mask_logits)

        if not utils.UPDATE_BATCHNORM_STATS:
            print('BatchNorm frozen')
            for l in self.model.layers:
                if 'bn' in l.name:
                    l.trainable = False

        if utils.FREEZE_BACKBONE:
            print('Backbone frozen')
            for l in self.model.layers:
                if '_outercloudviz_' not in l.name:
                    l.trainable = False
        elif utils.FREEZE_DECODER:
            print('Decoder frozen')
            print('Trainable Layers -')
            for l in self.model.layers:
                if '_down0_0' not in l.name:
                    l.trainable = False
                else:
                    print(l.name)

        if self.mode == 'train':
            self.base_lr = base_learn_rate
            self.init_training_graph()
        if init_model:
            self.init()

    def init(self, load_checkpoint_fpath=None, run_number=None, step_number=None):
        if load_checkpoint_fpath is None:
            if not self.load_final_model:
                if run_number is None:
                    run_number = self.num_runs - 1
                prev_dirpath = os.sep.join([self.model_save_dir_root, 'run-' + str(run_number), 'trained_models'])
                if step_number is None:
                    if os.path.isdir(prev_dirpath):
                        fpaths = glob(prev_dirpath + '/*')
                        cts = np.array([os.path.getctime(fp) for fp in fpaths])
                        if cts.shape[0] == 0:
                            print('Using random weights as no model was found at', load_checkpoint_fpath)
                            return
                        load_checkpoint_fpath = fpaths[cts.argmax()]
                    else:
                        load_checkpoint_fpath = os.path.join(utils.FINAL_MODEL_DIR, 'model.ckpt')
                else:
                    paths = glob(prev_dirpath + os.sep + '*.' + str(step_number + 1).zfill(2) + '-*')
                    load_checkpoint_fpath = paths[0]
            else:
                if run_number is None and step_number is None:
                    load_checkpoint_fpath = os.path.join(utils.FINAL_MODEL_DIR, utils.FINAL_MODEL_NAME)
                else:
                    prev_dirpath = os.sep.join([self.model_save_dir_root, 'run-' + str(run_number), 'trained_models'])
                    paths = glob(prev_dirpath + os.sep + '*.' + str(step_number).zfill(2) + '-*')
                    load_checkpoint_fpath = paths[0]
        ps = glob(load_checkpoint_fpath + '*')
        self.load_checkpoint_fpath = load_checkpoint_fpath
        if len(ps) > 0:
            if self.load_final_model:
                load_checkpoint_fpath = ps[0]
            print('💥(☞ﾟヮﾟ)☞ Restoring from checkpoint at', load_checkpoint_fpath, '☜(ﾟヮﾟ☜)')
            self.model.load_weights(load_checkpoint_fpath, by_name=True, skip_mismatch=True)
        else:
            print('Using random weights as no model was found at', load_checkpoint_fpath)

    def get_gen(self):
        for _ in range(self.val_data_feeder.data_feeder.batches_per_epoch):
            yield self.val_data_feeder.get_data_batch()

    def infer_final(self, im_bgr_uint8, shadow_thresh=.5, viz=True):
        if len(im_bgr_uint8.shape) == 4:
            n, h_org, w_org, _ = im_bgr_uint8.shape
            ims = []
            for i in range(n):
                ims.append(utils.input_infer_preprocess(im_bgr_uint8[i])[0])
            im = np.array(ims)
        else:
            h_org, w_org, _ = im_bgr_uint8.shape
            im = utils.input_infer_preprocess(im_bgr_uint8)
        shadow_masks = self.__inference_worker(im)
        ms = []
        vs = []
        i = 0
        for shadow_mask in shadow_masks:
            if shadow_mask.shape[0] != h_org or shadow_mask.shape[1] != w_org:
                m = cv2.resize(shadow_mask.numpy(), (w_org, h_org), interpolation=cv2.INTER_NEAREST)
            else:
                m = shadow_mask.numpy()
            m = np.squeeze(m)
            if viz:
                im_viz = utils.overlay_mask(im_bgr_uint8[i], m, thresh=shadow_thresh)
            else:
                im_viz = None
            ms.append(m)
            vs.append(im_viz)
            i += 1
        ms = np.array(ms)
        vs = np.array(vs)
        return vs, ms

    def __inference_worker(self, ims):
        if ims.shape[0] > 1:
            return self.model.predict_on_batch(ims)
        # return self.model(np.tile(ims, [utils.BATCH_SIZE, 1, 1, 1]), training=False)[:1]
        return self.model(ims, training=False)

    def init_shadow_training_graph(self, y_true, y_pred_):
        pos_indices = tf.squeeze(tf.where(tf.greater(y_true, 0)))
        gt_ds_flattened = tf.reshape(y_true, [-1, ])
        gt_pos_ds_flattened = tf.gather(gt_ds_flattened, pos_indices)
        num_pos_indices = tf.cast(tf.reduce_sum(tf.ones_like(gt_pos_ds_flattened)), tf.float32)

        hard_negative_mining_coeff = 1.

        neg_indices = tf.squeeze(tf.where(tf.less_equal(gt_ds_flattened, 0)))
        gt_neg_ds_flattened_raw = tf.gather(gt_ds_flattened, neg_indices)

        num_negs_raw = tf.cast(tf.cast(tf.reduce_sum(tf.ones_like(gt_neg_ds_flattened_raw)), tf.int32), tf.float32)
        num_neg_indices = tf.cast(tf.clip_by_value(hard_negative_mining_coeff * num_pos_indices, 0., num_negs_raw),
                                  tf.int32)  # 0 is depth
        pred_depths, pred_confs = y_pred_
        depths_flattened = tf.reshape(pred_depths, [-1, ])
        confs_flattened = tf.reshape(pred_confs, [-1, ])
        pred_pos_ds_flattened = tf.gather(depths_flattened, pos_indices, axis=0)
        gt_pos_ds_flattened = tf.gather(gt_ds_flattened, pos_indices, axis=0)

        confs_flattened_pos = tf.gather(confs_flattened, pos_indices, axis=0)
        confs_flattened_neg = tf.gather(confs_flattened, neg_indices, axis=0)

        _, pos_bottomk_indices = tf.nn.top_k(1. - confs_flattened_pos, k=num_pos_indices)
        _, neg_topk_indices = tf.nn.top_k(confs_flattened_neg, k=num_neg_indices)

        pred_pos_confs_flattened = tf.gather(confs_flattened_pos, pos_bottomk_indices, axis=0)
        pred_neg_confs_flattened = tf.gather(confs_flattened_neg, neg_topk_indices, axis=0)

        conf_loss = tf.reduce_mean(1. - pred_pos_confs_flattened) + tf.reduce_mean(pred_neg_confs_flattened)
        d_loss = tf.clip_by_value(tf.abs((pred_pos_ds_flattened - gt_pos_ds_flattened)) / 5., 0., 1.)
        l = (conf_loss + d_loss) / 2.
        return l

    def distance_error(self, y_true, y_pred):
        depths, confs = y_pred
        d_pred = tf.reshape(depths, [-1, ])
        d_gt = tf.reshape(y_true, [-1, ])
        pos_indices = tf.squeeze(tf.where(tf.greater(y_true, 0)))
        pred_d = tf.gather(d_pred, pos_indices)
        gt_d = tf.gather(d_gt, pos_indices)
        err = tf.reduce_mean(tf.abs(pred_d - gt_d))
        return err

    def init_training_graph(self):
        self.model.compile(optimizer='adam', loss=self.init_shadow_training_graph, metrics=[self.distance_error])

    def train(self):
        exp_dec = tf.keras.optimizers.schedules.ExponentialDecay(self.base_lr, self.num_train_steps,
                                                                 self.lr_exp_decay_power)
        lr_sc = tf.keras.callbacks.LearningRateScheduler(exp_dec)
        save_fpath = os.path.join(self.model_save_dirpath, self.model_name_prefix) \
                     + '-weights.{epoch:02d}-{precision:.2f}-{recall:.2f}.hdf5'
        utils.force_makedir(self.model_save_dirpath)

        saver_keras = tf.keras.callbacks.ModelCheckpoint(save_fpath, save_freq='epoch',
                                                         monitor='loss', save_best_only=False,
                                                         save_weights_only=True)
        num_epochs = int(np.round(utils.NUM_TRAIN_STEPS / self.keras_feeder.streamer.data_feeder.batches_per_epoch))
        logdir = os.sep.join([self.model_save_dir_root, 'logs'])
        utils.force_makedir(logdir)
        logdir += os.sep + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False, write_images=True,
                                                              histogram_freq=1,
                                                              update_freq=utils.PRINT_LOSS_EVERY_N_STEPS)
        infer_sample_callback = InferSamples(self.sample_inference_writeout_dirpath, self.model)
        self.model.fit(self.keras_feeder, callbacks=[lr_sc, saver_keras, tensorboard_callback],
                       steps_per_epoch=self.keras_feeder.streamer.data_feeder.batches_per_epoch,
                       epochs=num_epochs, validation_data=self.keras_feeder_val)

    def die(self):
        print('Killing Data Streamer...')
        self.data_feeder.die()
        print('Dead ☠')
        print('Killing Session...')
        self.sess.close()
        print('Dead')

    def save(self, save_fpath=None, train_step=None, save_suffix=None):
        if train_step is None:
            train_step = self.train_step_tensor
        if save_suffix is None:
            save_suffix = self.__save_suffix
        if save_fpath is None:
            save_fpath = os.path.join(self.model_save_dirpath, self.model_name_prefix) + '-' + save_suffix
            utils.force_makedir(self.model_save_dirpath)
        # self.model_reader.save(self.sess, save_fpath, global_step=train_step)
        self.model.save_weights(save_fpath, overwrite=False)
        print('The checkpoint has been created with prefix -', save_fpath)
