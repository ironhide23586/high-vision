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

import cv2
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from keras_unet_collection import models
import utils
from data_utils.data_io import SegmapDataStreamer, StreamerContainer
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

    def __init__(self, data_feeder=None, val_data_feeder=None, mode=utils.MODE,
                 conf_thresh=utils.CONFIDENCE_THRESHOLD, hard_negative_mining_coeff=utils.HARD_NEGATIVE_MINING_COEFF,
                 update_bn_stats=utils.UPDATE_BATCHNORM_STATS,
                 num_train_steps=utils.NUM_TRAIN_STEPS, base_learn_rate=utils.BASE_LEARN_RATE,
                 lr_exp_decay_power=utils.LEARN_RATE_EXPONENTIAL_DECAY_POWER,
                 model_save_dir_root=utils.MODEL_SAVE_DIR_ROOT, model_name_prefix=utils.MODEL_NAME_PREFIX,
                 load_final_model=False, init_model=False):
        self.mode = mode
        self.update_bn_stats = update_bn_stats
        self.input_tensor = tf.keras.Input(shape=[utils.IM_DIM, utils.IM_DIM, 4], dtype=tf.float32)
        self.n_stack_ups = 4
        self.n_stacks_down = 4
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
                                                 output_activation=None, batch_norm=True,
                                                 pool='max',
                                                 unpool=False, backbone=None,
                                                 weights=None, freeze_backbone=utils.FREEZE_BACKBONE,
                                                 freeze_batch_norm=not self.update_bn_stats,
                                                 name='transunet')
        self.pred_shadow_mask_probs = tf.nn.sigmoid(self.pred_shadow_mask_logits)

        self.model = tf.keras.Model(inputs=self.input_tensor, outputs=self.pred_shadow_mask_probs)
        # for l in self.model.layers:
        #     if 'denoiser' not in l.name:
        #         l.trainable = False

        if not utils.UPDATE_BATCHNORM_STATS:
            print('BatchNorm frozen')
            for l in self.model.layers:
                if 'bn' in l.name:
                    l.trainable = False

        if utils.FREEZE_BACKBONE:
            print('Backbone frozen')
            for l in self.model.layers:
                if '_expander' not in l.name:
                    l.trainable = False
        elif utils.FREEZE_DECODER:
            print('Decoder frozen')
            for l in self.model.layers:
                if '_expander' in l.name:
                    l.trainable = False

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
                    paths = glob(prev_dirpath + os.sep + '*.' + str(step_number + 1).zfill(2) + '-*')
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

    def get_trainer_hard_negative_miner_graph(self, gt_mask_tensor, pred_mask_tensor, hard_negative_mining_coeff,
                                              conf_thresh=.5):
        # n = utils.IM_DIM * utils.IM_DIM
        gts_flattened = tf.reshape(gt_mask_tensor, [-1, ])
        pos_indices = tf.squeeze(tf.where(tf.greater(gts_flattened, conf_thresh)))
        gt_pos_confs_flattened = tf.gather(gts_flattened, pos_indices)
        num_pos_indices = tf.cast(tf.reduce_sum(input_tensor=gt_pos_confs_flattened), tf.float32)

        neg_indices_raw = tf.squeeze(tf.where(tf.less_equal(gts_flattened, conf_thresh)))
        gt_neg_confs_flattened_raw = tf.gather(gts_flattened, neg_indices_raw)
        num_negs_raw = tf.cast(tf.cast(tf.reduce_sum(gt_neg_confs_flattened_raw + 1), tf.int32), tf.float32)
        num_neg_indices = tf.cast(tf.clip_by_value(hard_negative_mining_coeff * num_pos_indices,
                                                   0., num_negs_raw), tf.int32)

        preds_flattened = tf.reshape(pred_mask_tensor, [-1, ])
        pred_pos_confs_flattened = tf.gather(preds_flattened, pos_indices, axis=0)

        preds_flattened_neg = tf.gather(preds_flattened, neg_indices_raw)
        _, neg_topk_indices = tf.nn.top_k(preds_flattened_neg, k=num_neg_indices)
        pred_neg_confs_flattened = tf.gather(preds_flattened_neg, neg_topk_indices, axis=0)
        gt_neg_confs_flattened = tf.gather(tf.gather(gts_flattened, neg_indices_raw), neg_topk_indices)
        # self.tmp = [num_pos_indices, num_negs_raw, num_neg_indices,
        #             gt_pos_confs_flattened, gt_neg_confs_flattened,
        #             pred_pos_confs_flattened, pred_neg_confs_flattened,
        #             preds_flattened, pos_indices, neg_topk_indices]
        return gt_pos_confs_flattened, gt_neg_confs_flattened, pred_pos_confs_flattened, pred_neg_confs_flattened, \
               pos_indices

    def init_shadow_training_graph(self, y_true, y_pred):
        # gt_pos_shadows_flattened, gt_neg_shadows_flattened, pred_pos_shadows_flattened, pred_neg_shadows_flattened, \
        # pos_indices = self.get_trainer_hard_negative_miner_graph(y_gt, y_pred,
        #                                                          utils.HARD_NEGATIVE_MINING_COEFF)
        # pred_shadow_loss_logits = tf.concat([pred_pos_shadows_flattened, pred_neg_shadows_flattened], 0)
        # gt_shadow_loss_logits = tf.concat([gt_pos_shadows_flattened, gt_neg_shadows_flattened], 0)

        # total_shadow_loss = tf.reduce_mean(1. - pred_pos_shadows_flattened) \
        #                     + tf.reduce_mean(pred_neg_shadows_flattened)
        # total_shadow_loss = tf.keras.losses.binary_crossentropy(gt_shadow_loss_logits, pred_shadow_loss_logits)
        # total_shadow_loss = losses.tversky_loss(y_gt, tf.round(y_pred))

        y_true_pos = tf.reshape(y_true, [-1, ])
        y_pred_pos = tf.reshape(y_pred, [-1, ])
        true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
        false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
        alpha = utils.FOCAL_TVERSKY_FALSE_NEGATIVE_COEFF
        smooth = 1.
        sc = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
        # total_shadow_loss = tf.pow((1. - sc), utils.FOCAL_TVERSKY_POWER)
        total_shadow_loss = 1. - sc
        return total_shadow_loss

    def init_training_graph(self):
        self.model.compile(optimizer='adam', loss=self.init_shadow_training_graph, metrics=['Precision', 'Recall'])

    def eval_step(self, data_feeder):
        ims, labels_gt = data_feeder.get_data_batch()
        _, h, w, _ = ims.shape
        shadow_mask_pred = self.__inference_worker(ims)
        shadow_mask_gt = labels_gt[0]
        shadow_mask_pred_threshed = np.zeros_like(shadow_mask_pred)
        shadow_mask_pred_threshed[shadow_mask_pred > .5] = 1
        shadow_gt, shadow_pred = shadow_mask_gt.flatten(), shadow_mask_pred_threshed.flatten()
        shadow_prec, shadow_rec, shadow_fsc, _ = precision_recall_fscore_support(shadow_gt, shadow_pred)
        shadow_prec = shadow_prec[1]
        shadow_rec = shadow_rec[1]
        shadow_fsc = shadow_fsc[1]
        shadow_stats = np.array([shadow_prec, shadow_rec, shadow_fsc])
        return shadow_stats

    def eval(self):
        print('Evaluating...')
        shadows = []
        for i in range(self.data_feeder.num_streamers):
            for _ in tqdm(range(self.data_feeder.streamers[i].data_feeder.batches_per_epoch)):
                # for _ in tqdm(range(10)):  # FOR DEBUG ONLY
                shadow_stats = self.eval_step(self.data_feeder.streamers[i])
                shadows.append(shadow_stats)
        shadows = np.array(shadows)
        shadow_prf = shadows.mean(axis=0)
        return shadow_prf

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
        logdir = os.sep.join([self.model_save_dir_root, self.model_save_dirname, 'logs'])
        utils.force_makedir(logdir)
        logdir += os.sep + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False, write_images=True,
                                                              histogram_freq=1,
                                                              update_freq=utils.PRINT_LOSS_EVERY_N_STEPS)
        infer_sample_callback = InferSamples(self.sample_inference_writeout_dirpath, self.model)
        self.model.fit(self.keras_feeder, callbacks=[lr_sc, saver_keras, tensorboard_callback, infer_sample_callback],
                       steps_per_epoch=self.keras_feeder.streamer.data_feeder.batches_per_epoch,
                       epochs=num_epochs, validation_data=self.keras_feeder_val)
        k = 0

    # def train(self):
    #     print('Training...')
    #     im_paths = glob(utils.SAMPLE_IMAGES_DIR + '/*')
    #     train_step = self.train_step_tensor.eval(self.sess)
    #     utils.force_makedir(self.sample_inference_writeout_dirpath)
    #     try:
    #         for _ in range(self.num_train_steps):
    #             if train_step % utils.SAVE_FREQUENCY == 0:
    #                 self.save()
    #                 for path in im_paths:
    #                     print('Inferring', path)
    #                     im_org = cv2.imread(path)
    #                     ext = '.' + path.split('.')[-1]
    #                     out_viz_fpath = self.sample_inference_writeout_dirpath + os.sep \
    #                                     + path.split(os.sep)[-1].replace(ext, '_inferred-' + str(train_step) + '.png')
    #                     im_viz, shadow_mask = self.infer(im_org, viz=True)
    #                     cv2.imwrite(out_viz_fpath, im_viz)
    #                     cv2.imwrite(out_viz_fpath.replace('.png', '-mask.png'), shadow_mask * 255)
    #                     print('Output written to', out_viz_fpath)
    #             all_losses, train_step, learn_rate = self.train_step()
    #             loss_dict = dict(zip(self.loss_out_keys, all_losses))
    #             if train_step % utils.PRINT_LOSS_EVERY_N_STEPS == 0:
    #                 print('Train Step =', train_step, '; learn_rate =', learn_rate, loss_dict)
    #     except KeyboardInterrupt:
    #         print('Training abort signal received, releasing resources...')
    #         self.die()
    #         exit(0)

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


class IrvisEval:

    def __init__(self, model_dir=utils.MODEL_SAVE_DIR_ROOT):
        self.model_dir = model_dir
        self.model_dirs = np.array(glob(model_dir + os.sep + '*'))
        run_indices = np.array([int(dp.split('-')[-1]) for dp in self.model_dirs])
        self.model_dirs = self.model_dirs[np.argsort(run_indices)]
        self.results_data = {}
        self.model_fpaths = []
        self.model_steps = None
        self.irvis_nn = None
        self.results_write_path = None
        self.num_models = 0
        self.locked_nn_steps = []
        self.processed_models_log_fpath = None

    def eval_input_model(self, i):
        print('Evaluating model at', self.model_fpaths[i])
        self.update_log(i)
        if str(self.model_steps[i]) in self.results_data:
            print('Evaluation results already exist -', self.results_data[str(self.model_steps[i])])
            print('Skipping...')
            return
        self.irvis_nn.init(load_checkpoint_fpath=self.model_fpaths[i])
        shadow_prf = self.irvis_nn.eval()
        res = {'shadow_mask_prf_eval_stats_pixelwise': {'precision': float(shadow_prf[0]),
                                                        'recall': float(shadow_prf[1]),
                                                        'fscore': float(shadow_prf[2])}}
        print('Done!, results-')
        print(res)
        if os.path.isfile(self.results_write_path):
            self.results_data = json.load(open(self.results_write_path, 'r'))
        self.results_data[str(self.model_steps[i])] = res
        print('Updating results file at', self.results_write_path)
        json.dump(self.results_data, open(self.results_write_path, 'w'), indent=4, sort_keys=True)

    def get_existing_models_data(self, model_dir):
        print('Evaluating models from', model_dir)
        self.results_write_path = model_dir + os.sep + 'evaluation_results.json'
        self.processed_models_log_fpath = model_dir + os.sep + 'processed_model_steps.log'

        model_fpaths = np.array(glob(model_dir + os.sep + 'trained_models' + os.sep + '*.meta'))
        model_steps = np.array([int(dp.split('-')[-1].replace('.meta', '')) for dp in model_fpaths])

        if os.path.isfile(self.processed_models_log_fpath):
            with open(self.processed_models_log_fpath, 'r') as f:
                d = f.readlines()
                self.locked_nn_steps = [int(l.strip()) for l in d]
                print('Locked train steps -', self.locked_nn_steps)
        if os.path.isfile(self.results_write_path):
            print('Existing evaluation results found at', self.results_write_path)
            print('Reading from it to avoid re-evaluating models already evaluated....')
            self.results_data = json.load(open(self.results_write_path, 'r'))
            self.locked_nn_steps += list(map(int, list(self.results_data.keys())))
        else:
            print('No existing evaluation results found, evaluating all', model_fpaths.shape[0], 'models in dir...')
        self.locked_nn_steps = list(set(self.locked_nn_steps))
        idx = np.arange(model_steps.shape[0])
        idx = [i for i in idx if model_steps[i] not in self.locked_nn_steps]
        if len(idx) == 0:
            self.model_fpaths = []
            self.model_steps = None
            self.num_models = 0
        else:
            np.random.shuffle(idx)
            # idx = idx[:4]  # FOR DEBUG ONLY
            self.model_fpaths = model_fpaths[idx]
            self.model_steps = model_steps[idx]
            self.num_models = self.model_fpaths.shape[0]
            self.model_fpaths = [p.replace('.meta', '') for p in self.model_fpaths]

    def update_log(self, i):
        self.locked_nn_steps.append(self.model_steps[i])
        self.locked_nn_steps = np.sort(list(set(self.locked_nn_steps)))
        write_str = [str(s) + '\n' for s in self.locked_nn_steps]
        with open(self.processed_models_log_fpath, 'w') as f:
            f.writelines(write_str)
        print('Inserted Step', self.model_steps[i], 'in to log at', self.processed_models_log_fpath)

    def eval_model_dir(self, model_dir, monitor=False):
        if not monitor:
            self.get_existing_models_data(model_dir)
            for i in range(self.num_models):
                self.eval_input_model(i)
        else:
            while True:
                self.get_existing_models_data(model_dir)
                if self.num_models > 0:
                    self.eval_input_model(np.random.randint(self.num_models))
                else:
                    print('No new models found, sleeping....')
                    time.sleep(2)

    def begin_evaluation(self, run_number=None, monitor=False):
        if run_number is not None:
            self.model_dirs = np.array([self.model_dir + os.sep + 'run-' + str(run_number)])
        # data_streamer = DetectionDataStreamer()
        # detection_data_streamer = DetectionDataStreamer()
        segmap_data_streamer = SegmapDataStreamer()
        data_streamer = StreamerContainer([segmap_data_streamer])
        self.irvis_nn = IrvisNN(data_feeder=data_streamer, init_model=False)
        for model_dir in self.model_dirs:
            self.eval_model_dir(model_dir, monitor=monitor)
        print('Evaluation complete! :)')

    def __get_conf_prf(self, r, keyname):
        return r[keyname]['precision'], r[keyname]['recall'], r[keyname]['fscore']

    def __get_clf_prf(self, r):
        prfs = []
        for label_name in utils.label_names:
            prfs.append([r['classification_prf_eval_stats'][label_name]['precision'],
                         r['classification_prf_eval_stats'][label_name]['recall'],
                         r['classification_prf_eval_stats'][label_name]['fscore']])
        return prfs

    def __plot_worker(self, xs, ys, colors, labels, out_prefix, out_suffix, arg_func=np.argmax):
        print('Plotting', labels)
        import matplotlib.pyplot as plt
        title_str = ''
        plt.clf()
        for i in range(len(labels)):
            plt.plot(xs, ys[:, i], '-', color=colors[i], label=labels[i])
            title_str += 'Best ' + labels[i] + ' is at step ' + str(xs[arg_func(ys[:, i])]) + '\nwith value ' + \
                         str(ys[arg_func(ys[:, i]), i]) + '\n'
        plt.title(title_str)
        plt.legend(loc='best')
        plt.xlabel('Train Step')
        plt.ylabel('Score/Error')
        plt.savefig(out_prefix + out_suffix, bbox_inches='tight', dpi=200)

    def plot_results(self, eval_results, out_prefix):
        steps = np.sort(np.array(list(map(int, eval_results.keys()))))
        results = np.array([eval_results[str(step)] for step in steps])
        shadow_stats = np.array([self.__get_conf_prf(r, 'shadow_mask_prf_eval_stats_pixelwise') for r in results])
        self.__plot_worker(steps, shadow_stats, ['red', 'blue', 'black'], ['Precision', 'Recall', 'F-Score'],
                           out_prefix, '_shadow_prec_rec_fsc.jpg')

    def visualize_evaluation(self, run_number=None):
        if run_number is not None:
            self.model_dirs = np.array([self.model_dir + os.sep + 'run-' + str(run_number)])
        for model_dir in self.model_dirs:
            print('Visualizing evaluation results from', model_dir)
            eval_results_fpath = model_dir + os.sep + 'evaluation_results.json'
            if not os.path.isfile(eval_results_fpath):
                print(eval_results_fpath, 'not found, skipping...')
                continue
            eval_results = json.load(open(eval_results_fpath, 'r'))
            result_viz_dir = model_dir + os.sep + 'plots'
            utils.force_makedir(result_viz_dir)
            self.plot_results(eval_results, result_viz_dir + os.sep + model_dir.split(os.sep)[-1])
