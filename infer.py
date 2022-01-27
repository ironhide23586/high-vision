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

from glob import glob
import os
import time
import hashlib
import sys
import fnmatch
from tqdm import tqdm

import cv2
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import utils

utils.SAMPLE_IMAGES_DIR = 'scratchspace/train_features'


from nets_v2.nn import IrvisNN
# from data_utils.data_io import SegmapDataStreamer


# segmap_data_streamer_val = SegmapDataStreamer(mode='test', batch_size=1)


# def get_gen():
#     for _ in range(segmap_data_streamer_val.data_feeder.batches_per_epoch):
#         yield segmap_data_streamer_val.get_data_batch()


def process_fpaths(irvis_nn, im_paths_, out_dir):
    utils.force_makedir(out_dir + '/overlays')
    utils.force_makedir(out_dir + '/masks')
    print('NOTE: Inference of first image will take longer due to hardware initialization.')
    tps = 0
    fps = 0
    fns = 0
    tns = 0
    batch_size = 1
    im_paths = fnmatch.filter(im_paths_, '*_vv*')
    n = len(im_paths) // batch_size
    for i_ in tqdm(range(n)):
        ims = []
        paths = []
        start_idx = i_ * batch_size
        for i in range(start_idx, start_idx + batch_size):
            if i >= len(im_paths):
                i = len(im_paths) - 1
            path = im_paths[i]
            paths.append(path)
            print('Inferring', path)
            im_vv = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_LOAD_GDAL)
            im_vh = cv2.imread(path.replace('_vv', '_vh'), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_LOAD_GDAL)
            im_vv = im_vv - im_vv.min()
            im_vh = im_vh - im_vh.min()
            im_filler = (im_vv + im_vh) / 2.
            im = np.rollaxis(np.array([im_vv, im_vh, im_filler]), 0, 3)
            ims.append(im)
        ims = np.array(ims)
        st = time.time()
        overlays, shadow_masks = irvis_nn.infer_final(ims)
        et = time.time()
        duration = (et - st) / ims.shape[0]

        for i in range(ims.shape[0]):
            shadow_mask = shadow_masks[i].copy()
            path = paths[i]
            ext = '.' + path.split('.')[-1]
            out_viz_fpath = out_dir + '/overlays' + os.sep + path.split(os.sep)[-1].replace(ext, '.png')
            out_mask_fpath = out_dir + '/masks' + os.sep + path.split(os.sep)[-1].replace(ext, '.png')
            gt_fpath = path.replace('images', 'labels')
            shadow_mask[shadow_mask < .5] = 0.
            shadow_mask[shadow_mask >= .5] = 1.
            cv2.imwrite(out_mask_fpath, shadow_mask * 255)
            gt_im = cv2.imread(gt_fpath)[:, :, 0]
            # gt_im = cv2.resize(gt_im, (utils.IM_DIM, utils.IM_DIM), interpolation=cv2.INTER_NEAREST)
            # gt_im = cv2.resize(gt_im, (1280, 1280), interpolation=cv2.INTER_NEAREST)
            y_pred = shadow_mask.flatten()
            y_gt = gt_im.flatten().astype(np.float) / 255.

            tp = np.logical_and(y_pred == 1, y_gt == 1).sum()
            fp = np.logical_and(y_pred == 1, y_gt == 0).sum()
            fn = np.logical_and(y_pred == 0, y_gt == 1).sum()
            tn = np.logical_and(y_pred == 0, y_gt == 0).sum()

            tps += tp
            fps += fp
            fns += fn
            tns += tn

            if np.unique(y_gt).shape[0] > 1:
                prec, rec, fsc, _ = precision_recall_fscore_support(y_gt, y_pred)
                prec = prec[1]
                rec = rec[1]
                fsc = fsc[1]
            else:
                prec = 0.
                rec = 0.
                fsc = 0.
            out_viz_fpath = out_viz_fpath.replace('.png', '-' + '_'.join(list(map(str, [prec, rec, fsc]))) + '.png')
            cv2.imwrite(out_viz_fpath, utils.overlay_mask(overlays[i], gt_im, color=(0, 255, 0), alpha=.35))

            prec_running = tps / (tps + fps)
            rec_running = tps / (tps + fns)
            fsc_running = 2. / ((1. / prec_running) + (1. / rec_running))
            print('Inference Time =', duration)
            print(prec_running, rec_running, fsc_running)
    print('Outputs written to', out_dir)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        utils.set_gpu_id(sys.argv[1])
    irvis_nn = IrvisNN(mode='infer')
    irvis_nn.init()

    # keras_feeder_val = tf.data.Dataset.from_generator(get_gen, (tf.float32, tf.float32))
    # irvis_nn.model.compile(metrics=['Precision', 'Recall'])
    # irvis_nn.model.evaluate(keras_feeder_val, use_multiprocessing=True, workers=8)

    im_paths = glob(utils.SAMPLE_IMAGES_DIR + '/*')
    out_dir = 'scratchspace/outs-' + irvis_nn.load_checkpoint_fpath.split(os.sep)[-1] + '-' +\
              hashlib.sha256(utils.SAMPLE_IMAGES_DIR.encode('latin')).hexdigest()[:10]

    process_fpaths(irvis_nn, im_paths, out_dir)

    # process_video(irvis_nn, INPUT_VIDEO_PATH, START_SECOND, END_SECOND)
