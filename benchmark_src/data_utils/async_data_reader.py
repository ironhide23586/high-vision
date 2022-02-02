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


import logging
from threading import Thread
from queue import Queue
import time

import numpy as np

import utils


class TrainFeeder:

    def __init__(self, ingestor, batch_size=utils.BATCH_SIZE, batches_per_queue=utils.BATCHES_PER_ASYNC_QUEUE):
        self.data_reader_keepalive = True
        self.ingestor = ingestor
        self.batch_size = batch_size
        self.epochs = 0
        self.batch_iters = 0
        self.epoch_size_total = self.ingestor.dataset.epoch_size
        if self.batch_size > self.epoch_size_total:
            logging.warning('Batch size exceeds epoch size, setting batch size to epoch size')
            self.batch_size = self.epoch_size_total
        self.batches_per_epoch = self.epoch_size_total // self.batch_size
        self.epoch_size = self.batch_size * self.batches_per_epoch
        self.batch_data_x = []
        self.batch_data_y = []
        self.batch_data_x_fpaths = []
        self.batch_data_y_fpaths = []
        self.batch_fpaths = []
        self.im_coords = None
        self.total_iters = 0
        self.train_state = {'epoch': 1, 'batch': 0, 'total_iters': 0, 'previous_epoch_done': False}
        self.start_batch_queue_populater(batches_per_queue=batches_per_queue)

    def random_sliding_square_crop(self, im):
        h, w, _ = im.shape
        if h == w:
            return im.copy()
        min_dim = w
        max_dim = h
        if h < w:
            min_dim = h
            max_dim = w
        start_idx = np.random.randint(max_dim - min_dim)
        end_idx = start_idx + min_dim
        if h < w:
            im_cropped = im[:, start_idx:end_idx, :]
        else:
            im_cropped = im[start_idx:end_idx, :, :]
        return im_cropped

    def center_crop(self, x):
        h, w, _ = x.shape
        offset = abs((w - h) // 2)
        if h < w:
            x_pp = x[:, offset:offset + h, :]
        elif w < h:
            x_pp = x[offset:offset + w, :, :]
        else:
            x_pp = x.copy()
        return x_pp

    def __ingest_indices(self, start_idx, end_idx):
        ims = []
        gt_list = []
        idx = start_idx
        bs = end_idx - start_idx
        # for idx in range(start_idx, end_idx):
        while len(ims) < bs:
            im, gts = self.ingestor.get_data_train_format(idx)
            # if im is None or gts is None:
            #     idx += 1
            #     continue
            ims.append(im)
            gt_list.append(gts)
            idx += 1
        ims = np.array(ims).astype(np.float32)
        gt_list_out = np.array(gt_list).astype(np.float32)
        return ims, gt_list_out

    def get_data(self, batch_size=None):
        if batch_size is not None:
            logging.info('External batch_size provided, recomputing batches_per_epoch and epoch_size')
            self.batch_size = batch_size
            if self.batch_size > self.epoch_size_total:
                logging.warning('Batch size exceeds epoch size, setting batch size to epoch size')
                self.batch_size = self.epoch_size_total
            self.batches_per_epoch = self.epoch_size_total // self.batch_size
            self.epoch_size = self.batch_size * self.batches_per_epoch
        self.batch_iters += 1
        self.total_iters += 1
        self.epoch_completed = False
        if self.batch_iters > self.batches_per_epoch:
            logging.info('---------> Loading new epoch in to Buffered Batch Reader')
            self.epoch_completed = True
            self.batch_iters = 1
            self.epochs += 1
        train_state = {'epoch': self.epochs + 1, 'batch': self.batch_iters, 'total_iters': self.total_iters,
                       'previous_epoch_done': self.epoch_completed}
        start_idx = (self.batch_iters - 1) * self.batch_size
        end_idx = start_idx + self.batch_size
        ims, gts = self.__ingest_indices(start_idx, end_idx)
        return ims, gts, train_state

    def __queue_filler_process(self):
        while self.data_reader_keepalive:
            if self.buffer.full():
                time.sleep(2)
                continue
            ims, gts, train_state = self.get_data()
            if ims.shape[0] > 0:
                self.buffer.put([ims, gts, train_state])
            else:
                print('Empty batch, skipping....')

    def start_batch_queue_populater(self, batches_per_queue=20):
        logging.info('Starting Populator process for batch buffer')
        self.buffer = Queue(maxsize=batches_per_queue)
        self.queue_filler_thread = Thread(target=self.__queue_filler_process)
        self.queue_filler_thread.start()

    def die(self):
        self.data_reader_keepalive = False
        self.queue_filler_thread.join()

    def dequeue(self):
        if not self.buffer.empty():
            self.batch_data_x, self.batch_data_y, self.train_state = self.buffer.get()
            if self.train_state['previous_epoch_done']:
                logging.info('----------------EPOCH ' + str(self.train_state['epoch'] - 1)
                             + ' COMPLETE----------------')
            logging.info('Epoch ' + str(self.train_state['epoch']) + ', Batch ' + str(self.train_state['batch']))
            return self.batch_data_x, self.batch_data_y
        else:
            # logging.warning('Buffer empty, waiting for it to repopulate..')
            while self.buffer.empty():
                time.sleep(1)
                continue
            return self.dequeue()
