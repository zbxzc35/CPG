# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from configure import cfg
from roi_data_layer.minibatch_wsl import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue
import os
import cv2
import shutil


class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        if cfg.TRAIN.SHUFFLE == False:
            self._cur = 0
            self._perm = np.arange(len(self._roidb))
            return
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((np.random.permutation(horz_inds),
                              np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1, ))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]

            if cfg.TRAIN.ROI_AU:
                blobs, roidb = get_minibatch(minibatch_db, self._num_classes,
                                             db_inds)
                for i, ind in enumerate(db_inds):
                    self._roidb[ind] = roidb[i]
            else:
                blobs = get_minibatch(minibatch_db, self._num_classes, db_inds)

            return blobs
            # return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(1280)
            self._prefetch_process = BlobFetcher(self._blob_queue, self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()

            # Terminate the child process when the parent exists

            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()

            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
                         max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.ROIS_PER_IM, 5)
        self._name_to_top_map['rois'] = idx
        idx += 1

        if cfg.CONTEXT:
            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.ROIS_PER_IM,
                             9)
            self._name_to_top_map['rois_context'] = idx
            idx += 1

            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.ROIS_PER_IM,
                             9)
            self._name_to_top_map['rois_frame'] = idx
            idx += 1

        if cfg.USE_ROI_SCORE:
            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.ROIS_PER_IM)
            self._name_to_top_map['roi_scores'] = idx
            idx += 1

        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH)
        self._name_to_top_map['roi_num'] = idx
        idx += 1

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, self._num_classes)
        self._name_to_top_map['labels'] = idx
        idx += 1

        if cfg.TRAIN.OPG_CACHE and len(top) > idx:
            if os.path.exists(cfg.TRAIN.OPG_CACHE_PATH):
                shutil.rmtree(cfg.TRAIN.OPG_CACHE_PATH)
            os.makedirs(cfg.TRAIN.OPG_CACHE_PATH)

            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.ROIS_PER_IM,
                             self._num_classes)
            self._name_to_top_map['opg_filter'] = idx
            idx += 1

            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 1)
            self._name_to_top_map['opg_io'] = idx
            idx += 1
        else:
            cfg.TRAIN.OPG_CACHE = False

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        if False:
            num_roi_vis = 100
            # channel_swap = (0, 3, 1, 2)
            channel_swap = (0, 2, 3, 1)
            ims_blob = blobs['data'].copy()
            rois_blob = blobs['rois'].copy()

            ims = ims_blob.transpose(channel_swap)
            for i in range(cfg.TRAIN.IMS_PER_BATCH):
                im = ims[i]
                im += cfg.PIXEL_MEANS
                im = im.astype(np.uint8).copy()

                height = im.shape[0]
                width = im.shape[1]

                num_img_roi = 0
                for j in range(rois_blob.shape[0]):
                    roi = rois_blob[j]
                    if roi[0] != i:
                        continue
                    num_img_roi += 1
                    if num_img_roi > num_roi_vis:
                        break
                    x1 = int(roi[1] * width)
                    y1 = int(roi[2] * height)
                    x2 = int(roi[3] * width)
                    y2 = int(roi[4] * height)
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 1)

                cv2.imshow('image ' + str(i), im)
            cv2.waitKey(0)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        if cfg.TRAIN.SHUFFLE == False:
            self._cur = 0
            self._perm = np.arange(len(self._roidb))
            return
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((np.random.permutation(horz_inds),
                              np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1, ))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            if cfg.TRAIN.ROI_AU:
                blobs, roidb = get_minibatch(minibatch_db, self._num_classes,
                                             db_inds)
                for i, ind in enumerate(db_inds):
                    self._roidb[ind] = roidb[i]
            else:
                blobs = get_minibatch(minibatch_db, self._num_classes, db_inds)
            self._queue.put(blobs)
