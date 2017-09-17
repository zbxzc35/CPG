# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr
import cv2
import os
from configure import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import datasets.ds_utils
import utils.im_transforms
import pdb


def get_minibatch(roidb, num_classes, db_inds):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    # roi_per_image = cfg.TRAIN.ROIS_PER_IM

    # Get the input image blob, formatted for caffe
    # im_crops is define as RoIs with form (y1,x1,y2,x2)
    im_blob, im_scales, im_crops, im_shapes = _get_image_blob(
        roidb, random_scale_inds, db_inds)

    # row col row col to x1 y1 x2 y2
    im_crops = np.array(im_crops, dtype=np.uint16)
    im_crops = im_crops[:, (1, 0, 3, 2)]

    blobs = {'data': im_blob}

    # Now, build the region of interest and label blobs
    roi_blob = np.zeros((0, 5), dtype=np.float32)
    roi_context_blob = np.zeros((0, 9), dtype=np.float32)
    roi_frame_blob = np.zeros((0, 9), dtype=np.float32)
    roi_score_blob = np.zeros((0, 1), dtype=np.float32)
    roi_num_blob = np.zeros((0, 1), dtype=np.float32)
    label_blob = np.zeros((0, num_classes), dtype=np.float32)
    opg_filter_blob = np.zeros((0, num_classes), dtype=np.float32)
    opg_io_blob = np.zeros((0, 1), dtype=np.float32)
    for i_im in xrange(num_images):
        # x1 y1 x2 y2
        im_roi = roidb[i_im]['boxes'].astype(np.float32)
        im_label = roidb[i_im]['gt_classes']
        if cfg.USE_ROI_SCORE:
            im_roi_score = roidb[i_im]['box_scores']

        im_crop = im_crops[i_im]

        # TODO(YH): CROP is conflict with OPG_CACHE and ROI_AU, thereforce caffe should check the validity of RoI
        # 删除超出CROP的RoI
        # drop = (im_roi[:, 0] >= im_crop[2]) | (im_roi[:, 1] >= im_crop[3]) | (
        # im_roi[:, 2] <= im_crop[0]) | (im_roi[:, 3] <= im_crop[1])
        # im_roi = im_roi[~drop]
        # if cfg.USE_ROI_SCORE:
        # im_roi_score = im_roi_score[~drop]

        # Check RoI
        datasets.ds_utils.validate_boxes(
            im_roi, width=im_shapes[i_im][1], height=im_shapes[i_im][0])

        roi_per_this_image = np.minimum(cfg.TRAIN.ROIS_PER_IM,
                                         im_roi.shape[0])
        im_roi = im_roi[:roi_per_this_image, :]
        if cfg.USE_ROI_SCORE:
            im_roi_score = im_roi_score[:roi_per_this_image]

        if cfg.TRAIN.OPG_CACHE:
            filter_blob_this = np.zeros(
                (roi_per_this_image, num_classes), dtype=np.float32)
            for target_size in cfg.TRAIN.SCALES:
                if target_size == cfg.TRAIN.SCALES[random_scale_inds[i_im]]:
                    continue
                filter_name = str(db_inds[i_im] * 10000 + target_size)
                # print filter_name
                filter_path = os.path.join(cfg.TRAIN.OPG_CACHE_PATH,
                                           filter_name)

                if os.path.exists(filter_path):
                    filter_this = cpg.cpg_utils.binaryfile_to_blobproto_to_array(
                        filter_path).astype(np.float32)
                    # filter_blob_this = np.logical_or(
                    # filter_blob_this,
                    # cpg.cpg_utils.binaryfile_to_blobproto_to_array(filter_path)).astype(np.float32)
                    # filter_blob_this = np.add(
                    # filter_blob_this,
                    # cpg.cpg_utils.binaryfile_to_blobproto_to_array(filter_path)).astype(np.float32)
                    filter_blob_this = np.maximum(filter_blob_this,
                                                  filter_this)
            io_blob_this = np.array(
                [
                    db_inds[i_im] * 10000 +
                    cfg.TRAIN.SCALES[random_scale_inds[i_im]]
                ],
                dtype=np.float32)

            opg_filter_blob = np.vstack((opg_filter_blob, filter_blob_this))
            opg_io_blob = np.vstack((opg_io_blob, io_blob_this))

        if cfg.TRAIN.ROI_AU:
            # pdb.set_trace()
            offset = 1.0 / im_scales[i_im] / cfg.SPATIAL_SCALE
            offset_step = cfg.TRAIN.ROI_AU_STEP

            if cfg.TRAIN.OPG_CACHE:
                filter_blob_this_sum = np.sum(filter_blob_this, 1)
                au_ind = filter_blob_this_sum == 0
            else:
                au_ind = np.ones(roi_per_this_image, dtype=np.bool)
            offsets = np.random.randint(
                2 * offset_step + 1, size=(np.sum(au_ind),
                                           4)).astype(np.float32)
            offsets -= offset_step
            offsets *= offset

            au_roi_o = im_roi[au_ind]
            au_roi_n = im_roi[~au_ind]
            au_roi = au_roi_o + offsets

            keep = datasets.ds_utils.filter_validate_boxes(
                au_roi, im_shapes[i_im][1], im_shapes[i_im][0])
            au_roi[~keep] = au_roi_o[~keep]

            ovrs = datasets.ds_utils.overlaps(au_roi, au_roi_n)
            thresholded = ovrs >= 0.5
            keep = np.sum(thresholded, 1) == 0
            au_roi[~keep] = au_roi_o[~keep]

            # im_roi = np.vstack((im_roi, au_roi))
            im_roi[au_ind] = au_roi

            # roi_per_this_image = np.minimum(cfg.ROIS_PER_IM, im_roi.shape[0])
            # im_roi = im_roi[:roi_per_this_image, :]
            # if cfg.USE_ROI_SCORE:
            # au_roi_score = im_roi_score[au_ind]
            # im_roi_score = np.vstack((im_roi_score, au_roi_score))
            # im_roi_score = im_roi_score[:roi_per_this_image]

            # roidb[i_im]['boxes'] = im_roi

        if cfg.CONTEXT:
            im_inner_roi, im_outer_roi = get_inner_outer_roi(
                im_roi, cfg.CONTEXT_RATIO)

        # project
        roi = _project_im_roi(im_roi, im_scales[i_im], im_crop)
        if cfg.CONTEXT:
            roi_inner = _project_im_roi(im_inner_roi, im_scales[i_im],
                                          im_crop)
            roi_outer = _project_im_roi(im_outer_roi, im_scales[i_im],
                                          im_crop)

        batch_ind = i_im * np.ones((roi.shape[0], 1))
        roi_blob_this_image = np.hstack((batch_ind, roi))
        roi_blob = np.vstack((roi_blob, roi_blob_this_image))
        if cfg.CONTEXT:
            roi_context_blob_this_image = np.hstack((batch_ind, roi_outer,
                                                      roi))
            roi_context_blob = np.vstack((roi_context_blob,
                                           roi_context_blob_this_image))

            roi_frame_blob_this_image = np.hstack((batch_ind, roi,
                                                    roi_inner))
            roi_frame_blob = np.vstack((roi_frame_blob,
                                         roi_frame_blob_this_image))

        if cfg.USE_ROI_SCORE:
            roi_score_blob = np.vstack((roi_score_blob, im_roi_score))
        else:
            roi_score_blob = np.vstack((roi_score_blob, np.zeros(
                (roi_per_this_image, 1), dtype=np.float32)))

        im_roi_num = np.ones((1))
        im_roi_num[0] = roi.shape[0]
        roi_num_blob = np.vstack((roi_num_blob, im_roi_num))

        # Add to label
        if cfg.USE_BG:
            im_label = np.hstack((im_label, [1.0]))
        label_blob = np.vstack((label_blob, im_label))

    blobs['roi'] = roi_blob
    if cfg.CONTEXT:
        blobs['roi_context'] = roi_context_blob
        blobs['roi_frame'] = roi_frame_blob

    if cfg.USE_ROI_SCORE:
        # n * 1 to n
        blobs['roi_score'] = np.add(
            np.reshape(roi_score_blob, [roi_score_blob.shape[0]]), 1)
    else:
        blobs['roi_score'] = np.ones((roi_blob.shape[0]), dtype=np.float32)

    blobs['roi_num'] = roi_num_blob

    blobs['label'] = label_blob
    if cfg.TRAIN.OPG_CACHE:
        blobs['opg_filter'] = opg_filter_blob
        blobs['opg_io'] = opg_io_blob

    # print "roi_blob: ", roi_blob
    # print "roi_context_blob: ", roi_context_blob
    # print "roi_frame_blob: ", roi_frame_blob
    # print "roi_score_blob: ", roi_score_blob
    # print "roi_num_blob: ", roi_num_blob
    # print "label_blob: ", label_blob

    if cfg.TRAIN.ROI_AU:
        return blobs, roidb
    return blobs


def _get_image_blob(roidb, scale_inds, db_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    im_crops = []
    im_shapes = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        im = im.astype(np.float32)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_shapes.append(im.shape)

        if cfg.TRAIN.USE_DISTORTION:
            im = utils.im_transforms.ApplyDistort(im)

            # hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            # s0 = npr.random() * (cfg.TRAIN.SATURATION - 1) + 1
            # s1 = npr.random() * (cfg.TRAIN.EXPOSURE - 1) + 1
            # s0 = s0 if npr.random() > 0.5 else 1.0 / s0
            # s1 = s1 if npr.random() > 0.5 else 1.0 / s1
            # hsv = np.array(hsv, dtype=np.float)
            # hsv[:, :, 1] = np.minimum(s0 * hsv[:, :, 1], 255)
            # hsv[:, :, 2] = np.minimum(s1 * hsv[:, :, 2], 255)
            # hsv = np.array(hsv, dtype=np.uint8)
            # im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if cfg.TRAIN.USE_CROP:
            im_shape = np.array(im.shape)
            crop_dims = im_shape[:2] * cfg.TRAIN.CROP

            r0 = npr.random()
            r1 = npr.random()
            s = im_shape[:2] - crop_dims
            s[0] *= r0
            s[1] *= r1
            im_crop = np.array(
                [s[0], s[1], s[0] + crop_dims[0] - 1, s[1] + crop_dims[1] - 1],
                dtype=np.uint16)

            im = im[im_crop[0]:im_crop[2] + 1, im_crop[1]:im_crop[3] + 1, :]
        else:
            im_crop = np.array(
                [0, 0, im.shape[0] - 1, im.shape[1] - 1], dtype=np.uint16)

        if cfg.OPG_DEBUG:
            im_save = im

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)

        if cfg.OPG_DEBUG:
            im_save = cv2.resize(
                im_save,
                None,
                None,
                fx=im_scale,
                fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
            cv2.imwrite('tmp/' + str(cfg.TRAIN.PASS_IM) + '_.png', im_save)
            cfg.TRAIN.PASS_IM = cfg.TRAIN.PASS_IM + 1

        im_scales.append(im_scale)
        im_crops.append(im_crop)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales, im_crops, im_shapes


def get_inner_outer_roi(im_roi, ratio):
    assert ratio > 1, 'ratio should be lager than one in get_inner_outer_roi'
    roi = im_roi.astype(np.float32, copy=True)
    # x1 y1 x2 y2
    roi_w = roi[:, 2] - roi[:, 0]
    roi_h = roi[:, 3] - roi[:, 1]

    roi_inner_w = roi_w / ratio
    roi_inner_h = roi_h / ratio

    roi_outer_w = roi_w * ratio
    roi_outer_h = roi_h * ratio

    inner_residual_w = roi_w - roi_inner_w
    inner_residual_h = roi_h - roi_inner_h

    outer_residual_w = roi_outer_w - roi_w
    outer_residual_h = roi_outer_h - roi_h

    roi_inner = np.copy(roi)
    roi_outer = np.copy(roi)

    # print roi_inner.dtype, roi_inner.shape
    # print inner_residual_w.dtype, inner_residual_w.shape
    # print (inner_residual_w / 2).dtype, (inner_residual_w / 2).shape

    roi_inner[:, 0] += inner_residual_w / 2
    roi_inner[:, 1] += inner_residual_h / 2
    roi_inner[:, 2] -= inner_residual_w / 2
    roi_inner[:, 3] -= inner_residual_h / 2

    roi_outer[:, 0] -= outer_residual_w / 2
    roi_outer[:, 1] -= outer_residual_h / 2
    roi_outer[:, 2] += outer_residual_w / 2
    roi_outer[:, 3] += outer_residual_h / 2

    return roi_inner, roi_outer


def _project_im_roi(im_roi, im_scale_factor, im_crop):
    """Project image RoIs into the rescaled training image."""
    im_roi[:, 0] = np.minimum(
        np.maximum(im_roi[:, 0], im_crop[0]), im_crop[2])
    im_roi[:, 1] = np.minimum(
        np.maximum(im_roi[:, 1], im_crop[1]), im_crop[3])
    im_roi[:, 2] = np.maximum(
        np.minimum(im_roi[:, 2], im_crop[2]), im_crop[0])
    im_roi[:, 3] = np.maximum(
        np.minimum(im_roi[:, 3], im_crop[3]), im_crop[1])
    crop = np.tile(im_crop[:2], [im_roi.shape[0], 2])
    roi = (im_roi - crop) * im_scale_factor

    # For YAROIPooling Layer
    # roi = (im_roi - crop)
    # width = im_crop[2] - im_crop[0]
    # height = im_crop[3] - im_crop[1]
    # roi[:, 0] = roi[:, 0] / width
    # roi[:, 1] = roi[:, 1] / height
    # roi[:, 2] = roi[:, 2] / width
    # roi[:, 3] = roi[:, 3] / height

    return roi
