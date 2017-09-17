# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr
import cv2
import os
from configure import cfg
from utils.blob import im_list_to_blob
import datasets.ds_utils
import utils.im_transforms


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)

    processed_ims = []
    # Now, build the region of interest and label blobs
    roi_blob = np.zeros((0, 8), dtype=np.float32)
    for i_im in xrange(num_images):
        # 处理图像
        img = cv2.imread(roidb[i_im]['image'])
        print i_im,roidb[i_im]['image']
        img = img.astype(np.float32)
        if roidb[i_im]['flipped']:
            img = img[:, ::-1, :]

        img_shape = img.shape

        # 处理ROI
        # x1 y1 x2 y2
        img_roi = roidb[i_im]['boxes'].astype(np.float32)

        # Check RoI
        datasets.ds_utils.validate_boxes(
            img_roi, width=img_shape[1], height=img_shape[0])

        roi = _normalize_img_roi(img_roi, img_shape)

        # 处理标签
        gt_classes = roidb[i_im]['gt_classes']
        #-------------------------------------------------------------

        if cfg.TRAIN.USE_DISTORTION:
            # cv2.imshow('before distort',img.astype(np.uint8))
            img = utils.im_transforms.ApplyDistort(img)
            # cv2.imshow('after distort',img.astype(np.uint8))
            # cv2.waitKey(0)

        # crop_bbox is define as RoIs with form (x1,y1,x2,y2)
        # if cfg.TRAIN.USE_CROP:
        # img, crop_bbox = utils.im_transforms.ApplyCrop(img)
        # else:
        # crop_bbox = np.array(
        # [0, 0, img.shape[0] - 1, img.shape[1] - 1], dtype=np.uint16)

        # expand_bbox is define as RoIs with form (x1,y1,x2,y2)
        if cfg.TRAIN.USE_EXPAND:
            img, expand_bbox = utils.im_transforms.ApplyExpand(img)
        else:
            expand_bbox = np.array([0, 0, 1, 1], dtype=np.uint16)

        roi, gt_classes = _transform_img_roi(
            roi, gt_classes, do_crop=True, crop_bbox=expand_bbox)


        #-------------------------------------------------------------
        if cfg.TRAIN.USE_SAMPLE:
            sampled_bboxes = utils.im_transforms.GenerateBatchSamples(
                roi, img_shape)
            if len(sampled_bboxes) > 0:
                rand_idx = npr.randint(len(sampled_bboxes))
                sampled_bbox = sampled_bboxes[rand_idx]

                img = utils.im_transforms.Crop(img, sampled_bbox)
                roi, gt_classes = _transform_img_roi(
                    roi,
                    gt_classes,
                    do_crop=True,
                    crop_bbox=sampled_bbox)


        #-------------------------------------------------------------
        target_size = cfg.TRAIN.SCALES[random_scale_inds[i_im]]
        img, img_scale = prep_im_for_blob(img, cfg.PIXEL_MEANS, target_size,
                                          cfg.TRAIN.MAX_SIZE)

        processed_ims.append(img)

        roi, gt_classes = _transform_img_roi(
            roi,
            gt_classes,
            do_resize=True,
            img_scale=img_scale,
            img_shape=img_shape)

        instance_id = np.zeros_like(gt_classes)
        old_c = -1
        for c_i in range(gt_classes.shape[0]):
            c = gt_classes[c_i]
            if c == old_c:
                instance_id[c_i] = instance_id[c_i - 1] + 1
            else:
                old_c = c

        gt_classes = gt_classes.reshape((roi.shape[0], 1))
        instance_id = instance_id.reshape((roi.shape[0], 1))

        # TODO(YH): 目前全部设置不难
        difficult = np.zeros_like(gt_classes)

        batch_ind = i_im * np.ones((roi.shape[0], 1))

        # print batch_ind, gt_classes, instance_id, roi, difficult
        # print batch_ind.shape, gt_classes.shape, instance_id.shape, roi.shape, difficult.shape
        roi_blob_this_image = np.hstack((batch_ind, gt_classes, instance_id,
                                         roi, difficult))
        roi_blob = np.vstack((roi_blob, roi_blob_this_image))

    # Create a blob to hold the input images
    im_blob = im_list_to_blob(processed_ims)
    blobs = {'data': im_blob}

    roi_blob = np.expand_dims(roi_blob, axis=0)
    roi_blob = np.expand_dims(roi_blob, axis=0)
    blobs['label'] = roi_blob

    return blobs


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    if cfg.RESIZE_WRAP:
        im_scale_h = float(target_size) / float(im_shape[0])
        im_scale_w = float(target_size) / float(im_shape[1])
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_w,
            fy=im_scale_h,
            interpolation=cv2.INTER_LINEAR)
        im_scale = [im_scale_h, im_scale_w]
    else:
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
        im_scale = [im_scale, im_scale]

    return im, im_scale


def _normalize_img_roi(img_roi, img_shape):
    roi_normalized = np.copy(img_roi)
    roi_normalized[:, 0] = roi_normalized[:, 0] / img_shape[1]
    roi_normalized[:, 1] = roi_normalized[:, 1] / img_shape[0]
    roi_normalized[:, 2] = roi_normalized[:, 2] / img_shape[1]
    roi_normalized[:, 3] = roi_normalized[:, 3] / img_shape[0]
    return roi_normalized


def _transform_img_roi(img_roi,
                       gt_classes,
                       do_crop=False,
                       crop_bbox=[0, 0, 1, 1],
                       do_resize=False,
                       img_scale=[1, 1],
                       img_shape=[1, 1]):
    if do_resize:
        img_roi = _UpdateBBoxByResizePolicy(img_roi, img_scale, img_shape)

    # if !MeetEmitConstraint(img_roi,bbox):
    # return np.array([])

    if do_crop:
        img_roi, gt_classes = _project_img_roi(img_roi, gt_classes, crop_bbox)
    return img_roi, gt_classes


def _project_img_roi(img_roi, gt_classes, src_bbox):
    num_roi = img_roi.shape[0]
    roi = []
    gt = []
    for i in range(num_roi):
        roi_this = img_roi[i, :]
        gt_this = gt_classes[i]
        if utils.im_transforms.MeetEmitConstraint(src_bbox, roi_this):
            roi.append(roi_this)
            gt.append(gt_this)
    img_roi = np.array(roi, dtype=np.float32)
    gt_classes = np.array(gt, dtype=np.float32)

    # assert img_roi.shape[0]>0
    if img_roi.shape[0] == 0:
        return np.zeros(
            (0, 4), dtype=np.float32), np.zeros(
                (0), dtype=np.float32)

    src_width = src_bbox[2] - src_bbox[0]
    src_height = src_bbox[3] - src_bbox[1]

    proj_roi = np.zeros_like(img_roi)
    proj_roi[:, 0] = (img_roi[:, 0] - src_bbox[0]) / src_width
    proj_roi[:, 1] = (img_roi[:, 1] - src_bbox[1]) / src_height
    proj_roi[:, 2] = (img_roi[:, 2] - src_bbox[0]) / src_width
    proj_roi[:, 3] = (img_roi[:, 3] - src_bbox[1]) / src_height

    proj_roi[:, 0] = np.minimum(np.maximum(proj_roi[:, 0], 0.0), 1.0)
    proj_roi[:, 1] = np.minimum(np.maximum(proj_roi[:, 1], 0.0), 1.0)
    proj_roi[:, 2] = np.minimum(np.maximum(proj_roi[:, 2], 0.0), 1.0)
    proj_roi[:, 3] = np.minimum(np.maximum(proj_roi[:, 3], 0.0), 1.0)

    return proj_roi, gt_classes


def _UpdateBBoxByResizePolicy(roi, img_scale, img_shape):
    new_shape = [
        1.0 * img_shape[0] * img_scale[0], 1.0 * img_shape[1] * img_scale[1]
    ]

    roi[:, 0] = roi[:, 0] * img_shape[1]
    roi[:, 1] = roi[:, 1] * img_shape[0]
    roi[:, 2] = roi[:, 2] * img_shape[1]
    roi[:, 3] = roi[:, 3] * img_shape[0]

    roi[:, 0] = roi[:, 0] * img_scale[1]
    roi[:, 1] = roi[:, 1] * img_scale[0]
    roi[:, 2] = roi[:, 2] * img_scale[1]
    roi[:, 3] = roi[:, 3] * img_scale[0]

    roi[:, 0] = roi[:, 0] / new_shape[1]
    roi[:, 1] = roi[:, 1] / new_shape[0]
    roi[:, 2] = roi[:, 2] / new_shape[1]
    roi[:, 3] = roi[:, 3] / new_shape[0]

    return roi
