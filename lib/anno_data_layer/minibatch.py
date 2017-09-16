# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
import os
from configure import cfg
from utils.blob import im_list_to_blob
import datasets.ds_utils


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)

    # Get the input image blob, formatted for caffe
    # im_crops is define as RoIs with form (y1,x1,y2,x2)
    im_blob, im_scales, im_crops, im_shapes = _get_image_blob(
        roidb, random_scale_inds)

    # row col row col to x1 y1 x2 y2
    im_crops = np.array(im_crops, dtype=np.uint16)
    im_crops = im_crops[:, (1, 0, 3, 2)]

    blobs = {'data': im_blob}

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 8), dtype=np.float32)
    for i_im in xrange(num_images):
        # x1 y1 x2 y2
        im_rois = roidb[i_im]['boxes'].astype(np.float32)

        im_crop = im_crops[i_im]

        # TODO(YH): CROP is conflict with OPG_CACHE and ROI_AU, thereforce caffe should check the validity of RoI
        # 删除超出CROP的RoI
        # drop = (im_rois[:, 0] >= im_crop[2]) | (im_rois[:, 1] >= im_crop[3]) | (
        # im_rois[:, 2] <= im_crop[0]) | (im_rois[:, 3] <= im_crop[1])
        # im_rois = im_rois[~drop]
        # if cfg.USE_ROI_SCORE:
        # im_roi_scores = im_roi_scores[~drop]

        # Check RoI
        datasets.ds_utils.validate_boxes(
            im_rois, width=im_shapes[i_im][1], height=im_shapes[i_im][0])

        rois = _project_im_rois(im_rois, im_scales[i_im], im_crop)

        gt_classes = roidb[i_im]['gt_classes']

        instance_id = np.zeros_like(gt_classes)
        old_c = -1
        for c_i in range(gt_classes.shape[0]):
            c = gt_classes[c_i]
            if c == old_c:
                instance_id[c_i] = instance_id[c_i - 1] + 1
            else:
                old_c = c

        gt_classes = gt_classes.reshape((rois.shape[0], 1))
        instance_id = instance_id.reshape((rois.shape[0], 1))

        difficult = np.zeros_like(gt_classes)

        batch_ind = i_im * np.ones((rois.shape[0], 1))
        # print batch_ind, gt_classes, instance_id, rois, difficult
        # print batch_ind.shape, gt_classes.shape, instance_id.shape, rois.shape, difficult.shape
        rois_blob_this_image = np.hstack((batch_ind, gt_classes, instance_id,
                                          rois, difficult))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

    rois_blob = np.expand_dims(rois_blob, axis=0)
    rois_blob = np.expand_dims(rois_blob, axis=0)
    blobs['label'] = rois_blob

    return blobs


def _get_image_blob(roidb, scale_inds):
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
            im = ApplyDistort(im)
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


def _project_im_rois(im_rois, im_scale_factor, im_crop):
    """Project image RoIs into the rescaled training image."""
    im_rois[:, 0] = np.minimum(
        np.maximum(im_rois[:, 0], im_crop[0]), im_crop[2])
    im_rois[:, 1] = np.minimum(
        np.maximum(im_rois[:, 1], im_crop[1]), im_crop[3])
    im_rois[:, 2] = np.maximum(
        np.minimum(im_rois[:, 2], im_crop[2]), im_crop[0])
    im_rois[:, 3] = np.maximum(
        np.minimum(im_rois[:, 3], im_crop[3]), im_crop[1])
    crop = np.tile(im_crop[:2], [im_rois.shape[0], 2])
    rois = (im_rois - crop)
    rois[:, 1] = rois[:, 1] * im_scale_factor[0]
    rois[:, 3] = rois[:, 3] * im_scale_factor[0]
    rois[:, 0] = rois[:, 0] * im_scale_factor[1]
    rois[:, 2] = rois[:, 2] * im_scale_factor[1]

    # For YAROIPooling Layer
    rois = (im_rois - crop)
    width = im_crop[2] - im_crop[0]
    height = im_crop[3] - im_crop[1]
    rois[:, 0] = rois[:, 0] / width
    rois[:, 1] = rois[:, 1] / height
    rois[:, 2] = rois[:, 2] / width
    rois[:, 3] = rois[:, 3] / height

    return rois


def ApplyDistort(in_img):
    prob = npr.random()
    if prob > 0.5:
        # Do random brightness distortion.
        out_img = RandomBrightness(in_img, cfg.TRAIN.brightness_prob,
                                   cfg.TRAIN.brightness_delta)

        # Do random contrast distortion.
        out_img = RandomContrast(in_img, cfg.TRAIN.contrast_prob,
                                 cfg.TRAIN.contrast_lower,
                                 cfg.TRAIN.contrast_upper)

        # Do random saturation distortion.
        out_img = RandomSaturation(in_img, cfg.TRAIN.saturation_prob,
                                   cfg.TRAIN.saturation_lower,
                                   cfg.TRAIN.saturation_upper)

        # Do random hue distortion.
        out_img = RandomHue(in_img, cfg.TRAIN.hue_prob, cfg.TRAIN.hue_delta)

        # Do random reordering of the channels.
        out_img = RandomOrderChannels(in_img, cfg.TRAIN.random_order_prob)
    else:
        # Do random brightness distortion.
        out_img = RandomBrightness(in_img, cfg.TRAIN.brightness_prob,
                                   cfg.TRAIN.brightness_delta)

        # Do random saturation distortion.
        out_img = RandomSaturation(in_img, cfg.TRAIN.saturation_prob,
                                   cfg.TRAIN.saturation_lower,
                                   cfg.TRAIN.saturation_upper)

        # Do random hue distortion.
        out_img = RandomHue(in_img, cfg.TRAIN.hue_prob, cfg.TRAIN.hue_delta)

        # Do random contrast distortion.
        out_img = RandomContrast(in_img, cfg.TRAIN.contrast_prob,
                                 cfg.TRAIN.contrast_lower,
                                 cfg.TRAIN.contrast_upper)

        # Do random reordering of the channels.
        out_img = RandomOrderChannels(in_img, cfg.TRAIN.random_order_prob)

    return out_img


def convertTo(in_img, alpha, beta):
    out_img = in_img.astype(np.float32)
    out_img = out_img * alpha + beta
    out_img = np.clip(out_img, 0, 255)
    out_img = out_img.astype(in_img.dtype)
    return out_img


def RandomBrightness(in_img, brightness_prob, brightness_delta):
    prob = npr.random()
    if prob < brightness_prob:
        assert brightness_delta > 0, "brightness_delta must be non-negative."
        delta = npr.uniform(-brightness_delta, brightness_delta)
        out_img = AdjustBrightness(in_img, delta)
    else:
        out_img = in_img
    return out_img


def AdjustBrightness(in_img, delta):
    if abs(delta) > 0:
        # out_img = cv2.convertTo(in_img, 1, 1, delta)
        out_img = convertTo(in_img, 1, delta)
    else:
        out_img = in_img
    return out_img


def RandomContrast(in_img, contrast_prob, lower, upper):
    prob = npr.random()
    if prob < contrast_prob:
        assert upper >= lower, 'contrast upper must be >= lower.'
        assert lower >= 0, 'contrast lower must be non-negative.'
        delta = npr.uniform(lower, upper)
        out_img=AdjustContrast(in_img, delta)
    else:
        out_img = in_img


def AdjustContrast(in_img, delta):
    if abs(delta - 1.0) > 1e-3:
        # out_img = cv2.convertTo(in_img, -1, delta, 0)
        out_img = convertTo(in_img, delta, 0)
    else:
        out_img = in_img
    return out_img


def RandomSaturation(in_img, saturation_prob, lower, upper):
    prob = npr.random()
    if prob < saturation_prob:
        assert upper >= lower, 'saturation upper must be >= lower.'
        assert lower >= 0, 'saturation lower must be non-negative.'
        delta = npr.uniform(lower, upper)
        out_img = AdjustSaturation(in_img, delta)
    else:
        out_img = in_img
    return out_img


def AdjustSaturation(in_img, delta):
    if abs(delta - 1.0) != 1e-3:
        # Convert to HSV colorspae.
        out_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)

        # Split the image to 3 channels.
        h, s, v = cv2.split(out_img)

        # Adjust the saturation.
        # channels[1] = cv2.convertTo(channels[1], -1, delta, 0)
        s = convertTo(s, delta, 0)
        out_img = cv2.merge((h, s, v))

        # Back to BGR colorspace.
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2BGR)
    else:
        out_img = in_img
    return out_img


def RandomHue(in_img, hue_prob, hue_delta):
    prob = npr.random()
    if prob < hue_prob:
        assert hue_delta >= 0, 'hue_delta must be non-negative.'
        delta = npr.uniform(-hue_delta, hue_delta)
        out_img = AdjustHue(in_img, delta)
    else:
        out_img = in_img
    return out_img


def AdjustHue(in_img, delta):
    if abs(delta) > 0:
        # Convert to HSV colorspae.
        out_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)

        # Split the image to 3 channels.
        h, s, v = cv2.split(out_img)

        # Adjust the hue.
        # channels[0] = cv2.convertTo(channels[0], -1, 1, delta)
        h = convertTo(h, 1, delta)
        out_img = cv2.merge((h, s, v))

        # Back to BGR colorspace.
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2BGR)
    else:
        out_img = in_img
    return out_img


def RandomOrderChannels(in_img, random_order_prob):
    prob = npr.random()
    if prob < random_order_prob:
        # Split the image to 3 channels.
        channels = cv2.split(out_img)
        assert len(channels) == 3

        # Shuffle the channels.
        channels = npr.shuffle(channels)
        out_img = cv2.merge(channels)
    else:
        out_img = in_img
    return out_img
