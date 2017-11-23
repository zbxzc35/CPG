#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""Test a Fast R-CNN network on an image database."""

import _init_paths
from wsl.test import test_net_ensemble
from wsl.test import test_net_ensemble2
from wsl.config import cfg_wsl
from configure import cfg, cfg_basic_generation, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time
import os
import sys
import numpy as np
import csv


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--result',
        dest='result_dirs',
        help='multi test result dirs',
        default=None,
        nargs='+')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str)
    parser.add_argument(
        '--wait',
        dest='wait',
        help='wait until net file exists',
        default=True,
        type=bool)
    parser.add_argument(
        '--imdb',
        dest='imdb_name',
        help='dataset to test',
        default='voc_2007_test',
        type=str)
    parser.add_argument(
        '--comp',
        dest='comp_mode',
        help='competition mode',
        action='store_true')
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--num_dets',
        dest='max_per_image',
        help='max number of detections per image',
        default=10000,
        type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    cfg_basic_generation(cfg_wsl)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    test_net_ensemble(args.result_dirs, imdb, max_per_image=args.max_per_image)
