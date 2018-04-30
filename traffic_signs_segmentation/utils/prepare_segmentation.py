"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
""" Generate images and labels for segmentation, expects a dataset created by clean_and_split.py
    Input dataset dataset is used to generate:
    - feature images with the resolution defined in ml.cfg
    - mask label images (indexed PNG)
    - false positives markings are filtered via load_all_rois()
"""

from multiprocessing import Pool

import argparse
import logging

import apollo_python_common.log_util as log_util

import dataset
import utils


def do_work(generator, payload):
    pool = Pool(16)
    pool.map(generator, payload.items())


def main():

    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    logger.info('Prepare segmentation')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_path",
                        type=str, required=True)
    args = parser.parse_args()

    rois = dataset.load_all_rois(args.input_path, dataset.no_false_positives)
    utils.make_dirs([args.output_path])

    dirs = ["/images/", "/labels/"]
    dirs = [args.output_path + d for d in dirs]
    utils.make_dirs(dirs)

    do_work(dataset.FeatureGenerator(args.output_path + "/images/"), rois)
    do_work(dataset.LabelGenerator(args.output_path + "/labels/"), rois)


if __name__ == "__main__":
    main()
