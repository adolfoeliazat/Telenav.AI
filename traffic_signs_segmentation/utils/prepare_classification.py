"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Generates data for classification. The data is split in folders with the name of the category
There is a special class for false positives
"""

import argparse
import os
import logging

from multiprocessing import Pool

import apollo_python_common.log_util as log_util
import configuration
import dataset
import utils


def do_work(generator, payload):
    pool = Pool(16)
    pool.map(generator, payload.items())


class ImageValidFilter(object):
    """
    Filters existing images
    """

    def __init__(self, path, images):
        self.path = path
        self.images = images

    def __call__(self, image):
        return os.path.basename(image) in self.images


def generate_rois(images_path, rois, output_path):
    """
    Generates images used for classification
    """
    images = set([os.path.basename(img)
                  for img in utils.collect_images(images_path)])

    classification_generator = dataset.ClassificationGenerator(
        output_path, ImageValidFilter(images_path, images))
    do_work(classification_generator, rois)


def main():

    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    logger.info('Prepare classification')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_path",
                        type=str, required=True)
    args = parser.parse_args()

    conf = configuration.load_configuration()
    dirs = [str(v) for _, v in conf.type2class_id.items()] + ["false_pos"]
    dirs = [args.output_path + "/" + d for d in dirs]
    utils.make_dirs(dirs)

    rois = dataset.load_all_rois(args.input_path, dataset.no_false_positives)
    generate_rois(os.path.dirname(args.input_path), rois, args.output_path)

    rois = dataset.load_all_rois(args.input_path, dataset.only_false_positives)
    generate_rois(os.path.dirname(args.input_path), rois, args.output_path)


if __name__ == "__main__":
    main()
