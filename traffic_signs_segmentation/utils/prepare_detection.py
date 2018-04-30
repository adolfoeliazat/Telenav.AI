"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Generates data in the format needed by caffe for detection
"""

import argparse
import os

from multiprocessing import Pool

import dataset
import utils


def do_work(generator, payload):
    pool = Pool(16)
    pool.map(generator, payload.items())


class ImageValidFilter:
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
    Generates images used for object detection
    """

    images = set([os.path.basename(img)
                  for img in utils.collect_images(images_path)])

    detection_generator = dataset.DetectionGenerator(
        output_path, ImageValidFilter(images_path, images))
    do_work(detection_generator, rois)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_path",
                        type=str, required=True)

    args = parser.parse_args()

    utils.make_dirs([args.output_path + "/annotations/"])

    rois = dataset.load_all_rois(args.input_path, dataset.no_false_positives)
    generate_rois(os.path.dirname(args.input_path), rois,
                  args.output_path + "/annotations/")


if __name__ == "__main__":
    main()
