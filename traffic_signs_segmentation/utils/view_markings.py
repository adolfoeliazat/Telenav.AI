"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Blends the feature images and the mask/label images
"""

import argparse
import os
import sys

from PIL import Image

import utils


def blend_image(background_path, mask_path):
    """
    Blends two images
    :param background_path: background image path
    :param mask_path: mask image path
    :return: blended image
    """
    background = Image.open(background_path)
    mask = Image.open(mask_path)

    background = background.convert("RGBA")
    mask = mask.convert("RGBA")

    return Image.blend(background, mask, 0.5)


def blend_images(backgrounds_path, masks_path, output_path):
    """
    Blend a list of background and mask images
    :param backgrounds_path: background images path
    :param masks_path: mask images path
    :param output_path: blended images output path
    :return:
    """
    backgrounds = utils.collect_images(backgrounds_path)
    for bg in backgrounds:
        blend = blend_image(bg, masks_path + "/" + os.path.basename(bg))
        blend.save(output_path + "/" + os.path.basename(bg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dataset_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_path",
                        type=str, required=False, default="data/")
    args = parser.parse_args()

    if not utils.valid_dataset(args.dataset_path):
        print("Invalid dataset")
        sys.exit(-1)

    utils.make_dirs([args.output_path])

    blend_images(
        args.dataset_path + "/train_features/",
        args.dataset_path + "/train_labels/",
        args.output_path)

    blend_images(
        args.dataset_path + "/test_features/",
        args.dataset_path + "/test_labels/",
        args.output_path)


if __name__ == "__main__":
    main()
