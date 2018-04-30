"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
 Clean up rois outside image.
 Clean up rois small than threshold.
 Clean up missing images.
 Clean up images with bad aspect ratio.
"""

import argparse
import os
import logging

import apollo_python_common.log_util as log_util

import dataset



def main():

    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    logger.info('Clean dataset')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_path",
                        type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    metadata = dataset.load_valid_rois(args.input_path)

    with open(args.output_path + "/rois.bin", "wb") as f_meta:
        f_meta.write(metadata.SerializeToString())

    with open(args.output_path + "/rois.txt", "w") as f_meta:
        f_meta.write(str(metadata))


if __name__ == "__main__":
    main()
