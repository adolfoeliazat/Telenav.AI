"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
 Split dataset in trainval|test by the split ratio defined in configuration
"""

import argparse
import random
import os
import shutil

import apollo_python_common.protobuf.orbb_metadata_pb2 as orbb_metadata_pb2
import apollo_python_common.proto_api as roi_metadata

import configuration
import utils


def generate_dataset(image_set, input_path, features, output_path):

    new_image_set = orbb_metadata_pb2.ImageSet()
    new_image_set.name = ""
    for image in image_set.images:
        if image.metadata.image_path not in features:
            continue
        new_image = new_image_set.images.add()
        new_image.CopyFrom(image)
        shutil.copy(input_path + "/" + image.metadata.image_path, output_path + "/" + image.metadata.image_path)

    with open(output_path + "/rois.bin", "wb") as f_meta:
        f_meta.write(new_image_set.SerializeToString())

    with open(output_path + "/rois.txt", "w") as f_meta:
        f_meta.write(str(new_image_set))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_path",
                        type=str, required=True)
    args = parser.parse_args()

    dirs = ["trainval/", "test/"]
    dirs = [args.output_path + d for d in dirs]
    utils.make_dirs(dirs)

    conf = configuration.load_configuration()
    images = utils.collect_images(os.path.dirname(args.input_path))

    # reproducible datasets
    random.seed(1337)
    random.shuffle(images)

    rois = roi_metadata.read_metadata(args.input_path)

    features_train = set(os.path.basename(i) for i in images[0:int(len(images) * conf.split_ratio)])
    generate_dataset(rois, os.path.dirname(args.input_path), features_train, dirs[0])

    features_test = set(os.path.basename(i) for i in images[int(len(images) * conf.split_ratio):])
    generate_dataset(rois, os.path.dirname(args.input_path), features_test, dirs[1])


if __name__ == "__main__":
    main()
