"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Auguments the data with the following:
slide window around roi, crop roi, rotate image, darken, brighten
"""

import argparse
import os
from random import randint
import shutil
from multiprocessing import Pool
import multiprocessing
from functools import partial
import numpy as np
import logging

import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
import orbb_metadata_pb2
import image as image_operations
import utils


# TODO: create all transformations as a suite of plugins
# TODO: create plugin interface so anyone can extend it
# TODO: combine transformations
# TODO: remove duplication
# TODO: introduce logging
# TODO: multiprocessing


def create_image_proto():
    image_proto = orbb_metadata_pb2.Image()
    image_proto.metadata.trip_id = ""
    image_proto.metadata.image_index = -1
    image_proto.metadata.region = ""
    return image_proto

def crop_roi(roi):
    """
    Crops the roi in left, right, up, down direction
    :param roi:
    :return:
    """

    direction = randint(0, 3)
    if direction == 0:
        crop_size = (roi.rect.br.col - roi.rect.tl.col) / 5
        roi.rect.tl.col = int(roi.rect.tl.col + crop_size)
    elif direction == 1:
        crop_size = (roi.rect.br.col - roi.rect.tl.col) / 5
        roi.rect.br.col = int(roi.rect.br.col - crop_size)
    elif direction == 2:
        crop_size = (roi.rect.br.row - roi.rect.tl.row) / 5
        roi.rect.tl.row = int(roi.rect.tl.row + crop_size)
    elif direction == 3:
        crop_size = (roi.rect.br.row - roi.rect.tl.row) / 5
        roi.rect.br.row = int(roi.rect.br.row - crop_size)

    roi.manual = True


def crop(image, metadata_list):
    """
    Crops the rois from an image and appends the new rois in metadata list
    :param image: rois contained by an image
    :param metadata_list: the appended metadata list
    :return: new file name
    """
    image = create_image_proto()
    image.metadata.image_path = "crop_20_" + image.metadata.image_path
    for roi in image.rois:
        roi_new = image.rois.add()
        roi_new.CopyFrom(roi)
        crop_roi(roi_new)
    metadata_list.append(image)
    return image.metadata.image_path


def slide_roi(roi, image_size):
    """
    Slides the roi in left, right, up, down direction
    :param roi:
    :return:
    """
    height, width, _ = image_size
    # left, right, up, down
    direction = randint(0, 1)
    if direction == 0:
        crop_size = (roi.rect.br.col - roi.rect.tl.col) / 5
        roi.rect.tl.col = max(0, int(roi.rect.tl.col - crop_size))
        roi.rect.br.col = int(roi.rect.br.col - crop_size)
    elif direction == 1:
        crop_size = (roi.rect.br.col - roi.rect.tl.col) / 5
        roi.rect.br.col = min(width - 1, int(roi.rect.br.col + crop_size))
        roi.rect.tl.col = int(roi.rect.tl.col + crop_size)
    elif direction == 2:
        crop_size = (roi.rect.br.row - roi.rect.tl.row) / 5
        roi.rect.tl.row = max(0, int(roi.rect.tl.row - crop_size))
        roi.rect.br.row = int(roi.rect.br.row - crop_size)
    elif direction == 3:
        crop_size = (roi.rect.br.row - roi.rect.tl.row) / 5
        roi.rect.br.row = min(height - 1, int(roi.rect.br.row + crop_size))
        roi.rect.tl.row = int(roi.rect.tl.row + crop_size)
    roi.manual = True


def slide_window(image, image_size, metadata_list):
    """
    Slides the rois from an image and appends the new rois in metadata list
    :param image: rois contained by an image
    :param metadata_list: the appended metadata list
    :return: new file name
    """
    image = create_image_proto()
    image.metadata.trip_id = ""
    image.metadata.image_index = -1
    image.metadata.region = ""
    image.metadata.image_path = "slide_" + image.metadata.image_path
    for roi in image.rois:
        roi_new = image.rois.add()
        roi_new.CopyFrom(roi)
        slide_roi(roi_new, image_size)
    metadata_list.append(image)
    return image.metadata.image_path


def rotate_ccw(image, image_size, rotation_matrix, metadata_list):
    """
    Modifies the rois using a counter clock wise rotation
    """

    height, width, _ = image_size
    new_image = create_image_proto()
    new_image.metadata.image_path = "rotate_ccw_" + image.metadata.image_path
    for roi in image.rois:
        points_to_transform = np.array([
            [roi.rect.tl.col, roi.rect.tl.row, 1],
            [roi.rect.br.col, roi.rect.br.row, 1],
            [roi.rect.br.col, roi.rect.tl.row, 1],
            [roi.rect.tl.col, roi.rect.br.row, 1]
        ])
        points = rotation_matrix.dot(points_to_transform.T)

        try:
            roi_new = orbb_metadata_pb2.Roi()
            roi_new.CopyFrom(roi)
            roi_new.rect.tl.col = max(0, int(points[0, 0]))
            roi_new.rect.tl.row = max(0, int(points[1, 2]))
            roi_new.rect.br.col = min(width - 1, int(points[0, 1]))
            roi_new.rect.br.row = min(height - 1, int(points[1, 3]))

            # Shit protobuf python interface
            valid_roi = new_image.rois.add()
            valid_roi.CopyFrom(roi_new)
        except ValueError:
            print("Rect goes outside in image {}.".format(new_image.metadata.image_path))
    metadata_list.append(new_image)
    return new_image.metadata.image_path


def rotate_cw(image, image_size, rotation_matrix, metadata_list):
    """
    Modifies the rois using a clock wise rotation
    """

    height, width, _ = image_size
    new_image = create_image_proto()
    new_image.metadata.image_path = "rotate_cw_" + image.metadata.image_path
    for roi in image.rois:
        points_to_transform = np.array([
            [roi.rect.tl.col, roi.rect.tl.row, 1],
            [roi.rect.br.col, roi.rect.br.row, 1],
            [roi.rect.br.col, roi.rect.tl.row, 1],
            [roi.rect.tl.col, roi.rect.br.row, 1]
        ])
        points = rotation_matrix.dot(points_to_transform.T)

        try:
            roi_new = orbb_metadata_pb2.Roi()
            roi_new.CopyFrom(roi)
            roi_new.rect.tl.col = max(0, int(points[0, 3]))
            roi_new.rect.tl.row = max(0, int(points[1, 0]))
            roi_new.rect.br.col = min(width - 1, int(points[0, 2]))
            roi_new.rect.br.row = min(height - 1, int(points[1, 1]))

            # Shit protobuf python interface
            valid_roi = new_image.rois.add()
            valid_roi.CopyFrom(roi_new)
        except ValueError:
            print("Rect goes outside in image {}.".format(new_image.metadata.image_path))
    metadata_list.append(new_image)
    return new_image.metadata.image_path


def darken_image(image, output_metadata):
    """
    Modifies the rois to point to the new darken image
    """
    darken_image_file = "darken_image_" + image.metadata.image_path
    darken_image = orbb_metadata_pb2.Image()
    darken_image.CopyFrom(image)
    darken_image.metadata.image_path = darken_image_file
    output_metadata.append(darken_image)
    return darken_image_file


def brighten_image(image, output_metadata):
    """
    Modifies the rois to point to the new brighten image
    """
    brighten_image_file = "brighten_image_" + image.metadata.image_path
    bright_image = orbb_metadata_pb2.Image()
    bright_image.CopyFrom(image)
    bright_image.metadata.image_path = brighten_image_file
    output_metadata.append(bright_image)
    return brighten_image_file


def paralel_augment(image_proto, in_path, out_path):
    """
    Augment data method called in parallel by multiple threads
    :param image: rois metadata
    :param in_path: input images folder
    :param out_path: out images folder
    :return: generated metadata
    """

    output_metadata = []
    image_path = in_path + "/" + image_proto.metadata.image_path
    image_data = image_operations.load_image(image_path)
    
    image_size = image_data.shape
    shutil.copy(image_path, out_path + "/")

    crop_file = crop(image_proto, output_metadata)
    
    shutil.copy(image_path, out_path + "/" + crop_file)

    slide_file = slide_window(image_proto, image_size, output_metadata)
    
    shutil.copy(image_path, out_path + "/" + slide_file)

    ccw_rotated_image, ccw_rotation_matrix = image_operations.rotate_image(
        image_data, 5)
    
    ccw_rotated_file = rotate_ccw(
        image_proto, image_size, ccw_rotation_matrix, output_metadata)

    image_operations.save_image(
        ccw_rotated_image, out_path + "/" + ccw_rotated_file)
    cw_rotated_image, cw_rotation_matrix = image_operations.rotate_image(
        image_data, -5)
    
    cw_rotated_file = rotate_cw(
        image_proto, image_size, cw_rotation_matrix, output_metadata)

    image_operations.save_image(cw_rotated_image, out_path + "/" + cw_rotated_file)
    darken_image_data = image_operations.adjust_gamma(image_data, 0.9)

    darken_image_file = darken_image(image_proto, output_metadata)

    image_operations.save_image(
        darken_image_data, out_path + "/" + darken_image_file)
   
    brighten_image_data = image_operations.adjust_gamma(image_data, 1.1)

    brighten_image_file = brighten_image(image_proto, output_metadata)

    image_operations.save_image(
        brighten_image_data, out_path + "/" + brighten_image_file)
    return output_metadata


def augment(metadata, input_path, threads_number, output_path):
    """
    Augments data in parallel
    :param metadata: input metadata
    :param input_path: input images folder
    :param threads_number: number of threads
    :param output_path: output images path
    :return: generated metadata
    """

    output_metadata = orbb_metadata_pb2.ImageSet()
    output_metadata.CopyFrom(metadata)
    pool = Pool(threads_number)
    metadata_list = pool.map(partial(paralel_augment, in_path=input_path, out_path=output_path), metadata.images)
    for partial_list in metadata_list:
        for image_roi in partial_list:
            image_added = output_metadata.images.add()
            image_added.CopyFrom(image_roi)
    return output_metadata


def main():

    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    logger.info('Dataset Augmentation')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_path",
                        type=str, required=True)

    args = parser.parse_args()
    utils.make_dirs([args.output_path])
    threads_number = int(multiprocessing.cpu_count() / 2)
    metadata = proto_api.read_metadata(args.input_path)
    output_metadata = augment(
        metadata, os.path.dirname(args.input_path), threads_number, args.output_path)
    output_metadata.name = ""
    proto_api.serialize_metadata(output_metadata, args.output_path)


if __name__ == "__main__":
    main()
