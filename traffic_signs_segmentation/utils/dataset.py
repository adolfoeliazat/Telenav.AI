"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import numpy as np
import os

from collections import defaultdict
from PIL import Image
import logging

import apollo_python_common.protobuf.orbb_metadata_pb2 as orbb_metadata_pb2
import apollo_python_common.proto_api as roi_metadata
import configuration
import image
import utils

# TODO: unify load_rois/load_all_rois, same thing with some filtering
conf = configuration.load_configuration()

def no_false_positives(roi):
    """
    Filters false positives
    :param roi: a protobuf roi
    :return: true if roi is a false positive
    """
    return roi.validation == orbb_metadata_pb2.FALSE_POSITIVE


def only_false_positives(roi):
    """
    Filters true positives
    :param roi: a protobuf roi
    :return: true if roi is a true positive
    """
    return roi.validation != orbb_metadata_pb2.FALSE_POSITIVE


def dummy_roi_filter(roi):
    return False


def load_all_rois(input_path, roi_filter=dummy_roi_filter):
    """
    Loads rois that pass the roi_filter
    :param input_path: path to metadata roi
    :param roi_filter: filters false positives or true positives
    :return: selected rois
    """
    metadata = roi_metadata.read_metadata(input_path)
    rois = defaultdict(list)
    for image in metadata.images:
        current_image = os.path.normpath(
            os.path.dirname(input_path) + "/" + image.metadata.image_path)
        if not os.path.exists(current_image):
            continue

        for roi in image.rois:
            if roi_filter(roi):
                continue
            rois[current_image].append(Roi(roi))
    return rois


def load_valid_rois(input_path):
    """
    Creates a valid metadata by selecting only the images and rois in which all the rois in an image meet the size
    condition
    :param input_path: rois path
    :return:
    """

    metadata = roi_metadata.read_metadata(input_path)
    valid_metadata = orbb_metadata_pb2.ImageSet()
    valid_metadata.name = ""
    for image_rois in metadata.images:
        current_image = os.path.dirname(input_path) + "/" + image_rois.metadata.image_path
        if not os.path.exists(current_image):
            continue

        width, height = image.get_resolution(current_image)

        if not image.valid_image(width, height, conf.ratio, conf.epsilon):
            logger = logging.getLogger(__name__)
            logger.info("invalid image because aspect ratio {} {} {}".format(width, height, image_rois.metadata.image_path))
            continue

        scaling_factor, width_crop_size = image.get_crop_scale(width, height)
        # width_crop_size containes the scaled down value
        width_crop_size = int(width_crop_size * scaling_factor)
        image_min_size = scaling_factor * conf.min_size

        valid_rois = []
        for roi in image_rois.rois:

            roi.rect.br.row = min(roi.rect.br.row, height - 1)
            roi.rect.br.col = min(roi.rect.br.col, width - width_crop_size - 1)
            roi.rect.tl.col = max(roi.rect.tl.col, width_crop_size)

            r_width = roi.rect.br.col - roi.rect.tl.col
            r_height = roi.rect.br.row - roi.rect.tl.row
            if r_width >= image_min_size and r_height >= image_min_size:
                valid_roi = orbb_metadata_pb2.Roi()
                valid_roi.rect.CopyFrom(roi.rect)
                valid_roi.algorithm = "DNN"
                valid_roi.manual = False
                valid_roi.type = roi.type
                valid_roi.validation = roi.validation
                valid_rois.append(valid_roi)
            else:
                continue

        if valid_rois :
            valid_image_rois = valid_metadata.images.add()
            valid_image_rois.rois.extend(valid_rois)
            valid_image_rois.metadata.image_path = image_rois.metadata.image_path
            valid_image_rois.metadata.trip_id = ""
            valid_image_rois.metadata.region =""
            valid_image_rois.metadata.image_index = -1
    return valid_metadata


# TODO: namedtuple
class Roi:
    """
    Stub class for metadata Roi class
    """
    def __init__(self, orbb_roi):
        self.r_type = orbb_roi.type
        self.tl_row = orbb_roi.rect.tl.row
        self.br_row = orbb_roi.rect.br.row
        self.tl_col = orbb_roi.rect.tl.col
        self.br_col = orbb_roi.rect.br.col
        self.false_pos = (orbb_roi.validation
                          == orbb_metadata_pb2.FALSE_POSITIVE)


class LabelGenerator:
    """
    Generates image labels used for segmentation dataset
    """

    def __init__(self, labels_path):
        self.labels_path = labels_path

    def __call__(self, image_rois):
        # TODO: cv2
        img = Image.open(image_rois[0])
        label = np.zeros((img.size[1], img.size[0]), np.uint8)

        for roi in image_rois[1]:
            # class_id = conf.type2class_id[roi.r_type]
            # For the moment we only support binary segmentation
            class_id = 1
            label[roi.tl_row:roi.br_row, roi.tl_col:roi.br_col] = class_id

        label_img = Image.fromarray(label, 'P')
        label_img.putpalette(conf.palette)
        label_path = self.labels_path + utils.to_png(image_rois[0])
        image.resize_image(label_img, (conf.image_size[0] // 2, conf.image_size[1] // 2))[0].save(label_path)

        label_file_path =  self.labels_path + "/val.txt"
        with open(label_file_path, "a+") as label_file:
            label_file.write(label_path[len(self.labels_path): ])
            label_file.write("\n")

class FeatureGenerator:
    """
    Feature image generator used for segmentation dataset
    """
    def __init__(self, features_path):
        self.features_path = features_path

    def __call__(self, image_rois):
        # TODO: histogram equalization per channel
        # TODO: cv2
        img = Image.open(image_rois[0])
        img_path = os.path.join(self.features_path,
                                os.path.basename(image_rois[0]))
        image.resize_image(img, (conf.image_size[0] // 2, conf.image_size[1] // 2))[0].save(img_path)
        feature_file_path = self.features_path + "/val.txt"
        with open(feature_file_path, "a+") as feature_file:
            feature_file.write(img_path[len(self.features_path): ])
            feature_file.write("\n")


class ClassificationGenerator:
    """
    Generates the dataset for classification
    """
    def __init__(self, classification_path, image_filter):
        self.classification_path = classification_path
        self.valid_image = image_filter

    def __call__(self, image_rois):
        if not self.valid_image(image_rois[0]):
            return

        # TODO: replace with cv2
        img = np.asarray(Image.open(image_rois[0]))
        i = 0
        detection_file = self.classification_path + "/val.txt"
        with open(detection_file, "a+") as label_file:
            for roi in image_rois[1]:
                # TODO: check why?
                try:
                    class_id = "false_pos" if roi.false_pos else conf.type2class_id[roi.r_type]
                    roi_img = img[roi.tl_row:roi.br_row, roi.tl_col:roi.br_col]

                    r, g, b = roi_img.T
                    roi_img_bgr = np.array((b,g,r)).T

                    roi_img_path = self.classification_path + "/" + str(class_id) + "/"
                    roi_img_path += str(i) + "_" + utils.to_png(image_rois[0])
                    # TODO: replace with cv2
                    Image.fromarray(roi_img_bgr).save(roi_img_path)
                    i = i + 1
                    id = 0
                    if class_id == "false_pos" :
                        id = conf.invalid_id
                    else:
                        id = class_id
                    label_file.write(roi_img_path[len(self.classification_path): ] + " " +str(id))
                    label_file.write("\n")
                except ValueError:
                    continue
                    print("Rect goes outside in image {}.".format(image_rois[0]))
                except KeyError:
                    continue


class DetectionGenerator:
    """
    Generate for each image a text file containing the roi type and box,
    according to the following format.
    Useful for importing into the DIGITS detection dataset.
    https://github.com/umautobots/vod-converter/blob/master/vod_converter/kitti.py
    http://www.cvlibs.net/datasets/kitti/eval_object.php
    """

    def __init__(self, detection_path, image_filter):
        self.detection_path = detection_path
        self.valid_image = image_filter

    def __call__(self, image_rois):
        image_path = image_rois[0]
        if not self.valid_image(image_path):
            return

        width, height = image.get_resolution(image_path)
        scaling_factor, width_crop_size = image.get_crop_scale(width, height)
        width_crop_size = int(width_crop_size * scaling_factor)

        detection_file = self.detection_path
        detection_file += utils.get_filename(image_path) + ".txt"
        with open(detection_file, "w+") as label_file:
            for roi in image_rois[1]:
                class_name = conf.type2name[int(roi.r_type)]
                tl_col = int((roi.tl_col - width_crop_size) / scaling_factor)
                tl_row = int(roi.tl_row / scaling_factor)
                br_col = int((roi.br_col - width_crop_size) / scaling_factor)
                br_row = int(roi.br_row / scaling_factor)
                label_file.write(class_name + (3 * " 0") + " "
                                 + str(tl_col) + " " + str(tl_row) + " "
                                 + str(br_col) + " " + str(br_row)
                                 + (7 * " 0"))
                label_file.write("\n")