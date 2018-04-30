"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import argparse
import os
import cv2
from collections import defaultdict
import random

import apollo_python_common.proto_api as proto_api
import orbb_definitions_pb2
import apollo_python_common.rectangle as rectangle


class Statistics(object):
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.miss_classified = 0
        self.iou = 0

    def recall(self):
        if self.true_positives + self.false_negatives > 0:
            return self.true_positives / (self.true_positives + self.false_negatives)
        else:
            return 0

    def precision(self):
        if (self.true_positives + self.false_positives) > 0:
            return self.true_positives / (self.true_positives + self.false_positives + self.miss_classified)
        else:
            return 0

    def accuracy(self):
        if self.true_positives + self.false_positives + self.miss_classified > 0:
            return self.true_positives / (
                    self.true_positives + self.false_positives + self.false_negatives + self.miss_classified)
        else:
            return 0

    def __str__(self):
        return "tp = {} fp = {} fn = {} mc = {} precision = {:.4f} recall = {:.4f} accuracy = {:4f} iou = {:4f}" \
            .format(self.true_positives, self.false_positives, self.false_negatives, self.miss_classified,
                    self.precision(), self.recall(), self.accuracy(), self.iou)


def valid_size(roi, min_size):
    rect = rectangle.Rectangle(roi.rect.tl.col, roi.rect.tl.row, roi.rect.br.col, roi.rect.br.row)
    return min(rect.width(), rect.height()) >= min_size


def copy_image_roi_to_folder(images_folder, destination_sub_folder, file_name, roi):
    if images_folder is None:
        return
    img = cv2.imread(images_folder + "/" + file_name)
    if not img is None:
        crop = img[roi.rect.tl.row:roi.rect.br.row, roi.rect.tl.col:roi.rect.br.col].copy()
        out_file_dir = images_folder + "/" + orbb_definitions_pb2.Mark.Name(
            roi.type) + "/" + destination_sub_folder + "/"
        if not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir)
        file_name = file_name + str(roi.rect.tl.row) + str(roi.rect.tl.col) + str(roi.rect.br.row) + str(
            roi.rect.br.col)
        cv2.imwrite(out_file_dir + file_name + ".jpg", crop)


def rois_intersect(roi_a, roi_b):
    rect_a = rectangle.Rectangle(roi_a.rect.tl.col, roi_a.rect.tl.row, roi_a.rect.br.col, roi_a.rect.br.row)
    rect_b = rectangle.Rectangle(roi_b.rect.tl.col, roi_b.rect.tl.row, roi_b.rect.br.col, roi_b.rect.br.row)
    intersection_area = rect_a.get_overlapped_rect(rect_b).area()
    if intersection_area:
        percentage = intersection_area / rect_a.area()
        return percentage


def select_best_area_match(actual_rois, expected_roi):
    max_percentage = 0
    actual_roi = None
    for rois_a in actual_rois:
            intersection_result = rois_intersect(rois_a, expected_roi)
            if intersection_result and intersection_result > max_percentage:
                max_percentage = intersection_result
                actual_roi = rois_a
    return actual_roi


def select_true_positives(actual_roi, expected_roi, statistics_dict, min_size, images_folder, file_name):
    if actual_roi.type == expected_roi.type and (valid_size(actual_roi, min_size) or valid_size(expected_roi, min_size)):
        type_string = orbb_definitions_pb2.Mark.Name(expected_roi.type)
        statistics_dict[type_string].true_positives += 1
        rect_a = rectangle.Rectangle(actual_roi.rect.tl.col, actual_roi.rect.tl.row, actual_roi.rect.br.col,
                           actual_roi.rect.br.row)
        rect_e = rectangle.Rectangle(expected_roi.rect.tl.col, expected_roi.rect.tl.row, expected_roi.rect.br.col,
                           expected_roi.rect.br.row)
        iou = rect_a.intersection_over_union(rect_e)
        statistics_dict[type_string].iou += iou
        copy_image_roi_to_folder(images_folder, "tp", file_name, actual_roi)


def select_miss_classified(actual_roi, expected_roi, statistics_dict, min_size, images_folder, file_name):
    if actual_roi.type != expected_roi.type and (valid_size(actual_roi, min_size) or valid_size(expected_roi, min_size)):
            type_string = orbb_definitions_pb2.Mark.Name(expected_roi.type)
            statistics_dict[type_string].miss_classified += 1
            copy_image_roi_to_folder(images_folder, "mc", file_name, expected_roi)


def select_false_negative(expected_roi, statistics_dict, min_size, images_folder, file_name):
    if valid_size(expected_roi, min_size):
        type_string = orbb_definitions_pb2.Mark.Name(expected_roi.type)
        statistics_dict[type_string].false_negatives += 1
        copy_image_roi_to_folder(images_folder, "fn", file_name, expected_roi)


def select_false_positives(actual_rois, expected_rois, statistics_dict, min_size, images_folder, file_name):
    for rois_a in actual_rois:
        found = False
        type_string = orbb_definitions_pb2.Mark.Name(rois_a.type)
        for rois_e in expected_rois:
            type_string = orbb_definitions_pb2.Mark.Name(rois_a.type)
            if rois_intersect(rois_e, rois_a):
                found = True
        if not found and valid_size(rois_a, min_size):
            statistics_dict[type_string].false_positives += 1
            copy_image_roi_to_folder(images_folder, "fp", file_name, rois_a)


def get_model_statistics(expected_dictionary, actual_dictionary, images_folder, min_size):
    statistics_dict = defaultdict(Statistics)
    for expected_file in expected_dictionary.keys():
        if expected_file not in actual_dictionary.keys():
            print("Error miss match file " + expected_file)
            actual_rois = list()
        else:
            actual_rois = actual_dictionary[expected_file]
        expected_rois = expected_dictionary[expected_file]
        for expected_roi in expected_rois:
            actual_roi = select_best_area_match(actual_rois, expected_roi)
            if actual_roi:
                select_true_positives(actual_roi, expected_roi, statistics_dict, min_size, images_folder, expected_file)
                select_miss_classified(actual_roi, expected_roi, statistics_dict, min_size, images_folder, expected_file)
            else:
                select_false_negative(expected_roi, statistics_dict, min_size, images_folder, expected_file)
        select_false_positives(actual_rois, expected_rois, statistics_dict, min_size, images_folder, expected_file)

    total_statistic = Statistics()
    for key, statistic in statistics_dict.items():
        if key not in ['INVALID']:
            total_statistic.true_positives += statistic.true_positives
            total_statistic.false_positives += statistic.false_positives
            total_statistic.false_negatives += statistic.false_negatives
            total_statistic.miss_classified += statistic.miss_classified
            total_statistic.iou += statistic.iou
    if total_statistic.true_positives > 0:
        total_statistic.iou = total_statistic.iou / total_statistic.true_positives
    else:
        total_statistic.iou = 0
    statistics_dict["Total"] = total_statistic
    return statistics_dict


def output_statistics(statistics_dict, result_file):
    with open(result_file, "w") as file:
        for key, statistics_element in statistics_dict.items():
            statistic_line = "{} {}".format(key, statistics_element)
            print(statistic_line)
            file.write(statistic_line + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expected_rois_file",
                        type=str, required=True)
    parser.add_argument("-a", "--actual_rois_file",
                        type=str, required=True)
    parser.add_argument("-i", "--images_folder",
                        type=str, required=False)
    parser.add_argument("-o", "--result_file",
                        type=str, required=False)
    args = parser.parse_args()

    expected_rois_file = args.expected_rois_file
    actual_rois_file = args.actual_rois_file
    images_folder = args.images_folder

    expected_metadata = proto_api.read_metadata(expected_rois_file)
    actual_metadata = proto_api.read_metadata(actual_rois_file)

    random.seed(1337)

    expected_dictionary = proto_api.create_metadata_dictionary(expected_metadata, True)
    actual_dictionary = proto_api.create_metadata_dictionary(actual_metadata, False)
    min_size = 25
    statistics_dict = get_model_statistics(expected_dictionary, actual_dictionary, images_folder, min_size)
    output_statistics(statistics_dict, args.result_file)


if __name__ == "__main__":
    main()

