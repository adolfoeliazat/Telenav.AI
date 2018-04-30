"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import argparse
import sys
import logging
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing

import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as meta
from apollo_python_common.generate_model_statistics import get_model_statistics
import apollo_python_common.io_utils as io_utils


MIN_SIZE = 15


def get_threshold_for_class(data):
    logger = logging.getLogger(__name__)
    expected_dict, actual_dict, class_name = data
    class_names = set(meta.get_class_names_from_metadata_dictionary(expected_dict)).union(
        set(meta.get_class_names_from_metadata_dictionary(actual_dict)))
    thresholds_per_class = dict([(c, 0.5) for c in class_names])
    logger.info("Searching best threshold for class {}".format(class_name))
    best_acc = 0
    best_threshold = 0.1
    for thr in tqdm(np.arange(0.1, 1.01, 0.01)): # searching best in range of 0.1 to 1 with the step 0.01
        thresholds_per_class[class_name] = round(thr, 2)
        confident_dict = get_confident_rois(actual_dict, thresholds_per_class)
        statistics_dict = get_model_statistics(expected_dict, confident_dict, None, MIN_SIZE)
        accuracy = statistics_dict['Total'].accuracy()
        if accuracy > best_acc:
            best_acc = accuracy
            best_threshold = thresholds_per_class[class_name]
    thresholds_per_class[class_name] = best_threshold
    logger.info('Best threshold for class {}: {}'.format(class_name, best_threshold))
    return class_name, best_threshold


def get_thresholds(expected_dict, actual_dict):
    logger = logging.getLogger(__name__)
    class_names = set(meta.get_class_names_from_metadata_dictionary(expected_dict)).union(
        set(meta.get_class_names_from_metadata_dictionary(actual_dict)))
    data = [(expected_dict, actual_dict, class_name) for class_name in class_names]
    threads_number = multiprocessing.cpu_count() // 2
    pool = Pool(threads_number)
    thresholds = pool.map(get_threshold_for_class, data)
    pool.close()
    thresholds_per_class = dict(thresholds)
    # Evaluating with best thresholds
    confident_dict = get_confident_rois(actual_dict, thresholds_per_class)
    statistics_dict = get_model_statistics(expected_dict, confident_dict, None, MIN_SIZE)
    for key, statistics_element in statistics_dict.items():
        statistic_line = "{} {}".format(key, statistics_element)
        logger.info(statistic_line)
    return thresholds_per_class


def get_confident_rois(rois_dict, thresholds_per_class):
    selected_dict = dict()
    for file_name, rois in rois_dict.items():
        rois_list = list()
        selected_dict[file_name] = rois_list
        for roi in rois:
            if roi.detections[0].confidence > thresholds_per_class[meta.get_roi_type_name(roi.type)]:
                rois_list.append(roi)
    return selected_dict


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate classes confidence thresholds for best accuracy.')

    parser.add_argument("-e", "--expected_rois_file",
                        type=str, required=True)
    parser.add_argument("-a", "--actual_rois_file",
                        type=str, required=True)
    parser.add_argument("-o", "--result_file", type=str, required=True,
                        help='The file where to store the pickled dictionary with thresholds.',
                        default='./classes_thresholds.pkl')
    return parser.parse_args(args)


def main():
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # parse arguments
    args = sys.argv[1:]
    args = parse_args(args)
    # get metadata
    expected_metadata = meta.read_metadata(args.expected_rois_file)
    actual_metadata = meta.read_metadata(args.actual_rois_file)
    # calculate best thresholds
    best_thresholds = get_thresholds(meta.create_metadata_dictionary(expected_metadata, True),
                            meta.create_metadata_dictionary(actual_metadata, True))
    io_utils.json_dump(best_thresholds, args.result_file)


if __name__ == '__main__':
    main()
