"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import keras
import os
import logging
import apollo_python_common.proto_api as meta
from apollo_python_common.generate_model_statistics import get_model_statistics
from retinanet.utils import predict_folder
from retinanet.traffic_signs_generator import RoisLabels


class TrafficSignsEval(keras.callbacks.Callback):
    def __init__(self, generator, ground_truth_proto_file, train_proto_file,
                 resolution=2592,
                 max_number_of_images=100,
                 roi_min_side_size=25,
                 lowest_score_threshold=0.5):
        self.generator = generator
        self.ground_truth_proto_file = ground_truth_proto_file
        self.rois_labels = RoisLabels(train_proto_file)
        self.resolution = resolution
        self.lowest_score_threshold = lowest_score_threshold
        self.images_folder = os.path.dirname(os.path.abspath(ground_truth_proto_file))
        self.max_number_of_images = max_number_of_images
        self.logger = logging.getLogger(__name__)
        self.expected_metadata = meta.read_metadata(self.ground_truth_proto_file)
        self.expected_dict = meta.create_metadata_dictionary(self.expected_metadata, True)
        self.roi_min_side_size = roi_min_side_size
        self.logger = logging.getLogger(__name__)
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.logger.info("Evaluating checkpoint:")
        self.__evaluate_traffic_signs()

    def __evaluate_traffic_signs(self):
        resolutions = [self.resolution]
        score_threshold_per_class = dict([(class_label, self.lowest_score_threshold) for class_label in self.rois_labels.classes.keys()])
        actual_metadata = predict_folder(self.model, self.images_folder, None, resolutions, self.rois_labels,
                                         score_threshold_per_class,
                                         draw_predictions=False, max_number_of_images=self.max_number_of_images,
                                         log_level=0)
        actual_dict = meta.create_metadata_dictionary(actual_metadata, True)
        for file_name in list(self.expected_dict.keys()):
            if file_name not in actual_dict:
                del self.expected_dict[file_name]
        statistics_dict = get_model_statistics(self.expected_dict, actual_dict, None, self.roi_min_side_size)
        self.logger.info(statistics_dict['Total'])



