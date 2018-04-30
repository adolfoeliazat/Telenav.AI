"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import numpy as np
import logging
import os.path
from collections import defaultdict
from keras_retinanet.preprocessing.generator import Generator

import apollo_python_common.image
import apollo_python_common.proto_api as meta
import apollo_python_common.io_utils as io_utils


class RoisLabels:
    def __init__(self,
                 rois_file_name):
        self.rois_file_name=rois_file_name
        self.rois_dict = self.__get_rois_dict_from_file_name()
        self.classes = dict([(class_name, id) for id, class_name in enumerate(self.__get_classes_from_rois_dict())])
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __get_rois_dict_from_file_name(self):
        roi_metadata = meta.read_metadata(self.rois_file_name)
        rois_dict = meta.create_metadata_dictionary(roi_metadata)
        return rois_dict

    def __get_classes_from_rois_dict(self):
        return meta.get_class_names_from_metadata_dictionary(self.rois_dict)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]


class TrafficSignsGenerator(Generator):
    def __init__(
            self,
            base_dir,
            image_data_generator,
            **kwargs
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info('Initializing TrafficSigns dataset from'.format(base_dir))
        self.image_names = []
        self.image_data = {}
        self.base_dir = base_dir
        self.rois_labels = RoisLabels(os.path.join(self.base_dir, 'rois.bin'))
        self.image_names = [filename for filename in self.rois_dict().keys() if os.path.isfile(os.path.join(self.base_dir, filename))]
        self.image_data = self.get_image_data()
        self.logger.info("Classes: {}".format(self.labels()))
        super().__init__(image_data_generator, **kwargs)
        self.logger.info('Dataset was initialised.')

    def rois_dict(self):
        return self.rois_labels.rois_dict

    def classes(self):
        return self.rois_labels.classes

    def labels(self):
        return self.rois_labels.labels

    def get_image_data(self):
        result = defaultdict(list)
        for img_file in self.image_names:
            rois = self.rois_dict()[img_file]
            for roi in rois:
                result[img_file].append({'x1': roi.rect.tl.col, 'x2': roi.rect.br.col,
                                         'y1': roi.rect.tl.row, 'y2': roi.rect.br.row,
                                         'class': meta.get_roi_type_name(roi.type)})
        return result

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return self.rois_labels.num_classes()

    def name_to_label(self, name):
        return self.rois_labels.name_to_label(name)

    def label_to_name(self, label):
        return self.rois_labels.label_to_name(label)

    def image_path(self, image_index):
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        return apollo_python_common.image.get_aspect_ratio(self.image_path(image_index))

    def image_size(self, image_index):
        return apollo_python_common.image.get_size(self.image_path(image_index))

    def load_image(self, image_index):
        img = apollo_python_common.image.get_bgr(self.image_path(image_index))
        return img

    def load_annotations(self, image_index):
        path   = self.image_names[image_index]
        annots = self.image_data[path]
        boxes  = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            class_name = annot['class']
            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])
            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes

