"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import os
from collections import defaultdict

import orbb_metadata_pb2
from orbb_definitions_pb2 import _MARK as ROI_MARK
import apollo_python_common.io_utils as io_utils


def get_new_metadata_file(imageset_name = "imageset"):
    metadata = orbb_metadata_pb2.ImageSet()
    metadata.name = imageset_name
    return metadata


def get_new_image_proto(trip_id, image_index, image_path, region, serialized=False):
    image = orbb_metadata_pb2.Image()
    image.metadata.trip_id = trip_id
    image.metadata.image_index = image_index
    image.metadata.image_path = image_path
    image.metadata.region = region
    if serialized:
        return image.SerializeToString()
    else:
        return image


def read_metadata(file_name):
    metadata = orbb_metadata_pb2.ImageSet()
    with open(file_name, 'rb') as f:
        metadata.ParseFromString(f.read())
    return metadata


def read_image_proto(serialized_image_proto):
    image = orbb_metadata_pb2.Image()
    image.ParseFromString(serialized_image_proto)
    return image


def merge_metadata_dictionaries(dict1, dict2):
    for filename, rois in dict2.items():
        if len(rois) > 0:
            dict1[filename].extend(rois)
    return dict1


def create_metadata_dictionary(metadata, check_validation=True):
    dictionary = defaultdict(list)
    for element in metadata.images:
        for roi in element.rois:
            if check_validation:
                if roi.validation == 0:
                    dictionary[str(element.metadata.image_path)].append(roi)
            else:
                dictionary[str(element.metadata.image_path)].append(roi)
    return dictionary


def get_class_names_from_metadata_dictionary(rois_dict):
    all_classes = set()
    for file_name, rois in rois_dict.items():
        for roi in rois:
            all_classes.add(get_roi_type_name(roi.type))
    return sorted(list(all_classes))


def create_metadata_from_dict(dict_meta):
    metadata = get_new_metadata_file()
    for filename, rois in dict_meta.items():
        append_new(metadata, filename, rois)
    return metadata


def check_metadata(rois_file):
    roi_metadata = read_metadata(rois_file)
    roi_dict = create_metadata_dictionary(roi_metadata)
    all_types_counts = dict()
    for file_name, rois in roi_dict.items():
        for roi in rois:
            roi_class = ROI_MARK.values_by_number[roi.type].name
            count = all_types_counts.get(roi_class, 0)
            all_types_counts[roi_class] = count+1
    return all_types_counts


def append_roi(image, current_roi):
    roi = image.rois.add()
    roi.type = current_roi.type
    roi.rect.CopyFrom(current_roi.rect)
    roi.manual = current_roi.manual
    roi.algorithm = current_roi.algorithm
    roi.algorithm_version = current_roi.algorithm_version
    for detection in current_roi.detections:
        new_detection = roi.detections.add()
        new_detection.type = detection.type
        new_detection.confidence = detection.confidence
    for component in current_roi.components:
        new_component = roi.components.add()
        new_component.type = component.type
        new_component.box.CopyFrom(component.box)
        new_component.value = component.value
    roi.validation = current_roi.validation


def append_new(output_metadata, filename, rois):
    image = output_metadata.images.add()
    image.metadata.image_path = os.path.basename(filename)
    image.metadata.trip_id = ""
    image.metadata.image_index = 0
    for current_roi in rois:
        append_roi(image, current_roi)


def append_existing(output_metadata, filename, rois):
    image_idx = [i for i, v in enumerate(output_metadata.images) if v.metadata.image_path == os.path.basename(filename)]
    for current_roi in rois:
        append_roi(output_metadata.images[image_idx[0]], current_roi)


def serialize_metadata(metadata, output_path, file_name="rois"):
    io_utils.create_folder(output_path)
    with open(os.path.join(output_path, "{}.bin".format(file_name)), "wb") as f:
        f.write(metadata.SerializeToString())

    with open(os.path.join(output_path, "{}.txt".format(file_name)), "w") as f:
        f.write(str(metadata))


def add_metadata(metadata, input_path):
    with open(input_path, "rb") as rois_file:
        metadata.ParseFromString(rois_file.read())


def get_roi_type_name(type_value):
    return ROI_MARK.values_by_number[type_value].name


def get_roi_type_value(type_name):
    return ROI_MARK.values_by_name[type_name].number


def copy_rois(image, filename,  metadata_list):
    output_image = orbb_metadata_pb2.Image()
    output_image.metadata.image_path = filename
    for roi in image.rois:
        output_roi = output_image.rois.add()
        output_roi.CopyFrom(roi)
    metadata_list.append(output_image)


def get_filtered_meta_dict(rois_file, selected_classes):
    roi_dict = create_metadata_dictionary(read_metadata(rois_file))
    result_dict = defaultdict(list)
    for file_base_name, rois in roi_dict.items():
        remaining_rois = list()
        for roi in rois:
            if get_roi_type_name(roi.type) in selected_classes:
                remaining_rois.append(roi)
        if len(remaining_rois) > 0:
            result_dict[file_base_name] = remaining_rois
    return result_dict

