"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import os
import argparse

import apollo_python_common.protobuf.orbb_metadata_pb2 as orbb_metadata_pb2
import apollo_python_common.protobuf.old_orbb_metadata_pb2 as old_orbb_metadata_pb2
import apollo_python_common.proto_api as proto_api


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_path", help="input old rois path", type=str, required=True)

    return parser.parse_args()


def read_old_metadata(file_name):
    metadata = old_orbb_metadata_pb2.Metadata()
    with open(file_name, 'rb') as f:
        metadata.ParseFromString(f.read())
    return metadata


def convert_old_metadata(old_metadata):
    new_metadata = orbb_metadata_pb2.ImageSet()
    new_metadata.name = "imageset"
    for element in old_metadata.image_rois:
        image = new_metadata.images.add()
        image.metadata.image_path = element.file
        image.metadata.trip_id = ""
        image.metadata.image_index = 0
        image.metadata.region = "US"
        for roi in element.rois:
            new_roi = image.rois.add()
            new_roi.type = roi.type
            new_roi.rect.CopyFrom(roi.rect)
            new_roi.manual = roi.manual
            new_roi.algorithm = roi.algorithm
            for detection in roi.detections:
                new_detection = new_roi.detections.add()
                new_detection.type = detection.type
                new_detection.confidence = detection.confidence
            for component in roi.components:
                new_component = new_roi.components.add()
                new_component.type = component.type
                new_component.box.CopyFrom(component.box)
                new_component.value = component.value
            new_roi.validation = roi.validation

    return new_metadata


if __name__ == "__main__":
    args = parse_arguments()

    old_metadata = read_old_metadata(args.input_path)
    converted_metadata = convert_old_metadata(old_metadata)

    dir_name = os.path.dirname(args.input_path)
    file_name = os.path.basename(args.input_path)
    new_file_name = 'converted_' + os.path.splitext(file_name)[0]

    proto_api.serialize_metadata(converted_metadata, dir_name, new_file_name)
