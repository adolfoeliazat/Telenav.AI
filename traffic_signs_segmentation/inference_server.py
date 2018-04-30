"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import argparse
import numpy as np
import os
import caffe
import grpc
import time
import sys

from concurrent import futures

import orbb_metadata_pb2, orbb_definitions_pb2
import apollo_python_common.proto_api as proto_api
import batch
import utils
import network_setup
import image as image_operation
import configuration
import logging
import logging.config
import apollo_python_common.log_util as log_util

# TODO: add logging
conf = configuration.load_configuration()

SEGMENTATION_PROTO_FILE = "./config/net.prototxt"
CLASSIFICATION_MODEL = "./config/classificationmodel.caffemodel"
CLASSIFICATION_PROTO_FILE = "./config/net_classification.prototxt"
MEAN = "./config/mean.npy"
SEGMENTATION_MODEL = "./config/segmmodel.caffemodel"
MEAN_BLOB = "./config/mean.blob"

# Available on ftp
required_files = [
    CLASSIFICATION_MODEL,
    CLASSIFICATION_PROTO_FILE,
    MEAN,
    SEGMENTATION_MODEL
    ]

class BatchHandler():
    """
    Handles a batch of images and produces a protobuf metadata with the detections
    """

    def __init__(self, detector, classifier):
        """
        :param detector: lambda function that detects traffic signs
        :param classifier: lambda function that classifies a traffic sign
        """
        self.metadata = orbb_metadata_pb2.ImageSet()
        self.metadata.name = "caffe"
        self.detector = detector
        self.classifier = classifier

    def convert_detections_to_rois(self, image_batch, net_output):
        """
        Converts the detections to protobuf rois
        :param image_batch: batch of images
        :param net_output: segmentation result
        :return:
        """
        for i, elem in enumerate(image_batch):
            # skip processing invalid images
            if elem.image_data is None:
                continue
            mask = np.argmax(net_output[i], axis=0)
            rects = image_operation.get_rects(mask)
            rois = [image_operation.get_classification_roi(elem.image_data, r) for r in rects]
            class_ids = list(map(self.classifier, rois))
            image = self.metadata.images.add()
            image.metadata.image_path = os.path.basename(elem.image_path)
            image.metadata.region = ""
            image.metadata.trip_id = ""
            image.metadata.image_index = 0
            add_detections(image, rects, class_ids)
            shape = elem.image_data.shape
            transform_detections(image.rois, elem.scale_factors, shape)

    def __call__(self, image_batch):
        """
        Handles a batch of images and generates detections
        :param image_batch:
        :return:
        """
        out = self.detector(image_batch)
        self.convert_detections_to_rois(image_batch, out)
        
    def get_result(self):
        """
        Returns protobuf metadata
        :return: protobuf metadata
        """
        return self.metadata


# TODO: classify all rois at once
def classify(net, transformer, roi):
    """
    Classification network forward pass
    :param net: Classification network
    :param transformer: Classification network transformer
    :param roi: Traffic sign region of interest
    :return: class id, confidence score
    """
    caffe.set_mode_gpu()
    net.blobs[net.inputs[0]].data[...] = transformer.preprocess(net.inputs[0], roi)
    results_array = net.forward()[net.outputs[0]][0]
    best_class_id = results_array.argmax()
    best_score = results_array[best_class_id]
    return [best_class_id, best_score]

def detect_batch(net, current_batch):
    """
    Segmentation network forward pass
    :param net: Segmentation network
    :param current_batch: image batch
    :return: segmentation output
    """
    caffe.set_mode_gpu()
    for i, elem in enumerate(current_batch):
        # skip processing invalid images
        if elem.image_data is not None:
            net.blobs[net.inputs[0]].data[i] = elem.image_data
    return net.forward()[net.outputs[0]]


def add_detections(image, rects, class_ids):
    """
    Appends roi detections to metadata
    :param image: roi metadata
    :param rects: np array with signs bounding boxes
    :param class_ids: signs class ids and confidences
    :return:
    """
    for i in range(len(rects)):
        threshold = conf.class_id2threshold[class_ids[i][0]]
        confidence = class_ids[i][1]
        if confidence < threshold or rects[i][2] < conf.min_size or rects[i][3] < conf.min_size:
            continue
        roi = image.rois.add()
        roi.algorithm = conf.algorithm
        roi.algorithm_version = conf.algorithm_version
        roi.manual = False
        roi.rect.tl.row = rects[i][1]
        roi.rect.tl.col = rects[i][0]
        roi.rect.br.row = rects[i][1] + rects[i][3]
        roi.rect.br.col = rects[i][0] + rects[i][2]
        roi.type = orbb_definitions_pb2.INVALID
        if class_ids[i][0] != conf.invalid_id:
            roi.type = conf.class_id2type[class_ids[i][0]]
            detection = roi.detections.add()
            detection.type = roi.type
            detection.confidence = class_ids[i][1]


def transform_detections(rois, transformer, shape):
    """
    Rescales the detections to the original image size
    :param rois: metadata rois
    :param transformer: scale information between original image and network shape
    :param shape: network input image shape
    :return:
    """

    _, height, width = shape
    height = height * transformer.scale_rows
    width = width * transformer.scale_cols + 2 * transformer.width_crop_size
    for roi in rois:
        roi.rect.tl.row = int(roi.rect.tl.row * transformer.scale_rows)
        roi.rect.tl.col = int(roi.rect.tl.col * transformer.scale_cols
                              + transformer.width_crop_size)

        roi.rect.br.row = int(min(roi.rect.br.row * transformer.scale_rows, height - 1))
        roi.rect.br.col = int(min(roi.rect.br.col * transformer.scale_cols
                                  + transformer.width_crop_size, width - 1))


class InferenceService:
    """
    Inference Service class
    """

    def __init__(self):
        self.net_segm, self.transformer_segm = network_setup.make_network(
            SEGMENTATION_PROTO_FILE,
            SEGMENTATION_MODEL,
            None)
        self.net_class, self.transformer_class = network_setup.make_network(
            CLASSIFICATION_PROTO_FILE,
            CLASSIFICATION_MODEL,
            MEAN)
        network_setup.convert_mean(MEAN_BLOB, MEAN)

    def process(self, images_path):
        """
        Server process method runs segmentation and classification and returns the rois metadata
        :param images_path: images path folder
        :return: rois metadata
        """
        logger = logging.getLogger(__name__)
        if not utils.exists_paths([images_path]):
            logger.info("Request path missing {}.".format(images_path))
            return None
        
        classifier = lambda x: classify(
            self.net_class, self.transformer_class, x)
        detector = lambda x: detect_batch(self.net_segm, x)

        batch_reader = batch.BatchReader(
            images_path,
            self.transformer_segm,
            network_setup.get_batch_size(self.net_segm))
        batch_handler = BatchHandler(detector, classifier)
        batch_processor = batch.BatchProcessor(batch_handler)
        batch_reader.start()
        batch_processor.start()
        batch_reader.join()
        batch_processor.join()
        return batch_handler.get_result()

if __name__ == "__main__":
    log_util.config(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, required=True)
    parser.add_argument(
        "-o", "--output_path", type=str, required=False, default="./")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    if not utils.exists_paths(required_files):
        logger.error("Paths missing {}".format(required_files))
        sys.exit(-1)

    server = InferenceService()
    response = server.process(args.input_path)
    proto_api.serialize_metadata(response, args.output_path)
