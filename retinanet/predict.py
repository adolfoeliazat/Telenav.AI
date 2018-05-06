"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import argparse
import logging
import os
import sys

import keras
import keras.preprocessing.image
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.keras_version import check_keras_version

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as meta
from retinanet.traffic_signs_generator import RoisLabels
from retinanet.utils import predict_folder, get_graph_name

RESOLUTIONS = [800, 1920]


def __parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--weights_file',
                        help='Weights to use for initialization (defaults to resnet50_traffic_signs_01.h5).',
                        default='snapshots/resnet50_traffic_signs_01.h5')
    parser.add_argument('--input_images_path',
                        help='Path to input dataset directory.',
                        default='./input')
    parser.add_argument('--output_images_path',
                        help='Path to dataset directory where images with predictions will be generated.',
                        default='./output')
    parser.add_argument('--train_meta_file',
                        help='Where the metadata file used for training is located (e.g. ./rois.bin).',
                        default='./input')
    parser.add_argument('--multi-gpu',     help='Number of GPUs to use for parallel processing.',
                        type=int, default=1)
    parser.add_argument('--lowest_score_threshold',
                        help='All predictions having the score below this value will be discarded.',
                        type=float, default=0.1)
    parser.add_argument('--threshold_file',
                        help='Threshold file to be used for minimum classes confidence.',
                        type=str, default='SAME')
    return parser.parse_args(args)


def __create_keras_model(gpu_options, multi_gpu, weights_file):
    # make sure Keras is the minimum required version
    check_keras_version()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.tensorflow_backend.set_session(sess)
    model = keras.models.load_model(weights_file, custom_objects=custom_objects)
    if multi_gpu > 1:
        predict_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        predict_model = model
    return predict_model


def __create_tensorflow_model(gpu_options, weights_file):
    with tf.Graph().as_default() as graph:
        with tf.gfile.GFile(weights_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, input_map=None, return_elements=None, name=get_graph_name())
        predict_model = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options))
    return predict_model


def get_model_for_pred(config):
    logger = logging.getLogger(__name__)
    weights_file, multi_gpu = config.weights_file, config.multi_gpu
    weights_file_extension = os.path.splitext(weights_file)[1]
    if "per_process_gpu_memory_fraction" in config and "allow_growth_gpu_memory" in config:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.per_process_gpu_memory_fraction,
                                    allow_growth=config.allow_growth_gpu_memory)
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.67)

    logger.info('Loading model from {}'.format(weights_file))
    if weights_file_extension.lower() == ".h5":
        predict_model = __create_keras_model(gpu_options, multi_gpu, weights_file)
    elif weights_file_extension.lower() == ".pb":
        predict_model = __create_tensorflow_model(gpu_options, weights_file)
    else:
        raise Exception("{} format is not valid as model weights".format(weights_file_extension))
    logger.info('The model was loaded.')
    return predict_model


def predict_one_folder(input_images_path, output_images_path, rois_labels, model,
                       score_threshold_per_class, out_rois_file_name='rois_retinanet'):
    metadata = predict_folder(model, input_images_path, output_images_path,
                              RESOLUTIONS, rois_labels, score_threshold_per_class,
                              draw_predictions=True, log_level=0, max_number_of_images=None)
    meta.serialize_metadata(metadata, output_images_path, out_rois_file_name)


def main():
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # parse arguments
    args = sys.argv[1:]
    args = __parse_args(args)

    rois_labels = RoisLabels(args.train_meta_file)
    if args.threshold_file=='SAME':
        score_threshold_per_class = dict([(class_label, args.lowest_score_threshold) for class_label in rois_labels.classes.keys()])
    else:
        score_threshold_per_class = io_utils.json_load(args.threshold_file)

    logger.info("Score thresholds: {}".format(score_threshold_per_class))
    model = get_model_for_pred(args)
    predict_one_folder(args.input_images_path, args.output_images_path, rois_labels, model,
                       score_threshold_per_class)


if __name__ == '__main__':
    main()
