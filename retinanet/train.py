"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import argparse
import functools
import os
import sys
import warnings
import logging
import keras
import keras.preprocessing.image
from keras.utils import multi_gpu_model
import tensorflow as tf

from keras_retinanet import losses
from keras_retinanet import layers
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.anchors import make_shapes_callback, anchor_targets_bbox
from keras_retinanet.utils.model import freeze as freeze_model
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from retinanet.traffic_signs_generator import TrafficSignsGenerator
from retinanet.traffic_signs_eval import TrafficSignsEval


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, backbone, num_classes, weights, multi_gpu=0, freeze_backbone=False):
    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, backbone=backbone, nms=False, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)

        # append NMS for prediction only
        classification   = model.outputs[1]
        detections       = model.outputs[2]
        boxes            = keras.layers.Lambda(lambda x: x[:, :, :4])(detections)
        detections       = layers.NonMaximumSuppression(name='nms')([boxes, classification, detections])
        prediction_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[:2] + [detections])
    else:
        model            = model_with_weights(backbone_retinanet(num_classes, backbone=backbone, nms=True, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model   = model
        prediction_model = model

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    callbacks = []

    # save the prediction model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        io_utils.create_folder(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)
            ),
            verbose=1
        )
        checkpoint = RedirectModel(checkpoint, prediction_model)
        callbacks.append(checkpoint)

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'traffic_signs':
            # use prediction model for evaluation
            ground_truth_proto_file = os.path.join(args.val_path, 'rois.bin')
            evaluation = TrafficSignsEval(validation_generator, ground_truth_proto_file, os.path.join(args.train_path, 'rois.bin'))
            evaluation = RedirectModel(evaluation, prediction_model)
            callbacks.append(evaluation)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'auto',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    ))

    return callbacks


def create_generators(args):
    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(min_rotation=-0.122173,
                                                     max_rotation=0.122173)
    train_generator, validation_generator = None, None

    if args.dataset_type == 'traffic_signs':
        train_generator = TrafficSignsGenerator(
            args.train_path,
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            group_method='random',
            image_min_side=1080,
            image_max_side=2592
        )
        if args.val_path:
            validation_generator = TrafficSignsGenerator(
                args.val_path,
                None,
                batch_size=args.batch_size,
                group_method='random'
            )

    return train_generator, validation_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    :param parsed_args: parser.parse_args()
    :return: parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    if 'resnet' in parsed_args.backbone:
        from keras_retinanet.models.resnet import validate_backbone
    elif 'mobilenet' in parsed_args.backbone:
        from keras_retinanet.models.mobilenet import validate_backbone
    elif 'vgg' in parsed_args.backbone:
        from keras_retinanet.models.vgg import validate_backbone
    elif 'densenet' in parsed_args.backbone:
        from keras_retinanet.models.densenet import validate_backbone
    else:
        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(parsed_args.backbone))

    validate_backbone(parsed_args.backbone)

    return parsed_args


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    ts_parser = subparsers.add_parser('traffic_signs')
    ts_parser.add_argument('train_path', help='Path to folder containing files used for train. The rois.bin file should be there.')
    ts_parser.add_argument('val_path', help='Path to folder containing files used for validation (optional). The rois.bin file should be there.')

    def csv_list(string):
        return string.split(',')


    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',        help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',       help='Number of GPUs to use for parallel processing.', type=int, default=2)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',          help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',           help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',   help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--evaluate_score_threshold', help='Score thresholds to be used for all classes when evaluate.', default=0.5, type=float)

    return check_args(parser.parse_args(args))


def main(args=None):
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args)

    if 'resnet' in args.backbone:
        from keras_retinanet.models.resnet import resnet_retinanet as retinanet, custom_objects, download_imagenet
    elif 'mobilenet' in args.backbone:
        from keras_retinanet.models.mobilenet import mobilenet_retinanet as retinanet, custom_objects, download_imagenet
    elif 'vgg' in args.backbone:
        from keras_retinanet.models.vgg import vgg_retinanet as retinanet, custom_objects, download_imagenet
    elif 'densenet' in args.backbone:
        from keras_retinanet.models.densenet import densenet_retinanet as retinanet, custom_objects, download_imagenet
    else:
        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(args.backbone))

    # create the model
    if args.snapshot is not None:
        logger.info('Loading model, this may take a second...')
        model            = keras.models.load_model(args.snapshot, custom_objects=custom_objects)
        training_model   = model
        prediction_model = model
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = download_imagenet(args.backbone)

        logger.info('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=retinanet,
            backbone=args.backbone,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone
        )

    logger.info(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        compute_anchor_targets = functools.partial(anchor_targets_bbox, shapes_callback=make_shapes_callback(model))
        train_generator.compute_anchor_targets = compute_anchor_targets
        if validation_generator is not None:
            validation_generator.compute_anchor_targets = compute_anchor_targets

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=4
    )


if __name__ == '__main__':
    main()
