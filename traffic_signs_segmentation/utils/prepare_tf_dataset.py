"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import argparse
import collections
import os
import io
import random
import hashlib

import PIL.Image as Image
import tensorflow as tf

import configuration
import utils

from object_detection.utils import dataset_util

DetObject = collections.namedtuple(
    "DetObject", ["name", "xmin", "ymin", "xmax", "ymax"])

conf = configuration.load_configuration()


def parse_voc(line):
    elems = [e.strip() for e in line.split()]
    return DetObject(
        name=elems[0],
        xmin=int(elems[4]),
        ymin=int(elems[5]),
        xmax=int(elems[6]),
        ymax=int(elems[7]))


def get_image(img_path):
    if os.path.exists(img_path + ".jpg"):
        return img_path + ".jpg"
    return img_path + ".jpeg"


def make_tf_example(annotation_file, detections, dataset_path):
    img_path = os.path.join(
        dataset_path, "images", utils.get_filename(annotation_file))
    img_path = get_image(img_path)
    with tf.gfile.GFile(img_path, "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    img = Image.open(encoded_jpg_io)
    if img.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = img.width
    height = img.height

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for d in detections:
        xmin.append(float(d.xmin) / width)
        ymin.append(float(d.ymin) / height)
        xmax.append(float(d.xmax) / width)
        ymax.append(float(d.ymax) / height)
        classes_text.append(d.name.encode('utf8'))
        classes.append(conf.name2class_id[d.name])
        truncated.append(0)
        poses.append("Unspecified".encode('utf8'))
        difficult_obj.append(0)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            os.path.basename(img_path).encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            os.path.basename(img_path).encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(
            classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(
            difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def generate_tf_record(files, input_path, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    for annotation_file in files:
        with open(annotation_file, "r") as a_file:
            contents = a_file.readlines()
        objects = [parse_voc(line) for line in contents]
        tf_example = make_tf_example(annotation_file, objects, input_path)
        writer.write(tf_example.SerializeToString())

    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_path",
                        type=str, required=True)
    parser.add_argument("-t", "--train", action="store_true")
    args = parser.parse_args()

    annotations_dir = os.path.join(args.input_path, "annotations")
    annotation_files = utils.collect_files(annotations_dir)
    if args.train:
        random.seed(1337)
        random.shuffle(annotation_files)
        train = annotation_files[
            0:int(len(annotation_files) * conf.split_ratio)]
        generate_tf_record(
            train,
            args.input_path,
            os.path.join(args.output_path, "train.record"))
        val   = annotation_files[
            int(len(annotation_files) * conf.split_ratio):]
        generate_tf_record(
            val,
            args.input_path,
            os.path.join(args.output_path, "val.record"))
    else:
        generate_tf_record(
            annotation_files,
            args.input_path,
            os.path.join(args.output_path, "test.record"))


if __name__ == "__main__":
    main()
