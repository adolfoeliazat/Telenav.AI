"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import cv2
from PIL import Image
import os
from os.path import splitext

IMG_EXTENSIONS = ['.jpg', '.jpeg']


def __get_image_meta(image_path):
    # PIL is fast for metadata
    return Image.open(image_path)


def get_size(image_path):
    image = __get_image_meta(image_path)
    return image.width, image.height


def get_aspect_ratio(image_path):
    image = __get_image_meta(image_path)
    return image.width / image.height


def get_bgr(image_path):
    if os.path.isfile(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR
        if image is None:
            raise Exception("Image {} is invalid.".format(image_path))
        else:
            return image
    else:
        raise Exception("Image {} is missing.".format(image_path))


def get_rgb(image_path):
    image_bgr = get_bgr(image_path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def file_is_image(file_name):
    return splitext(file_name)[1] in IMG_EXTENSIONS and not file_name.startswith('._')


def rotate_image(image, angle):
    height, width, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    result = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return result


def flip_image(img):
    return cv2.flip(img,1)