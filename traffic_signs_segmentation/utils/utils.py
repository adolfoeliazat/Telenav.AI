"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import os
from os import listdir
from os.path import isfile, join

import shutil


def get_filename(image_path):
    """
    Returns file name from full path
    """
    return os.path.splitext(os.path.basename(image_path))[0]


def get_extension(image_path):
    """
    Returns file extension from full path
    """
    return os.path.splitext(os.path.basename(image_path))[1]


def is_image(image_path):
    """
    Check if file has a known image extension
    """
    valid_images = [".jpg", ".jpeg", ".png"]
    return get_extension(image_path).lower() in valid_images


def collect_images(images_path):
    """
    Collects image file names from a directory
    """
    images = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    return [os.path.normpath(images_path + "/" + img)
            for img in images if is_image(img)]


def collect_files(path):
    """
    Collects file names from a directory
    """
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]


def exists_paths(paths):
    """
    Check if path exists
    """
    return all(os.path.exists(p) for p in paths)


def make_dirs(dirs):
    """
    Makes directories recursive
    """
    [shutil.rmtree(d) for d in [d for d in dirs if os.path.exists(d)]]
    [os.makedirs(d) for d in dirs]


def valid_dataset(path):
    """
    Check if path to dataset is valid: has images and labels subdirectory
    """
    return exists_paths([path + "/images/", path + "/labels/"])


def to_png(image):
    """
    Replace image extension to 'png'
    """
    return os.path.splitext(os.path.basename(image))[0] + ".png"
