"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import errno
import json
import os
import pickle
import shutil
import uuid
from os.path import isfile, join

from apollo_python_common.image import file_is_image
from apollo_python_common.lightweight_types import AttributeDict


def create_folder(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def copy_images(images_paths, output_path):
    for image in images_paths:
        shutil.copy(image, output_path)


def get_random_file_name():
    return str(uuid.uuid4().hex)


def str2bool(v):
    if v.lower() in ("true","1"):
        return True

    if v.lower() in ("false","0"):
        return False

    raise ValueError("Value must be True/1 or False/0")


def json_load(json_path):
    with open(json_path) as json_data_file:
        data = json.load(json_data_file)
    return AttributeDict(data)


def json_dump(obj, file_name):
    create_folder(os.path.dirname(file_name))
    with open(file_name, 'w') as outfile:
        json.dump(obj, outfile, sort_keys=True, indent=4, ensure_ascii = False)


def pickle_dump(obj, file_name):
    create_folder(os.path.dirname(file_name))
    with open(file_name, "wb") as file_h:
        pickle.dump(obj, file_h)


def pickle_load(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)


def get_images_from_folder(input_folder):
    return [join(input_folder, f) for f in os.listdir(input_folder) if isfile(join(input_folder, f)) and file_is_image(f)]


def get_missing_paths(paths):
    return [path for path in paths if not os.path.exists(path)]


def require_paths(paths):
    for path in get_missing_paths(paths):
        raise IOError('Path {} does not exist.'.format(path))


