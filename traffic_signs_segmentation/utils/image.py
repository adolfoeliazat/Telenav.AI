"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import numpy as np
import cv2
import scipy.misc

from PIL import Image

import configuration

conf = configuration.load_configuration()

# TODO: remove hardcoding
MASK_VAL = 30


def valid_rois(rois):
    """
    Selects the rois that have a minimum side size
    """
    return [r for r in rois if r[2] >= conf.min_size and r[3] >= conf.min_size]


def get_rects(mask):
    """
    Returns the contour rectangles from the masked image
    """
    # TODO: in theory we don't need an extra copy,
    # but the type of the array is not what opencv expects
    mask_blend = mask[..., np.newaxis].astype(np.uint8)
    mask_blend[mask_blend != 0] = MASK_VAL

    _, thresh = cv2.threshold(mask_blend, MASK_VAL - 1, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours]

    return rects


def get_roi(image, rect):
    """
    Crops the rectangle from the original image
    """
    return image[:, rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]


def get_classification_roi(image, rect):
    """
    Returns the cropped and scaled sign to the dimension needed by classification
    """
    im = get_roi(image, rect).copy().transpose(1, 2, 0).astype(np.uint8)
    resized_image_filled = resize_image_fill(im, 82, 82)
    return resized_image_filled


# PIL does lazy loading and you can get the size from the image header
def get_resolution(image_path):
    """
    Returns the image resolution
    """
    return Image.open(image_path).size


def load_image(image_path):
    """
    Reads and returns an image
    """
    return cv2.imread(image_path)


def save_image(image_data, image_path):
    """
    Saves image
    """
    cv2.imwrite(image_path, image_data)


def get_new_width(image, net_input=conf.image_size):
    """
    Returns new image width after scaled and cropped to keep the aspect ratio
    of the net input
    """
    image_ratio = image.width / (1.0 * image.height)
    new_width = int(image_ratio * net_input[1])
    width_crop_size = (new_width - net_input[0]) / 2
    return new_width, width_crop_size


def get_crop_scale(width, height):
    """
    Returns new image scale ration and crop size to keep the aspect ratio
    of the configuration image size
    """
    scaling_factor = height / (1.0 * conf.image_size[1])
    width_crop_size = int((width / scaling_factor - conf.image_size[0]) / 2.0)
    return scaling_factor, width_crop_size

def resize_image(image, net_input=conf.image_size):
    """
    Returns cropped and scaled image to keep aspect ration
    """
    new_width, width_crop_size = get_new_width(image, net_input)
    resized_image = image.resize((new_width, net_input[1]))

    if width_crop_size == 0:
        return resized_image, width_crop_size

    cropped_image = resized_image.crop((
        width_crop_size,
        0,
        new_width - width_crop_size,
        net_input[1]))
    return cropped_image, width_crop_size


def resize_image_fill(image, height, width):
    """
    Resize image by filling with random noise
    """
    if image.shape[0] == height and image.shape[1] == width:
        return image
    channels = 3
    width_ratio = float(image.shape[1]) / width
    height_ratio = float(image.shape[0]) / height
    if width_ratio > height_ratio:
        resize_width = width
        resize_height = int(round(image.shape[0] / width_ratio))
        if (height - resize_height) % 2 == 1:
            resize_height += 1
    else:
        resize_height = height
        resize_width = int(round(image.shape[1] / height_ratio))
        if (width - resize_width) % 2 == 1:
            resize_width += 1
    image = scipy.misc.imresize(image, (resize_height, resize_width), 'bilinear')

    if width_ratio > height_ratio:
        padding = (height - resize_height) // 2
        noise_size = (padding, width)
        noise_size += (channels,)
        noise = np.random.randint(0, 255, noise_size).astype('uint8')
        image = np.concatenate((noise, image, noise), axis=0)
    else:
        padding = (width - resize_width) // 2
        noise_size = (height, padding)
        if channels > 1:
            noise_size += (channels,)
        noise = np.random.randint(0, 255, noise_size).astype('uint8')
        image = np.concatenate((noise, image, noise), axis=1)
    return image


def crop_image_to_ratio(image, ratio):
    """
    Returns cropped image that keeps the desired aspect ratio
    """
    height, width, _ = image.shape
    new_width = int(ratio * height)
    width_crop_size = int((width - new_width) / 2)

    if width_crop_size == 0:
        return image, width_crop_size

    return image[:, width_crop_size:
                 width - width_crop_size, :], width_crop_size


def valid_ratio(ratio, epsilon, image_path):
    """
    Whether or not the aspect ratio is valid
    """
    width, height = get_resolution(image_path)
    return valid_image(width, height, ratio, epsilon)


def valid_image(width, height, ratio, epsilon):
    """
    Invalidate images where the aspect ratio is over the epsilon threshold or image is in portrait
    """
    if height > width:
        return False

    img_ratio = width / (1.0 * height)
    return (img_ratio - ratio) <= epsilon


def rotate_image(image, angle):
    """
    Rotates image with given angle
    """
    height, width, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    result = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return result, rotation_matrix


def adjust_gamma(image, gamma=1.0):
    """
    Adjust image gamma
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
