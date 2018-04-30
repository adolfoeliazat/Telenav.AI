"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import random

import numpy as np
import caffe

class Dice(caffe.Layer):
    """
    A layer that calculates the Dice coefficient
    """
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute Dice coefficient.")
        # compute sum over all axes but the batch and channel axes
        self.sum_axes = tuple(range(1, bottom[0].data.ndim - 1))

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != 2*bottom[1].count:
            raise Exception("Prediction must have twice the number of elements of the input.")
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        label = bottom[1].data[:,0,:,:]
        # compute prediction
        prediction = np.argmax(bottom[0].data, axis=1)
        # area of predicted contour
        a_p = np.sum(prediction, axis=self.sum_axes)
        # area of contour in label
        a_l = np.sum(label, axis=self.sum_axes)
        # area of intersection
        a_pl = np.sum(prediction * label, axis=self.sum_axes)
        # dice coefficient
        dice_coeff = np.mean(2.*a_pl/(a_p + a_l))
        top[0].data[...] = dice_coeff

    def backward(self, top, propagate_down, bottom):
        pass
