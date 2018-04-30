"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import caffe

import numpy as np

def get_batch_size(net):
    """
    Returns network batch size
    """
    return net.blobs[net.inputs[0]].data.shape[0]


def get_net_inputs(net):
    """
    Returns network input data
    """
    return net.blobs[net.inputs[0]]


def convert_mean(path_binary, output_file):
    """
    Converts blob mean image to npy
    """
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(path_binary, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    out = arr[0]
    np.save(output_file, out)


def get_new_shape(net):
    """
    Returns network new shape
    """
    return (get_batch_size(net),) + tuple(get_net_inputs(net).data.shape[1:])


def get_transformer(net):
    """
    Returns network transformer
    """
    transformer = caffe.io.Transformer({"data": get_new_shape(net)})
    transformer.set_transpose("data", (2, 0, 1))
    return transformer


def make_network(network, weights, mean):
    """
    Creates a network and the transformer
    :param network: path to deploy proto
    :param weights: path to network weights
    :param mean: path to image mean
    """
    net = caffe.Net(network, weights, caffe.TEST)
    get_net_inputs(net).reshape(*get_new_shape(net))
    transformer = get_transformer(net)
    if mean:
        transformer.set_mean("data", np.load(mean).mean(1).mean(1))
    return net, transformer

