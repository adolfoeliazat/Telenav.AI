"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Edits segmentation train proto file with the path to train/test databases
"""

import argparse
import caffe
from google.protobuf.text_format import Merge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--original_proto",
                        type=str, required=True)
    parser.add_argument("-d", "--database_path",
                        type=str, required=True)
    parser.add_argument("-b", "--batch_size",
                        type=int, required=True)
    parser.add_argument("-t", "--training_proto",
                        type=str, required=True)
    args = parser.parse_args()

    net = caffe.proto.caffe_pb2.NetParameter()
    Merge((open(args.original_proto,'r+').read()), net)
    net.layer[0].data_param.source = args.database_path + "/trainval/images/train_features_lmdb"
    net.layer[0].data_param.batch_size = args.batch_size
    net.layer[1].data_param.source = args.database_path + "/trainval/labels/train_labels_lmdb"
    net.layer[1].data_param.batch_size = args.batch_size
    net.layer[2].data_param.source = args.database_path + "/test/images/test_features_lmdb"
    net.layer[2].data_param.batch_size = args.batch_size
    net.layer[3].data_param.source = args.database_path + "/test/labels/test_labels_lmdb"
    net.layer[3].data_param.batch_size = args.batch_size
    with open(args.original_proto, "w") as f:
        f.write(str(net))


if __name__ == "__main__":
    main()