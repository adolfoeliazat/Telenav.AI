"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Edits classification train proto file with the path to train/test databases and mean image
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
    parser.add_argument("-p", "--out_path",
                        type=str, required=True)
    args = parser.parse_args()

    net = caffe.proto.caffe_pb2.NetParameter()
    Merge((open(args.original_proto,'r+').read()), net)
    for layer in  net.layer:
        if 'Data' in layer.type:
            if layer.name == "train-data":
                layer.data_param.source = args.database_path + "/trainval/train_lmdb"
            elif layer.name == "val-data":
                layer.data_param.source = args.database_path + "/test/test_lmdb"
            layer.transform_param.mean_file = args.out_path + "mean.blob"
    with open(args.original_proto, "w") as f:
        f.write(str(net))

if __name__ == "__main__":
    main()
