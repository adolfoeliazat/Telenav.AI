"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Edits solver proto file by calculating the max_iter, test_interval and snapshot based on the number of images
in the database, batch size and number of epochs
"""

import argparse
import caffe
import lmdb
import math
from google.protobuf.text_format import Merge

def count_entries(lmdb_path):
    train_lmdb_env = lmdb.open(lmdb_path)
    with train_lmdb_env.begin() as txn:
        entries = txn.stat()['entries']
        return entries

def get_solver(solver_path):
    solver = caffe.proto.caffe_pb2.SolverParameter()
    with open(solver_path,'r+') as f:
        Merge(str(f.read()), solver)
    return solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--original_proto",
                        type=str, required=True)
    parser.add_argument("-t", "--train_database_path",
                        type=str, required=True)
    parser.add_argument("-v", "--test_database_path",
                        type=str, required=True)
    parser.add_argument("-s", "--snapshot_output",
                        type=str, required=True)
    parser.add_argument("-b", "--batch_size",
                        type=int, required=True)
    parser.add_argument("-e", "--epochs",
                        type=int, required=True) 
    parser.add_argument("-i", "--snapshot_interval",
                        type=int, required=True)                    
    parser.add_argument("--trainval_proto",
                        type=str, required=True)
    args = parser.parse_args()

    train_lmdb_size = count_entries(args.train_database_path)
    test_lmdb_size = count_entries(args.test_database_path)
    snapshot_output = args.snapshot_output
    BATCH_MULTIPLIER = 7
    BATCH_ACCUMULATION = BATCH_MULTIPLIER * args.batch_size
    solver = get_solver(args.original_proto)
    del solver.test_iter[0]
    solver.test_iter.extend([int(math.ceil(float(test_lmdb_size / args.batch_size )))])
    train_iter = int(math.ceil(float(train_lmdb_size / (args.batch_size * BATCH_ACCUMULATION))))
    solver.max_iter = args.epochs * train_iter
    solver.snapshot = args.snapshot_interval * train_iter
    solver.test_interval = train_iter
    solver.snapshot_prefix = snapshot_output
    solver.net = args.trainval_proto
    
    with open(args.original_proto, "w") as f:
        f.write(str(solver))

if __name__ == "__main__":
    main()
