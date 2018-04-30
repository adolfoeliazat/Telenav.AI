"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Creates a LMDB database in the format needed by caffe to store data
"""

import caffe
import lmdb
from PIL import Image
import numpy as np
import argparse


def get_arguments():
    # Import arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output LMDB folder')
    parser.add_argument('--input_images', type=str, required=True,
                        help='Path to .txt file with all images to store in \
                        the LMDB database')
    parser.add_argument('--input_images_path', type=str, required=True,
                        help='Path to folder with all images')
    parser.add_argument('--nb_classes', type=int, default=255, required=True,
                        help='Number of classes. Only required for labels')
    parser.add_argument('--labels', action="store_true",
                        help='Precise if dealing with labels and not RGB')
    return parser.parse_args()

def main():
    # Get all options
    args = get_arguments()
    MAX_DB_SIZE = 1000000000000
    in_db = lmdb.open(args.output_path, map_size=MAX_DB_SIZE)
    with in_db.begin(write=True) as in_txn:
        input_file_list = open( args.input_images )
        sorted_files = sorted(input_file_list.readlines())
        for in_idx, image_file in enumerate(sorted_files):
            print ('Loading image ', str(in_idx), ' : ', image_file.rstrip())
            image_file_path = args.input_images_path + image_file.rstrip()
            # load image:
            im = np.array(Image.open(image_file_path))
            if args.labels:
                # Convert from HxWx3 to HxWx1
                if len(im.shape) == 3:
                    im = im[:, :, 0]
                im = np.expand_dims(im, axis = 2)
            
            # Convert to CxHxW
            im = im.transpose((2, 0, 1))

            # Create the dataset
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx).encode(), im_dat.SerializeToString())
    in_db.close()

if __name__ == '__main__':
    main()
