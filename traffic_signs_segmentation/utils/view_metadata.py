"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Visualize metadata. Sign's bounding box and type overlay
"""

import argparse
import os

import roi_metadata
import utils
import visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_path",
                        type=str, required=True)
    args = parser.parse_args()
    utils.make_dirs([args.output_path])

    metadata = roi_metadata.read_metadata(args.input_path)
    visualization.visualize_metadata(
        os.path.dirname(args.input_path), metadata, args.output_path)


if __name__ == "__main__":
    main()
