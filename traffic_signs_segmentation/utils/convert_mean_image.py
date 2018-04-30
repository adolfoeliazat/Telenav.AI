"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Converts .blob image into .npy image
"""

import argparse
import network_setup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_blob",
                        type=str, required=True)
    parser.add_argument("-o", "--output_npy",
                        type=str, required=True)
    args = parser.parse_args()
    network_setup.convert_mean(args.input_blob, args.output_npy)


if __name__ == "__main__":
    main()
