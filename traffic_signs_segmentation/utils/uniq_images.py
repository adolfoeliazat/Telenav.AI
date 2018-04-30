"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import argparse

import os
import shutil
import sys

import pypuzzle

import utils

def collect_vectors(images):
    puzzle = pypuzzle.Puzzle()
    return [puzzle.get_cvec_from_file(img) for img in images]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dataset_path", type=str, required=True)
    parser.add_argument("-t", "--threshold", type=float, required=False, default=0.2)
    args = parser.parse_args()

    features_path = args.dataset_path + "/images/"
    labels_path = args.dataset_path + "/labels/"
    duplicates_path = args.dataset_path + "/duplicate/"

    if not utils.valid_dataset(args.dataset_path):
        print "Invalid dataset"
        sys.exit(-1)

    utils.make_dirs([duplicates_path])

    images = utils.collect_images(features_path)
    vectors = collect_vectors(images)

    duplicate_log = open(duplicates_path + "/duplicates.log", "w")
    puzzle = pypuzzle.Puzzle()
    for i in range(len(vectors) - 1):
        for j in range(i + 1, len(vectors)):
            if not utils.exists_paths([images[i], images[j]]):
                continue

            threshold = abs(puzzle.get_distance_from_cvec(vectors[i], vectors[j]) -
                                args.threshold)
            if threshold <= 0.01:
                duplicate_img = duplicates_path + os.path.basename(images[j])
                shutil.move(images[j], duplicate_img)
                shutil.move(labels_path + os.path.basename(images[j]), args.dataset_path +
                            "/duplicate/label_" + os.path.basename(images[j]))
                duplicate_log.write("Duplicate " + str(images[i]) + " " + str(images[j]) +
                                    " threshold " + str(threshold) + '\n')

if __name__ == "__main__":
     main()
