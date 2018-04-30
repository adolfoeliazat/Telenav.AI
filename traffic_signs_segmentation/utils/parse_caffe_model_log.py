"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Selects the best model from the log file.
The best model is considered the one with the smallest test loss
"""

import argparse
import operator
import shutil
from pathlib import Path

SNAPSHOT_SEARCH_STRING = "Snapshotting to binary proto file "
LOSS_LINE_OFFSET = 5 


def get_model_loss_pair(snapshot_line, loss_line):
    """
    Creates pairs of snapshot iteration and loss value
    """
    snapshot_path = snapshot_line[len(SNAPSHOT_SEARCH_STRING) + snapshot_line.rfind(SNAPSHOT_SEARCH_STRING):-1]
    loss_val_index = loss_line.rfind("loss = ")
    loss_val = loss_line[loss_val_index + len("loss = "):].split()[0]
    return {loss_val:snapshot_path}


def select_best_model(log_file):
    """
    Selects the best model from the log file.
    The best model is considered the one with the smallest test loss
    """
    log_lines = log_file.readlines()
    snapshot_loss_dict = dict()
    for idx, line in enumerate(log_lines):
        if SNAPSHOT_SEARCH_STRING in line:
            loss_line = log_lines[idx + LOSS_LINE_OFFSET]
            snapshot_loss_dict.update(get_model_loss_pair(line, loss_line))
    sorted_dict = sorted(snapshot_loss_dict.items(), key=operator.itemgetter(0))
    return sorted_dict[0][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        type=str, required=True)
    parser.add_argument("-o", "--output_name",
                        type=str, required=True)

    args = parser.parse_args()
    with open(args.input_path, "r") as log_file:
        best_model_path_str = select_best_model(log_file)
        shutil.copy(str(Path(best_model_path_str)), str(Path(best_model_path_str).parent) + "/" + args.output_name)


if __name__ == "__main__":
    main()
