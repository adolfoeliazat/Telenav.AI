#!/usr/bin/env bash

INPUT_PATH="/Users/adrianpopovici/Downloads/QA_US_SIGNS/train_selected_original/rois.bin"

echo 'Parameters:'
echo 'INPUT_PATH = ' $INPUT_PATH

set -e
PYTHONPATH=../../:$PYTHONPATH
export PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=2.

python recover_old_metadata.py --input_path $INPUT_PATH