#!/usr/bin/env bash

TEST_ROI='rois_test.bin'
PREDICT_ROI='rois_retinanet.bin'
RESULT_FILE='classes_thresholds.json'
echo 'Parameters:'
echo 'TEST_ROI:' $TEST_ROI
echo 'PREDICT_ROI:' $PREDICT_ROI
echo 'RESULT_FILE:' $RESULT_FILE

echo '------------------------------------'
PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES='-1' python -u ../generate_best_thresholds.py \
    -a $PREDICT_ROI \
    -e $TEST_ROI \
    -o $RESULT_FILE

