#!/usr/bin/env bash

TEST_ROI='/data/output/rois.bin'
PREDICT_ROI='/data/test_data/rois.bin'
RESULT_FILE='statistics.txt'

echo 'Parameters:'
echo 'TEST_ROI:' $TEST_ROI
echo 'PREDICT_ROI:' $PREDICT_ROI
echo 'RESULT_FILE:' $RESULT_FILE
echo '------------------------------------'

ML_FOLDER=./../

cd $ML_FOLDER

PYTHONPATH="${PYTHONPATH}:../:../apollo_python_common/:../apollo_python_common/protobuf/:./utils/:./"

python3 -u ../apollo_python_common/generate_model_statistics.py \
    --expected_rois_file $TEST_ROI \
    --actual_rois_file $PREDICT_ROI \
    --result_file $RESULT_FILE