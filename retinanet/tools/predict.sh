#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES="0"
WEIGHTS='/data/model/retinanet_resnet50_traffic_signs_v002.pb'
TRAIN_META_FILE='/data/train_data/rois.bin'
INPUT_PATH='/data/test_data'
THRESHOLD_FILE="/data/model/classes_thresholds.json"
LOWEST_SCORE_THRESHOLD="0.5"
OUTPUT_PATH='/data/output'

echo 'Parameters:'
echo 'CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES
echo 'LOWEST_SCORE_THRESHOLD='$LOWEST_SCORE_THRESHOLD
echo 'WEIGHTS='$WEIGHTS
echo 'TRAIN_META_FILE='$TRAIN_META_FILE
echo 'INPUT_PATH='$INPUT_PATH
echo 'OUTPUT_PATH='$OUTPUT_PATH
echo '----------------------------------------'
PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u ../predict.py \
    --weights_file $WEIGHTS \
    --input_images_path $INPUT_PATH \
    --output_images_path $OUTPUT_PATH \
    --train_meta_file $TRAIN_META_FILE \
    --lowest_score_threshold $LOWEST_SCORE_THRESHOLD \
    --threshold_file $THRESHOLD_FILE
