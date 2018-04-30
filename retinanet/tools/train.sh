#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES="0"
WEIGHTS='./snapshots/traffic_signs.h5'
TRAIN_PATH='/data/datasets/train'
VALIDATION_PATH='/data/datasets/validation'


echo 'Parameters:'
echo 'TRAIN_PATH:' $TRAIN_PATH
echo 'VALIDATION_PATH:' $VALIDATION_PATH
echo 'WEIGHTS:' $WEIGHTS
echo '------------------------------------'
PYTHONPATH=../../:../../apollo_python_common/protobuf/:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -u ../train.py \
    --weights $WEIGHTS --steps 45000 --multi-gpu 1 --batch-size 1 --evaluate_score_threshold 0.5 \
    traffic_signs $TRAIN_PATH $VALIDATION_PATH
