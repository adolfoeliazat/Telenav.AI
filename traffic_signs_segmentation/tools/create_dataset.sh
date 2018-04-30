#!/usr/bin/env bash
# You need to set this up first.
# path to the rois.bin file 
ROI_PATH=/data/rois.bin
# output model dir
MODEL_OUT_PATH=/data/caffe_model/
# path to caffe tools folder
CAFFE_TOOLS=/opt/caffe/build/tools/
# path to imagerecognition/python_modules/traffic_signs_segmentation
TRAFFIC_SIGNS_SEGMENTATION_FOLDER=./../

# Do not modify
CLEAN_DATASET=$MODEL_OUT_PATH/cleaned_input
DATASET_TO_AUGMENT=$MODEL_OUT_PATH/trainval/rois.bin
AUGMENTED_DATASET=$MODEL_OUT_PATH/trainval_augmented
CLASSIFICATION_PATH=$MODEL_OUT_PATH/classification_db
SEGMENTATION_PATH=$MODEL_OUT_PATH/segmentation_db
UTILS=./utils


CLASSIFICATION_RESIZE_WIDTH=64
CLASSIFICATION_RESIZE_HEIGHT=64

cd $TRAFFIC_SIGNS_SEGMENTATION_FOLDER

PYTHONPATH="${PYTHONPATH}:../:../apollo_python_common/:../apollo_python_common/protobuf/:./utils/:./"
export PYTHONPATH

#Copy images
mkdir -p $CLEAN_DATASET
cp -r `dirname $ROI_PATH`/. $CLEAN_DATASET

#Create dataset
python3 $UTILS/clean_dataset.py --input_path $ROI_PATH --output_path $CLEAN_DATASET
python3 $UTILS/split_dataset.py --input_path $CLEAN_DATASET/rois.bin --output_path $MODEL_OUT_PATH

python3 $UTILS/dataset_augmentation.py --input_path $DATASET_TO_AUGMENT --output_path $AUGMENTED_DATASET
python3 $UTILS/clean_dataset.py --input_path $AUGMENTED_DATASET/rois.bin --output_path $AUGMENTED_DATASET

mkdir -p $SEGMENTATION_PATH
python3 $UTILS/prepare_segmentation.py --input_path $AUGMENTED_DATASET/rois.bin --output_path $SEGMENTATION_PATH/trainval/
python3 $UTILS/prepare_segmentation.py --input_path $MODEL_OUT_PATH/test/rois.bin --output_path $SEGMENTATION_PATH/test/

python3 $UTILS/convert_lmdb.py --output_path $SEGMENTATION_PATH/trainval/images/train_features_lmdb/ \
    --input_images $SEGMENTATION_PATH/trainval/images/val.txt --input_images_path $SEGMENTATION_PATH/trainval/images/ --nb_classes 2

python3 $UTILS/convert_lmdb.py --output_path $SEGMENTATION_PATH/trainval/labels/train_labels_lmdb/ \
    --input_images $SEGMENTATION_PATH/trainval/labels/val.txt --input_images_path $SEGMENTATION_PATH/trainval/labels/ --nb_classes 2 --labels

python3 $UTILS/convert_lmdb.py --output_path $SEGMENTATION_PATH/test/images/test_features_lmdb/ \
    --input_images $SEGMENTATION_PATH/test/images/val.txt --input_images_path $SEGMENTATION_PATH/test/images/ --nb_classes 2

python3 $UTILS/convert_lmdb.py --output_path $SEGMENTATION_PATH/test/labels/test_labels_lmdb/ \
    --input_images $SEGMENTATION_PATH/test/labels/val.txt --input_images_path $SEGMENTATION_PATH/test/labels/ --nb_classes 2 --labels

mkdir -p $CLASSIFICATION_PATH
python3 $UTILS/prepare_classification.py --input_path $AUGMENTED_DATASET/rois.bin --output_path $CLASSIFICATION_PATH/trainval
python3 $UTILS/prepare_classification.py --input_path $MODEL_OUT_PATH/test/rois.bin --output_path $CLASSIFICATION_PATH/test

GLOG_logtostderr=1 $CAFFE_TOOLS/convert_imageset \
    --resize_height=$CLASSIFICATION_RESIZE_HEIGHT \
    --resize_width=$CLASSIFICATION_RESIZE_WIDTH \
    --shuffle \
    $CLASSIFICATION_PATH/trainval \
    $CLASSIFICATION_PATH/trainval/val.txt \
    $CLASSIFICATION_PATH/trainval/train_lmdb/

GLOG_logtostderr=1 $CAFFE_TOOLS/compute_image_mean \
    $CLASSIFICATION_PATH/trainval/train_lmdb/ \
    $MODEL_OUT_PATH/mean.blob

GLOG_logtostderr=1 $CAFFE_TOOLS/convert_imageset \
    --resize_height=$CLASSIFICATION_RESIZE_HEIGHT \
    --resize_width=$CLASSIFICATION_RESIZE_WIDTH \
    --shuffle \
    $CLASSIFICATION_PATH/test \
    $CLASSIFICATION_PATH/test/val.txt \
    $CLASSIFICATION_PATH/test/test_lmdb/

python3 $UTILS/convert_mean_image.py --input_blob $MODEL_OUT_PATH/mean.blob --output_npy $MODEL_OUT_PATH/mean.npy