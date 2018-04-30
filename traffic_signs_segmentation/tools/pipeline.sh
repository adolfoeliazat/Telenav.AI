# You need to set this up first.

ROI_PATH=/home/flaviub/train_test/rois.bin
# output model dir
MODEL_OUT_PATH=/home/flaviub/caffe_model_out/
# path to caffe tools folder
CAFFE_TOOLS=/opt/caffe/build/tools/
#Cuda visible devices
CUDA_VISIBLE_DEVICES=0,1
# path to imagerecognition/python_modules/traffic_signs_segmentation
TRAFFIC_SIGNS_SEGMENTATION_FOLDER=./../
# path to image test folder
TEST_FOLDER=/home/flaviub/train_test

# Do not modify
CLEAN_DATASET=$MODEL_OUT_PATH/cleaned_input
DATASET_TO_AUGMENT=$MODEL_OUT_PATH/trainval/rois.bin
AUGMENTED_DATASET=$MODEL_OUT_PATH/trainval_augmented
CLASSIFICATION_PATH=$MODEL_OUT_PATH/classification_db
SEGMENTATION_PATH=$MODEL_OUT_PATH/segmentation_db
ML_FOLDER=./config/
CAFFE_NETS_PATH=$ML_FOLDER
FTP_WEIGHTS_PATH=/signs_segmentation_caffe_weights/
UTILS=./utils

CLASSIFICATION_TRAIN_PROTO=$CAFFE_NETS_PATH/classification_train_val.prototxt
CLASSIFICATION_SOLVER_PATH=$CAFFE_NETS_PATH/classification_solver.prototxt
CLASSIFICATION_BATCH_SIZE=64
CLASSIFICATION_TRAIN_EPOCHS=100
CLASSIFICATION_SNAPSHOT_INTERVAL=5
CLASSIFICATION_RESIZE_WIDTH=64
CLASSIFICATION_RESIZE_HEIGHT=64

SEGMENTATION_TRAIN_PROTO=$CAFFE_NETS_PATH/segmentation_train_val.prototxt
SEGMENTATION_SOLVER_PATH=$CAFFE_NETS_PATH/segmentation_solver.prototxt
SEGMENTATATION_BATCH_SIZE=2
SEGMENTATION_TRAIN_EPOCHS=30
SEGMENTATION_SNAPSHOT_INTERVAL=5

cd $TRAFFIC_SIGNS_SEGMENTATION_FOLDER

PYTHONPATH="${PYTHONPATH}../:../apollo_python_common/:../apollo_python_common/protobuf/:./utils/:./"
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

#Train segmentation model
python3 $UTILS/edit_caffe_classification_train_proto.py --original_proto $CLASSIFICATION_TRAIN_PROTO --database_path $CLASSIFICATION_PATH --out_path $MODEL_OUT_PATH
python3 $UTILS/edit_caffe_solver.py -o $CLASSIFICATION_SOLVER_PATH -t $CLASSIFICATION_PATH/trainval/train_lmdb/ -v $CLASSIFICATION_PATH/test/test_lmdb/ -s $MODEL_OUT_PATH/classification \
        -b $CLASSIFICATION_BATCH_SIZE -e $CLASSIFICATION_TRAIN_EPOCHS -i $CLASSIFICATION_SNAPSHOT_INTERVAL --trainval_proto $CLASSIFICATION_TRAIN_PROTO

python3 $UTILS/edit_caffe_segmentation_train_proto.py -o $SEGMENTATION_TRAIN_PROTO -d $SEGMENTATION_PATH -b $SEGMENTATATION_BATCH_SIZE -t $SEGMENTATION_TRAIN_PROTO
python3 $UTILS/edit_caffe_solver.py -o $SEGMENTATION_SOLVER_PATH -t $SEGMENTATION_PATH/trainval/images/train_features_lmdb/ -v $SEGMENTATION_PATH/test/images/test_features_lmdb/ -s $MODEL_OUT_PATH/segmentation \
    -b $SEGMENTATATION_BATCH_SIZE -e $SEGMENTATION_TRAIN_EPOCHS -i $SEGMENTATION_SNAPSHOT_INTERVAL --trainval_proto $SEGMENTATION_TRAIN_PROTO

#Train segmentation model
$CAFFE_TOOLS/caffe train \
    -gpu $CUDA_VISIBLE_DEVICES \
    -solver $SEGMENTATION_SOLVER_PATH \
    -weights $CAFFE_NETS_PATH/segmentation_initial_weights.caffemodel 2>&1 | tee -a $MODEL_OUT_PATH/segmentation_model.log

#Train classification model
$CAFFE_TOOLS/caffe train \
     -gpu $CUDA_VISIBLE_DEVICES \
     -solver $CLASSIFICATION_SOLVER_PATH 2>&1 | tee -a $MODEL_OUT_PATH/classification_model.log

#Inference and Test
python3 $UTILS/parse_caffe_model_log.py --input_path $MODEL_OUT_PATH/segmentation_model.log --output_path segmentation_model.caffemodel
python3 $UTILS/parse_caffe_model_log.py --input_path $MODEL_OUT_PATH/classification_model.log --output_path classificationmodel.caffemodel
python3 $UTILS/convert_mean_image.py --input_blob $MODEL_OUT_PATH/mean.blob --output_npy $MODEL_OUT_PATH/mean.npy
cp $MODEL_OUT_PATH/segmentation_model.caffemodel $ML_FOLDER/
cp $MODEL_OUT_PATH/classificationmodel.caffemodel $ML_FOLDER/
cp $MODEL_OUT_PATH/mean.npy $ML_FOLDER/

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 ./inference_server.py &
sleep 10 && python3 ./inference_client.py --input_path $TEST_FOLDER --output_path $MODEL_OUT_PATH
kill $!
python3 -u ../apollo_python_common/generate_model_statistics.py --expected_rois_file $TEST_FOLDER/rois.bin --actual_rois_file $MODEL_OUT_PATH/rois.bin --result_file $MODEL_OUT_PATH/results.txt