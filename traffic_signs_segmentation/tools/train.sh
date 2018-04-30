# You need to set this up first.
# output model dir
MODEL_OUT_PATH=/data/caffe_model/
# path to caffe tools folder
CAFFE_TOOLS=/opt/caffe/build/tools/
#Cuda visible devices
CUDA_VISIBLE_DEVICES=0
# path to imagerecognition/python_modules/traffic_signs_segmentation
TRAFFIC_SIGNS_SEGMENTATION_FOLDER=./../

# Do not modify
CLEAN_DATASET=$MODEL_OUT_PATH/cleaned_input
DATASET_TO_AUGMENT=$MODEL_OUT_PATH/trainval/rois.bin
AUGMENTED_DATASET=$MODEL_OUT_PATH/trainval_augmented
CLASSIFICATION_PATH=$MODEL_OUT_PATH/classification_db
SEGMENTATION_PATH=$MODEL_OUT_PATH/segmentation_db
ML_FOLDER=./config/
CAFFE_NETS_PATH=$ML_FOLDER
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
SEGMENTATATION_BATCH_SIZE=1
SEGMENTATION_TRAIN_EPOCHS=30
SEGMENTATION_SNAPSHOT_INTERVAL=1

cd $TRAFFIC_SIGNS_SEGMENTATION_FOLDER

PYTHONPATH="${PYTHONPATH}../:../apollo_python_common/:../apollo_python_common/protobuf/:./utils/:./"
export PYTHONPATH

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
