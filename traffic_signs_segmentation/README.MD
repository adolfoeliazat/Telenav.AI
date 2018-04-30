# Traffic signs segmentation and classification
The method uses a segmentation model to detect the signs and a classification model to distinguish between different sign types. The segmentation model uses Fully Convolutional Networks, for more details please refer to https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf. The classification model is a modified AlexNet.

This code was tested with:
 - CUDA_VERSION 8
 - CUDNN_VERSION 6
 - Caffe vs 1.0
 - Ubuntu 16.04

## 1.Get the models

 - Download the models and network definitions from   
https://artefacts2.skobbler.net/repository/openai/CaffeModel\_08\_01_2018.tar.gz
 - Consider that *{HOST\_PATH\_TO\_PYTHON\_MODULES}* is the path to *python_modules* dir from **imagerecognition** repository and extract the files in *{HOST\_PATH\_TO\_PYTHON\_MODULES}/traffic\_signs\_segmentation/config*
 
 **Folder content :**
 
|  file name|  details|
|--|--|
| classification.caffemodel |model used to classify traffic signs  |
|classification_solver.prototxt | classification solver file
|classification_train_val.prototxt | classification network used to for training. A modified Alexnet with Xavier initialization
|mean.blob, mean.npy | image mean files used by the classification network
|net.protoxt|segmentation network used for inference
|segmentation\_initial\_weights.caffemodel|segmentation network pre-trained model http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel
|segmentation_solver.prototxt|segmentation network solver
|segmentation_train_val.prototxt|segmentation network used for training: https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s
|seggmodel.caffemodel|trained segmentation model

## 2. Train model

2.1 Create a docker container with Caffe

 - *{HOST_PATH_TO_DATA}*is the train dataset path
 -  Run ***imagerecognition/python\_modules/docker\_build\_image\_traffic\_signs\_segmentation_train.sh***
 - Run ***sudo nvidia-docker run --name traffic\_signs\_segmentation --net=host --rm -ti 
  -v {HOST\_PATH\_TO\_PYTHON\_MODULES}:python_modules -v {HOST_PATH_TO_DATA}:data telenav/traffic\_signs\_segmentation_train***  

2.2 Create dataset

 - Run ***python_modules/traffic_signs_segmentation/tools/create_dataset.sh***
 - This will create the lmdb files needed for training in */data/caffe_model/*

2.3 Train

 - Run ***python_modules/traffic_signs_segmentation/tools/train.sh***
 - Training may very depending on the train dataset size, around 1 week for 30 epochs and 20k images. Intermediary models are saved in */data/caffe_model/*
 
## 3. Test and evaluate

 3.1 Test
 
 - Running ***python_modules/traffic_signs_segmentation/tools/predict.sh*** at this point will generate predictions with the default downloaded weights
 - After training */data/caffe_model/* dir will contain snapshots with the models for segmentation and classification
 - Edit *python_modules/traffic_signs_segmentation/inference_server.py*: replace **CLASSIFICATION_MODEL**, **SEGMENTATION_MODEL**, **MEAN**, **MEAN_BLOB** constants with the paths to the previously trained models
 - Edit *python_modules/traffic_signs_segmentation/tools/predict.sh* replace: **INPUT_PATH** with the folder containing the images to test and **OUTPUT_PATH** with the folder that will contain the generated **rois.bin**
 - Run ***python_modules/traffic_signs_segmentation/tools/predict.sh*** to generate predictions with your trained model
 
 3.2 Evaluate
 
 - Edit *python_modules/traffic_signs_segmentation/tools/evaluate.sh*:
	 * For **TEST_ROI** parameter: set the path to the file containing ground truth ROIs for traffic signs in the test or validation dataset (e.g. /data/train_data/rois.bin).
	 * For **PREDICT_ROI** parameter: set the path to the file containing serialized detections in protobuf format generated at step 2.4 in **OUTPUT_PATH** folder
	 * **RESULT_FILE** parameter indicates the text file where evaluations metrics will be saved.
 - Run ***python_modules/traffic_signs_segmentation/tools/evaluate.sh***