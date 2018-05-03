# Traffic signs segmentation and classification
The method uses a segmentation model to detect the signs and a classification model to distinguish between different sign types. The segmentation model uses Fully Convolutional Networks, for more details please refer to https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf. The classification model is a modified AlexNet.

This code was tested with:
 - CUDA_VERSION 8
 - CUDNN_VERSION 6
 - Caffe vs 1.0
 - Ubuntu 16.04

## 1.Get the models and images dataset

 - Download the pre-trained model from:   
http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel
 - Consider that *{ROOT}* is the path to *Telenav.AI* repository and copy the model in *{ROOT}/traffic\_signs\_segmentation/config*
 - Download the zip file containing datasets from [here](https://s3.eu-central-1.amazonaws.com/telenav.ai/telenav_ai_dataset.zip). Then extract its content into a local folder: _{ARTIFACTS_PATH}_.
   After that, in the _{ARTIFACTS_PATH}_ folder should exist the following folders: _train_data_ and _test_data_.
 
 **Config Folder Content :**
 
|  file name|  details|
|--|--|
|classification_solver.prototxt | classification solver file
|classification_train_val.prototxt | classification network used to for training. A modified Alexnet with Xavier initialization
|net.protoxt|segmentation network used for inference
|segmentation_solver.prototxt|segmentation network solver
|segmentation_train_val.prototxt|segmentation network used for training: https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s
|seggmodel.caffemodel|trained segmentation model

## 2. Train model

2.1 Create a docker container with Caffe

 -  Run ***Telenav.AI/docker\_build\_image\_traffic\_signs\_segmentation_train.sh***
 -  Run ***sudo nvidia-docker run --name traffic\_signs\_segmentation --net=host --rm -ti 
  -v {Root}:/home/Telenav.AI -v {ARTIFACTS_PATH}/train_data:/data/input_data -v {ARTIFACTS_PATH}/test_data:/data/test_data telenav/traffic\_signs\_segmentation_train***  

2.2 Create dataset
  
 - Inside the docker container go to ***/home/Telenav.AI/traffic_signs_segmentation/tools/*** 
 - Run ***create_dataset.sh***
 - This will create the lmdb files needed for training in */data/caffe_model/*

2.3 Train
 
 - Inside the docker container go to ***/home/Telenav.AI/traffic_signs_segmentation/tools/*** 
 - Run ***train.sh***
 - Training may very depending on the train dataset size, around 1 week for 30 epochs and 20k images. Intermediary models are saved in */data/caffe_model/*
 
## 3. Test and evaluate

 3.1 Test
 
 - After training */data/caffe_model/* dir will contain snapshots with the models for segmentation and classification
 - Edit *Telenav.AI/traffic_signs_segmentation/inference_server.py*: replace **CLASSIFICATION_MODEL**, **SEGMENTATION_MODEL**, **MEAN**, **MEAN_BLOB** constants with the paths to the previously trained models
 - Edit *Telenav.AI/traffic_signs_segmentation/tools/predict.sh* replace: **INPUT_PATH** with the folder containing the images to test and **OUTPUT_PATH** with the folder that will contain the generated **rois.bin**
 - Run ***Telenav.AI/traffic_signs_segmentation/tools/predict.sh*** to generate predictions with your trained model
 
 3.2 Evaluate
 
 - Edit *Telenav.AI/traffic_signs_segmentation/tools/evaluate.sh*:
	 * For **TEST_ROI** parameter: set the path to the file containing ground truth ROIs for traffic signs in the test or validation dataset (e.g. /data/test_data/rois.bin).
	 * For **PREDICT_ROI** parameter: set the path to the file containing serialized detections in protobuf format generated at step 2.3 in **OUTPUT_PATH** folder
	 * **RESULT_FILE** parameter indicates the text file where evaluations metrics will be saved.
 - Run ***Telenav.AI/traffic_signs_segmentation/tools/evaluate.sh***