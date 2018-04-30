# Traffic signs object detection using [retinanet - Focal Loss for Dense Object Detection]
For more details, please refer to [arXiv paper](https://arxiv.org/abs/1708.02002).
Original Keras implementation from [fizyr](https://github.com/fizyr/keras-retinanet)

This code was tested with:  

- CUDA_VERSION 9.0.176  (required to be installed on the machine)
- CUDNN_VERSION 7.0.5.15 (required to be installed on the machine)
- Tensorflow 1.6  
- Keras 2.1.3  
- Ubuntu 16.04.4


**A. Get the artifacts**

Download required artifacts:  

1. Pre-trained model:  
    * Download it from https://artefacts2.skobbler.net/repository/openai/retinanet_model_v0.0.1.tar.gz .  
    * Extract it into a local folder _{ARTIFACTS_PATH}_: `tar xvzf retinanet_model_v0.0.1.tar.gz`  
    * See the readme.txt file located in {ARTIFACTS_PATH}/model for a list of traffic signs supported.  
2. Train (labeled) data from {TRAIN_DATA_REPOSITORY_PATH} into a local folder _{ARTIFACTS_PATH}/train_data_. Extract files from archive in this folder.  
3. Test (unlabeled) data from {TEST_DATA_REPOSITORY_PATH} into a local folder _{ARTIFACTS_PATH}/test_data_. Extract files from archive in this folder.  

**B. Installation**

We have the assumption that everything that is going to run, will run from a container docker. After you get the code:  

1. goto build folder: `cd ./imagerecognition/python_modules`  
2. build docker image used for development:  
    * `chmod +x ./docker_build_image_retinanet_mq.sh`  
    * `sh ./docker_build_image_retinanet_mq.sh`  
3. run the container: `sudo nvidia-docker run -v {ARTIFACTS_PATH}:/data/ --name retinanet -it telenav/retinanet_mq /bin/bash`  
    It will start run a new docker container where current folder is the right one to run train or predict steps detailed below.  
4. run in container:
    * `sudo chmod 777 /data/`  
    

**C. Training** 

1. Edit _train.sh_ script located in the current folder of the running docker container: `nano train.sh`  
    * For _WEIGHTS_ parameter:  
        * Use the value as 'imagenet' when you want to start a train almost from scratch.  
        * For a fine tune set the value as _/data/model/traffic_signs_python35.h5_.
    * For _TRAIN_PATH_ parameter set the value as _/data/train_data_.  
    * For _VALIDATION_PATH_ parameter set the value  _/data/train_data_.  
   For a more accurate evaluation during training phase you can split the train data in train and validation dataset.  
2. Run `train.sh` script. The model checkpoints (weights) will be saved in _./snapshots_  

**D. Predicting** 

1. Edit _predict.sh_ script located in the current folder of the running docker container: `nano predict.sh`  
    * For _WEIGHTS_ parameter:  
        * set the value _/data/model/traffic_signs_python35.h5_ when you just want to generate predictions using our pre-trained model.  
        * set the value as the path to one of your trained model as was described at step C if you trained a new model before.  
    * For _TRAIN_META_FILE_ set the value as _/data/train_data/rois.bin_  
    * _INPUT_PATH_ is the folder containing the images you want to use for your detections. Set its value to your's validation dataset or to _/data/test_data_.  
    * _THRESHOLD_FILE_ is a path to a json file containing confidence thresholds per class. The parameter's value can be set to:  
        * _SAME_ : then all classes will have as minimum threshold the value specified in _LOWEST_SCORE_THRESHOLD_.  
        * _/data/model/classes_thresholds.json_ : in order to use the thresholds we already computed for the given model  
    * For _LOWEST_SCORE_THRESHOLD_ see the description of _THRESHOLD_FILE_ above.  
    * _OUTPUT_PATH_ should be set to a folder where the images with detected traffic signs will be saved (e.g _/data/output_). Also, in that folder will be generated the file _rois_retinanet.bin_ containing predicted ROIs for all images serialized in protobuf format.  
2. Run `predict.sh` script. Ignore any error having the following format: _Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x7f7ef3f01320>>_  

**E. Evaluating** 

1. Edit _evaluate.sh_ script located in the current folder of the running docker container: _nano evaluate.sh_  
    * For _TEST_ROI_ parameter: set the path to the file containing ground truth ROIs for traffic signs in the test or validation dataset (e.g. _/data/train_data/rois.bin_).  
    * For _PREDICT_ROI_ parameter: set the path to the file containing serialized detections in protobuf format generated at step D in _OUTPUT_PATH_ folder (e.g. _./output/rois_retinanet.bin_).  
    * _RESULT_FILE_ parameter indicates the text file where evaluations metrics will be saved.  
2. Run `evaluate.sh` script.

**F. Optional: Generates best confidence thresholds maximizing the metric TP/(TP+FP+FN)** 

1. Edit _generate_best_thresholds.sh_ script located in the current folder of the running docker container.  
    * For _TEST_ROI_ parameter: set the path to the file containing ground truth ROIs for traffic signs in the validation dataset.  
    * For _PREDICT_ROI_ parameter: set the path to the file containing serialized detections in protobuf format generated at step D in _OUTPUT_PATH_ folder (e.g. _./output/rois_retinanet.bin_).  
    * _RESULT_FILE_ parameter indicates the path to the json file where best confidence thresholds will be saved. Later, this file can be used at step D for parameter _THRESHOLD_FILE_.  
2. Run `generate_best_thresholds.sh` script.  