# Traffic signs object detection using [retinanet - Focal Loss for Dense Object Detection]
For more details, please refer to [arXiv paper](https://arxiv.org/abs/1708.02002).
Original Keras implementation from [fizyr](https://github.com/fizyr/keras-retinanet)

This code was tested with:  

- CUDA_VERSION 9.0.176  (required to be installed on the machine)
- CUDNN_VERSION 7.0.5.15 (required to be installed on the machine)
- Tensorflow 1.7  
- Keras 2.1.5  
- Ubuntu 16.04.4


**A. Get the artifacts**
 
Download:
 * the zip file containing datasets from [here](https://s3.eu-central-1.amazonaws.com/telenav.ai/telenav_ai_dataset.zip). Then extract its content into a local folder: _{ARTIFACTS_PATH}_.
After that, in the _{ARTIFACTS_PATH}_ folder should exist the following folders: _train_data_ and _test_data_. Please read carefully the dataset license enclosed in the zip file.
 * the zip file containing pre-trained model from [here](https://s3.eu-central-1.amazonaws.com/telenav.ai/model_retinanet.zip). Then extract its content into a local folder _{ARTIFACTS_PATH}/model_.

**B. Installation**

We assume that everything that is going to run, will run from a docker container. Just execute:  

1. `git clone https://github.com/Telenav/Telenav.AI`
1. go inside: `cd ./Telenav.AI`
2. build the docker image used for development:  
    * `chmod +x ./docker_build_image_retinanet_mq.sh`  
    * `sh ./docker_build_image_retinanet_mq.sh`  
3. run the container: `sudo nvidia-docker run -v {ARTIFACTS_PATH}:/data/ --name retinanet -it telenav/retinanet_mq /bin/bash`  
    It will start the docker container and a new BaSH session into it.  
4. run in container:
    * `sudo chmod 777 /data/`
        

**C. Training** 

1. Edit _train.sh_ script located in the current folder of the running docker container: `nano train.sh`  
    * When one wants to train starting from: 
        * imagenet weights just use as parameter _--imagenet-weights_. 
        * from custom Keras weights just use for _WEIGHTS_ parameter a value like _/data/model/retinanet_resnet50_traffic_signs_v002.h5_ and for scripts the parameter _--weights $WEIGHTS_ instead of _--imagenet-weights_.  
    * For _TRAIN_PATH_ parameter set the value as _/data/train_data_.  
    * For _VALIDATION_PATH_ parameter set the value  _/data/train_data_.  
   For a more accurate evaluation during training phase you can split the train data in train and validation dataset.  
2. Run `train.sh` script. The model checkpoints (weights) will be saved in _./snapshots_  

**D. Predicting** 

1. Edit _predict.sh_ script located in the current folder of the running docker container: `nano predict.sh`  
    * For _WEIGHTS_ parameter:  
        * set the path to the trained model file, like _/data/model/retinanet_resnet50_traffic_signs_v002.pb_.  
    * For _TRAIN_META_FILE_ set the value: _/data/train_data/rois.bin_  
    * _INPUT_PATH_ is the folder containing the images you want to use for your detections. Set its value to your's validation dataset or to _/data/test_data_.  
    * _THRESHOLD_FILE_ is a path to a json file containing confidence thresholds per class. The parameter's value can be set to:  
        * _SAME_ : then all classes will have as minimum threshold the value specified in _LOWEST_SCORE_THRESHOLD_.  
        * _/data/model/classes_thresholds.json_ : when one wants to use the thresholds already computed for a given model. The _classes_thresholds.json_ can be generated as is explained at point F.
    * For _LOWEST_SCORE_THRESHOLD_ see the description of _THRESHOLD_FILE_ above.  
    * _OUTPUT_PATH_ should be set to a folder where the images with detected traffic signs will be saved (e.g _/data/output_). Also, in that folder will be generated the file _rois_retinanet.bin_ containing predicted ROIs for all images serialized in protobuf format.  
2. Run `predict.sh` script. Ignore any error having the following format: _Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x7f7ef3f01320>>_  

**E. Basic evaluation** 

1. Edit _evaluate.sh_ script located in the current folder of the running docker container: _nano evaluate.sh_  
    * For _TEST_ROI_ parameter: set the path to the file containing ground truth ROIs for traffic signs of yours validation dataset (e.g. _/data/train_data/rois.bin_).  
    * For _PREDICT_ROI_ parameter: set the path to the file containing serialized detections in protobuf format generated at step D in _OUTPUT_PATH_ folder (e.g. _./output/rois_retinanet.bin_).  
    * _RESULT_FILE_ parameter indicates the text file where evaluations metrics will be saved.  
2. Run `evaluate.sh` script.

**F. Optional: Generates best confidence thresholds maximizing the metric TP/(TP+FP+FN)** 

1. Edit _generate_best_thresholds.sh_ script located in the current folder of the running docker container.  
    * For _TEST_ROI_ parameter: set the path to the file containing ground truth ROIs for traffic signs in the validation dataset.  
    * For _PREDICT_ROI_ parameter: set the path to the file containing serialized detections in protobuf format generated at step D in _OUTPUT_PATH_ folder (e.g. _./output/rois_retinanet.bin_).  
    * _RESULT_FILE_ parameter indicates the path to the json file where best confidence thresholds will be saved. Later, this file can be used at step D for parameter _THRESHOLD_FILE_.  
2. Run `generate_best_thresholds.sh` script.  
