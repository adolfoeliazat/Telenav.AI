#!/usr/bin/env bash

PORT=${1:-31337}
CUDA_VISIBLE_DEVICES=0

docker build -t telenav/traffic_signs_segmentation_train -f traffic_signs_segmentation/docker/train/Dockerfile --build-arg PORT=$PORT --build-arg CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES .

# Run the container:
# sudo nvidia-docker run --name traffic_signs_segmentation --net=host  --rm -ti telenav/traffic_signs_segmentation_train -d -v /home/flaviub/:/home/flaviub/ -v /data/:/data