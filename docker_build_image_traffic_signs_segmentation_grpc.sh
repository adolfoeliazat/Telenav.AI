#!/usr/bin/env bash

PORT=${1:-31337}
CUDA_VISIBLE_DEVICES=0

docker build -t telenav/traffic_signs_segmentation_grpc -f traffic_signs_segmentation/docker/caffe/Dockerfile --build-arg PORT=$PORT --build-arg CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES .

# Run the container:
# sudo nvidia-docker run --name traffic_signs_segmentation --net=host  --rm -ti telenav/traffic_signs_segmentation_grpc 