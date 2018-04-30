#!/usr/bin/env bash

# Building image from dockerfile.
docker build -f retinanet/docker/Dockerfile -t telenav/retinanet_mq .

# Run the container:
# sudo nvidia-docker run -v /data/:/data/ --net=host --name retinanet_mq -d -it telenav/retinanet_mq
# sudo docker exec -it retinanet_mq /bin/bash
# sudo docker logs -f retinanet_mq


