**Telenav.AI**

At Telenav, we've always been about making people's lives less stressful, more productive, and more fun when they are on the go. Over the years, we have paved the way for a bigger end-to-end mobility transformation for all people on the go — starting with connected car solutions for drivers.

Telenav's OpenTerra project is looking for ways to accelerate map improvements that help us achieve this goal. Under that umbrella, we launched [OpenStreetCam](http://openstreetcam.org/), an open street level imagery collection platform. Drivers around the world have contributed well over 100M images containing valuable information such as sign posts

In order to support the mapping community around the world, we have decided to open-source our our datasets and our object detection code.

**Datasets**

The datasets are part of our Object Detection Competition. You can find the dataset [here](https://s3.eu-central-1.amazonaws.com/telenav.ai/telenav_ai_dataset.zip). It contains over 45 000 manually annotated images containing more than 55 000 signs divided in 23 different classes such as : traffic signals, stop signs, speed limits and turn restrictions.

A smaller version of the dataset, containing only 1000 images, can be found [here ](https://s3.eu-central-1.amazonaws.com/telenav.ai/telenav_ai_dataset_sample.zip)

**Code** 

This repository contains all the necessary resources to train our neural networks on the 
datasets provided. Moreover, you can train them on your own datasets if you bring them to the required protobuf format.

In order to see more details, check out [RetinaNet](https://github.com/Telenav/Telenav.AI/tree/master/retinanet) for instructions regarding our RetinaNet implementation or [Segmentation](https://github.com/Telenav/Telenav.AI/tree/master/traffic_signs_segmentation) for the Segmentation based one.

Also, see this [notebook](https://github.com/Telenav/Telenav.AI/tree/master/apollo_python_common/demo) for a quick demo on how you can use and visualize our data. 
