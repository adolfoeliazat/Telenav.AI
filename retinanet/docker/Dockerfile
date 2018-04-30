FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

MAINTAINER Adrian Margin

ARG PYTHON_VERSION=3.5
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_VERSION=4.2.0
ARG CONDA_DIR=/opt/conda
ARG TINI_VERSION=v0.16.1
ARG USERNAME=docker
ARG USERID=1000

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=$CUDA_HOME
ENV PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
    libcudnn7=7.0.5.15-1+cuda9.0 \
    libcudnn7-dev=7.0.5.15-1+cuda9.0 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils build-essential libssl-dev libffi-dev python-dev \
        cmake \
        git \
        ca-certificates \
        libice6 libsm6 libxt6 libxrender1 libfontconfig1 libcups2 graphviz \
        curl \
        unzip \
        sudo \
        nano \
        ffmpeg \
        wget \
        libgtk2.0-0 \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libleveldb-dev \
        liblmdb-dev \
        libsnappy-dev \
        libpng12-dev && \
    rm -rf /var/lib/apt/lists/*

# Conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN apt-get update && \
    wget https://repo.continuum.io/archive/Anaconda$CONDA_PYTHON_VERSION-$CONDA_VERSION-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/anaconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG CONDA=$CONDA_DIR/bin/conda

ARG PIP=$CONDA_DIR/bin/pip

RUN $CONDA update --yes conda

RUN $CONDA install -y h5py Pillow ipywidgets scikit-learn opencv \
  matplotlib pandas bcolz sympy scikit-image mkl-service flask

RUN  $PIP install --upgrade pip
RUN  $PIP install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp35-cp35m-linux_x86_64.whl
RUN  $PIP install keras==2.1.3
RUN  $PIP install grpcio statsd openpyxl tqdm pika elasticsearch
RUN  $PIP install dask --upgrade
RUN  $PIP install git+https://github.com/broadinstitute/keras-resnet

# user's home dir should be mapped from EFS
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
  chown $USERNAME $CONDA_DIR -R && \
  adduser $USERNAME sudo && \
  echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME
COPY ./ ./python_modules
RUN sudo chmod -R 777 ./python_modules/*
WORKDIR /home/$USERNAME/python_modules/retinanet/tools
