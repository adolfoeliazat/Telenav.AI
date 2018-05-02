FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
MAINTAINER flaviu.buciuman@telenav.com

ARG USERNAME=docker
ARG USERID=1000
ARG PORT=31337
ARG CUDA_VISIBLE_DEVICES

ENV USERNAME=$USERNAME
ENV USERID=$USERID
ENV PORT=$PORT
ENV CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        automake \
        autoconf \
        build-essential \
        bzip2 \
        ca-certificates \
        cmake \
        curl \
        git-core \
        google-mock \
        pkg-config \
        python3.5 \
        python3.5-dev \
        python3-pip \
        python3-numpy \
        python3-setuptools \
        python3-venv \
        libboost-all-dev \
        libopencv* \
        libleveldb-dev \
        libsnappy-dev \
        libatlas-base-dev \
        libhdf5* \
        libgflags-dev \
        libgoogle-glog-dev \
        libssl-dev \
        liblmdb-dev \
        libtool \
        unzip \
        sudo \
        nano \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Install newer version of protobuf
RUN wget https://github.com/google/protobuf/archive/v3.2.1.tar.gz && \
    tar xzvf v3.2.1.tar.gz && cd protobuf-3.2.1/ && \
    ./autogen.sh && ./configure --prefix=/usr && \
    make -j8 && make install && \
    cd ../ && rm -rf protobuf-3.2.1/ && rm ./v3.2.1.tar.gz

# Caffe
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT
ENV CLONE_TAG=1.0

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -D python_version=3 ..  && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# Install pip dependencies
RUN pip3 install pip==9.0.3 && \
    pip3 install appdirs==1.4.0 && \
    pip3 install cycler==0.10.0 && \
    pip3 install dask==0.13.0 && \
    pip3 install decorator==4.0.11 && \
    pip3 install enum34==1.1.6 && \
    pip3 install grpcio==1.1.0 && \
    pip3 install grpcio-tools==1.1.0 && \
    pip3 install matplotlib==2.0.0 && \
    pip3 install networkx==1.11 && \
    pip3 install numpy==1.12.0 && \
    pip3 install olefile==0.44 && \
    pip3 install opencv-python==3.2.0.6 && \
    pip3 install packaging==16.8 && \
    pip3 install Pillow==4.0.0 && \
    pip3 install protobuf==3.2.0 && \
    pip3 install pyparsing==2.1.10 && \
    pip3 install python-dateutil==2.6.0 && \
    pip3 install pytz==2016.10 && \ 
    pip3 install scikit-image==0.12.3 && \
    pip3 install scipy==0.18.1 && \
    pip3 install six==1.10.0 && \
    pip3 install toolz==0.8.2 && \
    pip3 install lmdb

# Setup environmental variables
ENV PYTHONPATH $HOME/libs/:$PYTHONPATH


