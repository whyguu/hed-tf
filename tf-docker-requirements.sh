#!/bin/bash
docker pull tensorflow/tensorflow:latest-gpu-py3
nvidia-docker run -ti --name=ygw-tf -p 8080:8888 -p 2121:22 -p 6666:6006 -v $pwd/docker:/workspace -w /workspace tensorflow/tensorflow:latest-gpu-py3 bash
apt-get update
pip3 install pyyaml
pip3 install opencv-python>=3.0
apt-get install -y libsm6 libxrender1 libfontconfig1 libxtst6