# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# located at the root directory.
# ---------------------------------------------------------------

FROM nvcr.io/nvidia/pytorch:20.01-py3
RUN yes | pip install tensorboardX==1.2
RUN yes | pip install wget
RUN yes | pip install scikit-learn
RUN yes | pip install ipymd
RUN yes | pip install Pillow
RUN yes | apt-get update
RUN yes | apt-get install tmux
RUN yes | apt-get install htop
RUN yes | apt-get install bc
RUN yes | conda install -c conda-forge accimage
RUN yes | pip install notifiers
RUN yes | pip install pyyaml
RUN yes | pip install ruamel.yaml
RUN yes | apt install rsync
RUN yes | apt install git
RUN yes | pip install dataclasses
RUN yes | pip install simple_parsing
RUN yes | pip install munch
COPY inputrc /root/.inputrc


