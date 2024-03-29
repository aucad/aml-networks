ARG BUILD_ARCH=amd64

FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y \
    && apt install -y build-essential git bc software-properties-common libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.9-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --set python3 /usr/bin/python3.9 \
    && python3 -m pip install --upgrade pip setuptools wheel

RUN mkdir -p /usr/src/aml-networks \
    && git clone --recurse-submodules https://github.com/aucad/aml-networks.git "/usr/src/aml-networks" \
    && cd /usr/src/aml-networks/RobustTrees && make -j4

RUN pip3 install -r "/usr/src/aml-networks/requirements.txt" --user \
    && python3 -m pip install "/usr/src/aml-networks/RobustTrees/python-package" --user

WORKDIR ./usr/src/aml-networks