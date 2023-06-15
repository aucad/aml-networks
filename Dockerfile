FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y \
    && apt install -y build-essential git bc software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.9-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --set python3 /usr/bin/python3.9

RUN mkdir -p /usr/src/aml-networks
COPY . /usr/src/aml-networks/.
RUN git clone --recurse-submodules https://github.com/chenhongge/RobustTrees.git "/usr/src/aml-networks/RobustTrees"
RUN chmod 777 /usr/src/aml-networks/RobustTrees/build.sh && cd /usr/src/aml-networks/RobustTrees && ./build.sh

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install -r /usr/src/aml-networks/requirements.txt
RUN python3 -m pip install -e /usr/src/aml-networks/RobustTrees/python-package

WORKDIR ./usr/src/aml-networks

