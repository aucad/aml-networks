FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y
RUN apt install -y build-essential git bc
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y python3.9-dev
RUN apt install -y python3.9-distutils
RUN apt install -y python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --set python3 /usr/bin/python3.9

# RUN git clone --recurse-submodules https://github.com/iotcad/aml-networks.git
COPY . /aml-networks/.
WORKDIR aml-networks/RobustTrees

RUN chmod 777 build.sh
RUN ./build.sh
WORKDIR ..

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt
RUN pip3 install -e "RobustTrees/python-package"