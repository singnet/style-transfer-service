## Create container using nvidia-docker and add shared memory size argument
FROM nightseas/cuda-torch:latest

ARG git_owner="singnet"
ARG git_repo="style-transfer"
ARG git_branch="master"
ARG snetd_version

ENV SINGNET_REPOS=/opt/singnet
ENV PROJECT_ROOT=${SINGNET_REPOS}/${git_repo}
ENV SERVICE_DIR=${PROJECT_ROOT}/service

## Installing common dependencies and python3-pip
RUN apt-get update && \
    apt-get install -y sudo python3-pip && \
    pip3 install -U pip==20.3.4

# Installing snet-daemon + dependencies
RUN SNETD_GIT_VERSION=`curl -s https://api.github.com/repos/singnet/snet-daemon/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")' || echo "v5.0.1"` && \
    SNETD_VERSION=${snetd_version:-${SNETD_GIT_VERSION}} && \
    cd /tmp && \
    wget https://github.com/singnet/snet-daemon/releases/download/${SNETD_VERSION}/snet-daemon-${SNETD_VERSION}-linux-amd64.tar.gz && \
    tar -xvf snet-daemon-${SNETD_VERSION}-linux-amd64.tar.gz && \
    mv snet-daemon-${SNETD_VERSION}-linux-amd64/snetd /usr/bin/snetd && \
    rm -rf snet-daemon-*

# Cloning service repository and downloading models
RUN mkdir -p ${SINGNET_REPOS} && \
    cd ${SINGNET_REPOS} &&\
    git clone -b ${git_branch} https://github.com/${git_owner}/${git_repo}.git &&\
    cd ${SERVICE_DIR} &&\
    cd original-lua-code &&\
    . ./models/download_models.sh

# Installing projects's original dependencies and building protobuf messages
RUN cd ${PROJECT_ROOT} &&\
    pip3 install -r requirements.txt &&\
    sh buildproto.sh

WORKDIR ${PROJECT_ROOT}
