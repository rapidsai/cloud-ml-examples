#!/bin/bash
set -e

# Overwrite HOME to WORKSPACE
export HOME=$WORKSPACE

# Install gpuCI tools
curl -s https://raw.githubusercontent.com/rapidsai/gpuci-tools/main/install.sh | bash
source ~/.bashrc
cd ~

# Set vars
DOCKER_IMG="rapidsai/cloud-ml"
DOCKER_TAG="${RAPIDS_VER}-cuda${CUDA_VER}-${IMG_TYPE}-${LINUX_VER}-py${PYTHON_VER}"
DOCKERFILE="common/docker/Dockerfile.training.unified"

# Show env
gpuci_logger "Exposing current environment..."
env

# Print dockerfile
gpuci_logger ">>>> BEGIN Dockerfile <<<<"
cat ${DOCKERFILE}
gpuci_logger ">>>> END Dockerfile <<<<"

# Docker Login
docker login --username "${DH_USER}" --password "${DH_TOKEN}"

# Build Image
gpuci_logger "Starting build..."
set -x # Print build command
docker build \
    --squash \
    --build-arg "RAPIDS_VER=${RAPIDS_VER}" \
    --build-arg "CUDA_VER=${CUDA_VER}" \
    --build-arg "IMG_TYPE=${IMG_TYPE}" \
    --build-arg "LINUX_VER=${LINUX_VER}" \
    --build-arg "PYTHON_VER=${PYTHON_VER}" \
    -t "${DOCKER_IMG}:${DOCKER_TAG}" \
    -f "${DOCKERFILE}" \
    .
set +x

# List image info
gpuci_logger "Displaying image info..."
docker images ${DOCKER_IMG}:${DOCKER_TAG}


# Upload image
gpuci_logger "Starting upload..."
GPUCI_RETRY_MAX=5
GPUCI_RETRY_SLEEP=120
gpuci_retry docker push ${DOCKER_IMG}:${DOCKER_TAG}
