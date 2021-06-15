# RAPIDS Cloud Machine Learning

RAPIDS is a suite of open-source libraries that bring GPU acceleration to data science pipelines. Users building cloud-based machine learning experiments can take advantage of this acceleration throughout their workloads to build models faster, cheaper, and more easily on the cloud platform of their choice. The [cloud-ml-examples](https://github.com/rapidsai/cloud-ml-examples) repository provides example notebooks and "getting started" code samples and this Docker repository provides a ready to run Docker container with RAPIDS and libraries/SDKs for AWS SageMaker, Azure ML and Google AI Platform. 

**NOTE:** Review our [prerequisites](#prerequisites) section to ensure your system meets the minimum requirements for RAPIDS.

### Current Version - RAPIDS v21.06

The RAPIDS images are based on [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda), and are intended to be drop-in replacements for the corresponding CUDA
images in order to make it easy to add RAPIDS libraries while maintaining support for existing CUDA applications.

### Image Tag Naming Scheme

The tag naming scheme for RAPIDS images incorporates key platform details into the tag as shown below:
```
21.06-cuda11.0-base-ubuntu18.04-py3.8
 ^       ^    ^        ^         ^
 |       |    type     |         python version
 |       |             |
 |       cuda version  |
 |                     |
 RAPIDS version        linux version
```
## Prerequisites

- NVIDIA Pascal™ GPU architecture or better
- CUDA [11.0/11.2](https://developer.nvidia.com/cuda-downloads) with a compatible NVIDIA driver
- Ubuntu 18.04/20.04 or CentOS 7
- Docker CE v18+
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

## More Information

Check out the [RAPIDS HPO](https://rapids.ai/hpo.html) webpage for video tutorials and blog posts.

Please submit issues with the container to this GitHub repository: https://github.com/rapidsai/docker

For issues with cloud-ml-examples file an issue in: https://github.com/rapidsai/cloud-ml-examples
