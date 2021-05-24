## **Augment SageMaker with a RAPIDS Conda Kernel**
This section describes the process required to augment a SageMaker notebook instance with a RAPIDS conda environment.

The RAPIDS Ops team builds and publishes the latest RAPIDS release as a packed conda tarball. 
> e.g.: https://rapidsai-data.s3.us-east-2.amazonaws.com/conda-pack/rapidsai/rapids0.19_cuda11.0_py3.8.tar.gz

We will use this packed conda environment to augment the set of Jupyter ipython kernels available in our SageMaker notebook instance.

The key steps of this are as follows:
1. During SageMaker Notebook Instance Startup
   - Select a RAPIDS compatible GPU as the SageMaker Notebook instance type (e.g., ml.p3.2xlarge) 
   - Attach the lifecycle configuration (via the 'Additional Options' dropdown) provided in this directory
2. Launch the instance
3. Once Jupyter is accessible select the 'rapids-XX' kernel when working with a new notebook.