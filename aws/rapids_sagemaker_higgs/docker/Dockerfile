FROM rapidsai/rapidsai-core:22.12-cuda11.5-runtime-ubuntu18.04-py3.9

# add sagemaker-training-toolkit [ requires build tools ], flask [ serving ], and dask-ml
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \ 
    && source activate rapids \
    && pip3 install sagemaker-training cupy-cuda115 flask \
    && pip3 install --upgrade protobuf
    
# Copies the training code inside the container
COPY rapids-higgs.py /opt/ml/code/rapids-higgs.py

# Defines rapids-higgs.py as script entry point
ENV SAGEMAKER_PROGRAM rapids-higgs.py