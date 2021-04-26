FROM rapidsai/rapidsai-core:0.19-cuda11.0-base-ubuntu18.04-py3.8
RUN apt-get update && \
apt-get install -y fuse && \
source activate rapids && \
pip install azureml-mlflow && \
pip install azureml-dataprep && \
pip install dask-ml
