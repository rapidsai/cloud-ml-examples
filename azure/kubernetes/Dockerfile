FROM docker.io/rapidsai/rapidsai-core:21.06-cuda11.2-runtime-ubuntu18.04-py3.8

RUN source activate rapids \
    && pip install xgboost \
    && pip install gcsfs \
    && pip install adlfs 
