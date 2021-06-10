FROM docker.io/rapidsai/rapidsai-core:0.19-cuda11.2-runtime-ubuntu18.04-py3.8

RUN source activate rapids \
    && pip install xgboost \
    && pip install gcsfs \
    # install fsspec back since gcfcs installs 2021.5.0 which breaks reading from gs:// paths
    && pip install fsspec==2021.4.0 
    