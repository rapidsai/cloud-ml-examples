FROM rapidsai/rapidsai-nightly:0.14-cuda10.1-runtime-ubuntu18.04-py3.7

ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    CLOUD_PATH="/opt/ml/code"

RUN apt-get update && apt-get install -y --no-install-recommends build-essential
RUN source activate rapids && pip install sagemaker-containers

COPY code/rapids_cloud_ml.py $CLOUD_PATH/rapids_cloud_ml.py
COPY code/train.py $CLOUD_PATH/train.py
ENV SAGEMAKER_PROGRAM $CLOUD_PATH/train.py

WORKDIR $CLOUD_PATH