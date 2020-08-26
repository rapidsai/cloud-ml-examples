FROM rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04-py3.7

RUN source activate rapids \
    && mkdir /opt/mlflow \
    && pip install \
	psycopg2-binary \
        mlflow \
        hyperopt

COPY airline_small.parquet /rapids/data/airline_small.parquet
COPY envs/ /rapids/envs/
