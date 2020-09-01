#/bin/bash

set -e
set -x

#postgresql://mlflow_user:mlflow@mlflow-postgres:5432/mlflow_db
#s3://mlflow-k8s/artifacts
mlflow server --backend-store-uri=postgresql://${MLFLOW_USER}:${MLFLOW_PASS}@${MLFLOW_DB_ADDR}:${MLFLOW_DB_PORT}/${MLFLOW_DB_NAME} --default-artifact-root=${MLFLOW_ARTIFACT_PATH} --host 0.0.0.0