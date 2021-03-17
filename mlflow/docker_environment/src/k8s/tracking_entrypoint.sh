#/bin/bash

# Launch our mlflow tracking server
set -e
set -x

mlflow server --backend-store-uri=postgresql://${MLFLOW_USER}:${MLFLOW_PASS}@${MLFLOW_DB_ADDR}:${MLFLOW_DB_PORT}/${MLFLOW_DB_NAME} --default-artifact-root=${MLFLOW_ARTIFACT_PATH} --host 0.0.0.0 --port 80
