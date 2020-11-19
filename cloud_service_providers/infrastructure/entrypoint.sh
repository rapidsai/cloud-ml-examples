source /conda/etc/profile.d/conda.sh
conda activate rapids

ARGS=( "$@" )
EXEC_CONTEXT=""

# If we're doing SageMaker HPO, this file will exist
aws_hpo_params_path="/opt/ml/input/config/hyperparameters.json"
if [[ -f "${aws_hpo_params_path}" ]]; then
  EXEC_CONTEXT="aws_sagemaker_hpo"
fi

# If we're doing GCP AI-Platform HPO, a number of AIP_XXX values will be set.
if [[ -n "$CLOUD_ML_HP_METRIC_FILE" ]]; then
  EXEC_CONTEXT="gcp_aip_hpo"
fi

if [[ $EXEC_CONTEXT == "aws_sagemaker_hpo" ]]; then
  ## SageMaker
  echo "Running SageMaker HPO entrypoint."
  cd ${AWS_CLOUD_PATH} || exit 1
  if [[ "$1" == "serve" ]]; then
      echo -e "@ entrypoint -> launching serving script \n"
      python serve.py
  else
      echo -e "@ entrypoint -> launching training script \n"
      python train.py
  fi

elif [[ $EXEC_CONTEXT == "gcp_aip_hpo" ]]; then
  # GCP
  echo "Running GCP AI-Platform HPO entrypoint."
  cd /opt/rapids/gcp || exit 1

  echo "Running: entrypoint.py ${ARGS[@]}"
  python entrypoint.py ${ARGS[@]}

else
  # Azure
  # TODO: Azure workflow is substantially different.
  echo "Running AzureML HPO entrypoint."
  cd /opt/rapids/azure || exit 1
  echo "Running: ${ARGS[@]}"
  eval "${ARGS[@]}"
fi
