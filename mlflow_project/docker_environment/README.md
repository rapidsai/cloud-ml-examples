# End to End - RAPIDS, hyperopt, and MLflow, on Kubernetes.
## Setup and Requirements
#### Hardware and Software
- **Cloud based Kubernetes cluster OR Microk8s compatible deployment system**.
    - Worker node(s) with NVIDIA GPU.
        - Pascal or higher is required to run RAPIDS examples.
    - NVIDIA [Kubernetes plugin](https://github.com/NVIDIA/k8s-device-plugin) (comes integreated with Microk8s).
    - Correctly installed and configured [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl).
- **AWS account capable of creating, listing, and populating S3 buckets**.
    - Ensure appropriate [permissions](https://docs.aws.amazon.com/AmazonS3/latest/user-guide/set-permissions.html).

#### Assumptions and Naming Conventions
- All shell commands are assumed to be run within the `/cloud-ml-examples/mlflow_project/docker_environment` directory.

- There are a number of configuration parameters that will be specific to your _environment_ and _deployment_:
    - `ARTIFACT BUCKET` : The name of the S3 bucket where MLflow will be configured to store experiment artifacts.
    - `AWS ACCT ID`: Service account ID
    - `AWS ACCT SECRET KEY`: Secret Key associated with `AWS ACCT ID`
    - `CONTAINER REPO URI` : Uri of the kubernetes container repo
    - `CONTAINER REPO PORT` : Port for the kubernetes container repo
    - `S3 REGION` : Region where your S3 bucket is created.
    
- `rapids-mlflow-example` : The tag for the training container for this exercise.
    - This is the base container object used by MLflow to run experiments in Kubernetes.
    - It must contain the appropriate RAPIDS libraries, mlflow, hyperopt, psycopg2 (for Postgres), and boto3 (for S3).
        - See [Dockerfile.training](Dockerfile.training)

## Configuration
### Kubernetes (k8s) environment.
#### Cluster Deployment
- **Cloud Service Provider (CSP) cluster**.
    - **TODO** Setup is outside our scope

- **Local Cluster**.
    - Docker configuration
        - Install the NVIDIA [docker container toolkit](https://github.com/NVIDIA/nvidia-docker).
        - `sudo vi /etc/docker/daemon.json` and add the json below. This will set your docker installation to use the 
        nvidia engine by default.
            ```json
            {
                "default-runtime": "nvidia",
                "runtimes": {
                    "nvidia": {
                        "path": "nvidia-container-runtime",
                        "runtimeArgs": []
                    }
                }
            }
            ```
    - Install and deploy [Microk8s](https://microk8s.io/) on a single machine.
        - Enable addons: `config`, `dns`, `gpu`, and `helm3`

#### Cluster Configuration
- Add S3 credentials to k8s cluster
    - This adds our S3 bucket credentials to the k8s cluster in a controlled way, that lets us avoid duplicating them
    in templates or passing them at experiment run time.
    - Note: these values are used within the `k8s_job_template.yaml` file, to populate `AWS_ACCESS_KEY_ID`, and
    `AWS_SECRET_ACCESS_KEY`, respectively.
    - `kubectl create secret generic awsacct --from-literal=awsacct=[AWS ACCT ID]`
    - `kubectl create secret generic awskey --from-literal=awskey=[AWS ACCT SECRET KEY]`
- Deploy a Postgres database service to be the MLflow tracking endpoint.
    - **TODO** HELM CHART
- Deploy an MLflow tracking server.
    - **TODO** WRITE HELM CHART
    - Provides experiment and published model UI.
    
### S3 as an _[artifact endpoint](https://www.mlflow.org/docs/latest/tracking.html#artifact-stores)_.
Steps here assume that you have a properly configured [awscli](https://aws.amazon.com/cli/) installation.
- Create a bucket.
    - `aws s3api create-bucket --bucket [ARTIFACT BUCKET] --region [S3 REGION]`
- Attach bucket policy.
    - **TODO**
    
### Create and publish a training container
- Build a training container and publish it to your k8s environment.
    - This creates the base docker container that MLflow will inject our project into, and deploy into our k8s cluster
    for training.
    - `docker build --tag rapids-mlflow-training:latest --file Dockerfile.training .`
    - `docker tag rapids-mlflow-training:latest [CONTAINER_REPO_URI]:[CONTAINER_REPO_PORT]/rapids-mlflow-training:latest`

- Create a new experiment, with the appropriate artifact endpoint.
    - Note: If you don't perform this step, MLflow will default to using the local file system for artifact storage, which
    means that your saved artifacts will live only as long as the training pod exists.
    - `mlflow experiments create --experiment-name RAPIDS-MLFLOW --artifact-location s3://[S3 BUCKET]/[S3 ARTIFACT PATH]`

### Run an RAPIDS + hyperopt experiment in MLflow + k8s
- `mlflow run . --backend kubernetes --backend-config ./k8s_config.json -e hyperopt --experiment-name RAPIDS-S3-1 
    -P conda-env=/rapids/envs/conda.yaml -P fpath=/rapids/data/airline_small.parquet`

