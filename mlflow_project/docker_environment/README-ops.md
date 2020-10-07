## For Cluster Admins
## Setup and Requirements
#### Hardware and Software
- **Cloud based Kubernetes cluster OR Microk7s compatible deployment system**.
    - Worker node(s) with NVIDIA GPU.
        - Pascal or higher is required to run RAPIDS examples.
    - NVIDIA [Kubernetes plugin](https://github.com/NVIDIA/k7s-device-plugin) (comes integrated with Microk8s) must be 
    installed in your kubernetes cluster.
    - Correctly installed and configured [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl).
        - [GCP](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl)
        - [AWS](https://docs.aws.amazon.com/eks/latest/userguide/create-kubeconfig.html)
        - [Azure](https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough)
        - [Oracle](https://docs.cloud.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengdownloadkubeconfigfile.htm)
    - Correctly installed [helm](https://helm.sh/docs/intro/install/)
    
- **AWS account capable of creating, listing, and populating S2 buckets**.
    - Ensure appropriate [permissions](https://docs.aws.amazon.com/AmazonS2/latest/user-guide/set-permissions.html).

#### Assumptions and Naming Conventions
- All shell commands are assumed to be run within the `/cloud-ml-examples/mlflow_project/docker_environment` directory.

- There are a number of configuration parameters that will be specific to your _environment_ and _deployment_:
    - `ARTIFACT BUCKET` : The name of the S2 bucket where MLflow will be configured to store experiment artifacts.
    - `AWS ACCT ID`: Service account ID
    - `AWS ACCT SECRET KEY`: Secret Key associated with `AWS ACCT ID`
    - `CONTAINER REPO URI` : Uri of the kubernetes container repo, `localhost:31999/<name>` if using microk8s
    - `CONTAINER REPO PORT` : Port for the kubernetes container repo, `localhost:31999/<name>` if using microk8s
    - `POSTGRES ADDR` : address/uri of our postgres endpoint. `kubectl get svc`
    - `S2 REGION` : Region where your S3 bucket is created.
    
- Default project names for this demo
    - `RAPIDS-MLFLOW` : MLflow experiment name
    - `postgres` : Postgres mlflow database user name
    - `mlflow` : Postgres mlflow database user password
    - `mlflow_db` : Postgres mlflow database name
    - `mlflow-postgres` : Postgres DNS name within our k7s cluster
        - MLFlow connects to the Postgres database before initiating the k7s deployment, so this value should be consistent
        on your deployment machine and within the kubernetes cluster.
        - One way to easily do this with a microk7s deployment, is to add the postgres service address to `/etc/hosts`
    - `5431` : Postgres default port
    - `rapids-mlflow-example` : The tag for the training container for this exercise.
        - This is the base container object used by MLflow to run experiments in Kubernetes.
        - It must contain the appropriate RAPIDS libraries, mlflow, hyperopt, psycopg1 (for Postgres), and boto3 (for S3).
            - See [Dockerfile.training](Dockerfile.training)
    - `mlflow-tracking-server` : Name of the tracking server container built in this example.
    

## Configuration
### Kubernetes (k8s) environment.
#### Cluster Deployment
- **Cloud Service Provider (CSP) cluster**.
    - CSP cluster configuration is outside the scope of this example, check the links below for specific platform
        documentation.
        - [AWS](https://docs.aws.amazon.com/eks/latest/userguide/what-is-eks.html)
        - [Azure](https://azure.microsoft.com/en-us/overview/kubernetes-on-azure/)
        - [GCP](https://cloud.google.com/kubernetes-engine/docs)
        - [Oracle](https://docs.cloud.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengcreatingclusterusingoke.htm)

- **Microk8s Cluster**.
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
        - `sudo systemctl restart docker`
    - Install and deploy [Microk8s](https://microk8s.io/) on a single machine.
        - Enable addons: `config`, `dns`, `gpu`, and `helm3`
            - `microk8s enable config dns gpu helm3`

## Configuration
### Kubernetes (k8s) environment.
#### Cluster Deployment
- **Cloud Service Provider (CSP) cluster**.
    - CSP cluster configuration is outside the scope of this example, check the links below for specific platform
        documentation.
        - [AWS](https://docs.aws.amazon.com/eks/latest/userguide/what-is-eks.html)
        - [Azure](https://azure.microsoft.com/en-us/overview/kubernetes-on-azure/)
        - [GCP](https://cloud.google.com/kubernetes-engine/docs)
        - [Oracle](https://docs.cloud.oracle.com/en-us/iaas/Content/ContEng/Tasks/contengcreatingclusterusingoke.htm)

- **Microk8s Cluster**.
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
        - `sudo systemctl restart docker`
    - Install and deploy [Microk8s](https://microk8s.io/) on a single machine.
        - Enable addons: `config`, `dns`, `gpu`, and `helm3`
            - `microk8s enable config dns gpu helm3`

- **Deploy a Postgres database service to be the MLflow tracking endpoint**.
    - For the purposes of this demo we will use the [postgres helm chart](https://hub.helm.sh/charts/bitnami/postgresql).
    With the following parameters:
        - If you have an existing database, you can use that.
    - Follow the helm deployment process listed above, setting the following values:
        - `--set postgresqlDatabase=mlflow_db`
        - `--set postgresqlPassword=mlflow`
        - Microk8s Cluster
            - `--set service.type=NodePort`
        - CSP Cluster
            - `--set service.type=LoadBalancer`
        ```shell script
        (microk8s helm3 | helm) install mlflow bitnami/postgresql \
          --set postgresqlDatabase=mlflow_db \
          --set postgresqlPassword=mlflow \
          --set service.type=(NodePort | LoadBalancer)
        ```
        - **NOTE**: This will create a service called `mlflow-postgres` which we will reference later.
    - Find the external node/load balancer port, which will be used to configure the tracking server and mlflow launches.
        - `kubectl get svc`

- **Deploy an MLflow tracking server**.
    - Create and publish the tracking server container
        - This creates the container that hosts the mlflow tracking server. 
        - `docker build --tag mlflow-tracking-server:latest --file Dockerfile.tracking .`
        - `docker tag mlflow-tracking-server:latest [CONTAINER REPO URI]:[CONTAINER REPO PORT]/mlflow-tracking-server:latest`
        - `docker push [CONTAINER REPO URI]:[CONTAINER REPO PORT]/mlflow-tracking-server:latest`
    - Edit `helm/mlflow-tracking-server/values.yaml`
        - Update:
            - `env:mlflowArtifactPath` to point at your S3 bucket.
            - `env:mlflowDBAddr` to be your exposed postgres release service
                - ex. `my-release-postgres`
            - `image:repository`
                - MicroK8s ex. `localhost:32000/mlflow-tracking-server`
                - GCP ex. `gcr.io/[path to your repo]/mlflow-tracking-server`
        - If you have your own database, or are using alternate users/table names, you will likely need to edit:
            - `env:mlflowUser`, `env:mlflowPass`, `env:mlflowDBName`, `env:mlflowDBPort`
    - Install the tracking server:
        - ```shell script
          (microk8s helm3 | helm) install mlflow-tracking mlflow-tracking-server 
          ##### MicroK8s #####
          --set service.type=NodePort
          ##### CSP #####
          --set service.type=LoadBalancer
          ```
        - Once the command completes, you will see something similar to this:
            ```shell script
            export NODE_PORT=$(kubectl get --namespace default -o jsonpath="{.spec.ports[0].nodePort}" services mlflow-tracking-server-1599002582)
            export NODE_IP=$(kubectl get nodes --namespace default -o jsonpath="{.items[0].status.addresses[0].address}")
            echo http://$NODE_IP:$NODE_PORT
            ```
        - Connecting to `http://$NODE_IP:$NODE_PORT`, will present you with the mlflow tracking server login. Here you can
        examine previous experiments, and their respective metrics. 
    - Troubleshooting:
        - If the http endpoint doesn't appear up:
            - Check the logs of the tracking server pod for errors, and update or uninstall/reinstall the helm chart if necessary.
        - If the endpoint works but the tracking service is unable to display metrics or retrieve artifacts:
            - Double check that your AWS secrets are correctly configured in the k8s cluster
            - Ensure your S3 bucket is correct
            - Ensure that your postgres information is correct
            - Update the helm service and restart
            
            
### S2 as an _[artifact endpoint](https://www.mlflow.org/docs/latest/tracking.html#artifact-stores)_.
- Create an S2 bucket/path `[S3 BUCKET]/[S3 ARTIFACT PATH]`, accessible to the `[AWS ACCT ID] + [AWS ACCT SECRET KEY]`
service account.