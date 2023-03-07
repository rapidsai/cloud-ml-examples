## Deploying a Triton service that uses the RAPIDS cuML Forest Inference Library (FIL) backend.

**NOTE:** For steps to setup a horizontally autoscalable Triton Service in EKS refer to the instructions in [Detailed_HPA_Setup](./Detailed_HPA_Setup.md).

### Overview

This example will illustrate the workflow to deploy a [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), with the [cuML Forest Inference Library (FIL)  backend](https://github.com/triton-inference-server/fil_backend) on [Amazon Elastic Kubernetes Service (EKS)](https://aws.amazon.com/eks). [FIL](https://docs.rapids.ai/api/cuml/stable/api.html?highlight=forestinference#cuml.ForestInference) allows GPU accelerated inference for random forest and boosted decision tree models, and the FIL backend is right now available as a fully integrated part of Triton.

It is assumed that you have an existing account with sufficient compute and GPU quota, a correctly configured `kubectl` utility, and a properly configured and accessible AWS S3 bucket.

**NOTE:** When referring to configuration parameters or elements that are `specific to your environment`, they will be represented as linux-style environment variables ex. ${YOUR_CONFIG_PARAM}.

 
Specific parameters used here include:

- REGION_NAME= < your preferred location >
- EKS_CLUSTER_NAME= < your cluster name >
- INSTANCE_TYPE= g4dn.12xlarge # or any other VM size with GPUs.
- AWS_ACCOUNT_ID= < select your ACCOUNT ID >
- REPOSITORY_NAME= titon_fil_backend # < name of the custom image we are going to build for triton backend, you can change this to your preference >
- S3_BUCKET_PATH= < s3 bucket name (existing or to be created) to host the models >

  
---

### Step 1: Prerequisites

- AWS account with at least the following permissions mentioned in https://eksctl.io/usage/minimum-iam-policies/ along with permissions to create, pull and push to AWS ECR repositories.
- System Softwares
    -  `docker`
    -  `aws-cli` , `eksctl`
    -  `helm` , `kubectl` , `istioctl`
- Python Libraries
```shell
pip install nvidia-pyindex
pip install tritonclient[all]
```
- Login credentials for aws-cli and kubectl configured.

**Note:** The rest of the demo uses `aws-cli`  [version 2](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html). Hence some commands will be a little different from `aws-cli` version 1. We will also be using `helm` 3 that does not need `tiller`.

  

### Step 2: Obtain the Triton Inference Server official NVIDIA image (and optionally build a custom one)

The [FIL backend plugin](https://github.com/triton-inference-server/fil_backend) is natively supported in the official [Triton Inference Server](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver) image available in the [NVIDIA NGC Catalog](https://ngc.nvidia.com/catalog). The images in this catalogue are the preferred way of deploying the FIL backend. For this example, we will use the most current image: `nvcr.io/nvidia/tritonserver:21.06.1-py3`. 

**You should skip to [**Step 3**](#Step-3:-Create-some-Triton-model-repository-entries,-or-use-the-provided-examples) if you are using the official NVIDIA image.**

For informational purpose, we will also show how to create a custom image with FIL backend below. 

#### Step 2.a: Create an ECR repository

```shell
aws ecr get-login-password --region ${REGION_NAME} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION_NAME}.amazonaws.com
aws ecr create-repository \
    --repository-name ${REPOSITORY_NAME} \
    --image-scanning-configuration scanOnPush=true \
    --region ${REGION_NAME}
```

#### Step 2.b: Create and push the image to the ECR repository

**On a different terminal** do the following:

```shell
git clone https://github.com/triton-inference-server/fil_backend.git
cd fil_backend
docker build --tag ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION_NAME}.amazonaws.com/${REPOSITORY_NAME} -f ops/Dockerfile .
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION_NAME}.amazonaws.com/${REPOSITORY_NAME}:latest
```


### Step 3: Create some Triton model repository entries, or use the provided examples

A set of sample models, along with their `.pbtext` definition are provided in `./model_repository`. The layout structure and requirements are defined in the [Triton server docs](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md),
and a brief introduction is also provided with the [FIL backend implementation](https://github.com/triton-inference-server/fil_backend/blob/main/README.md).

For the purpose of this demo we will use the models in `.model_repository` directory; however, feel free to add additional models. Make sure they follow the same structure in the docs. Here we assume the AWS `S3_BUCKET_PATH` bucket exists. You can create one in case it does not exist.

```shell
aws s3 cp --recursive ./model_repository s3://${S3_BUCKET_PATH}/model_repository/
aws s3 ls s3://${S3_BUCKET_PATH}/model_repository/
```

  
### Step 4: Configure an EKS cluster

We first need to create a Kubernetes cluster that will host our
Triton service, and has a GPU node pool with enough nodes and GPUs to illustrate horizontal scaling.

Lets first create a minimal yaml configuration file with name `eksctl_config.yaml` with the following in it

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
    name: <name set in EKS_CLUSTER_NAME>
    region: <region set in REGION_NAME>
    version: "1.20"

managedNodeGroups:
- name: nodegroup-managed-gpu-dev
    minSize: 0
    desiredCapacity: 2
    maxSize: 4
    instanceType: <instance type set in INSTANCE_TYPE>
    ssh:
    allow: true
```

To make things simple, we will use a [managed node group](https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html) to let EKS perform the provisioning and lifecycle management on our behalf. We are also allowing `ssh` access into the nodes for debugging. By default this will use the `~/.ssh/id_rsa.pub` file in your local machine.

With the `yaml` file, creating the EKS cluster is as simple as running the following command:

```shell
eksctl create cluster -f eksctl_config.yaml
```

This will take a few minutes before it completes. Grab a coffee :coffee: :coffee: . For more `yaml` configurations, you can refer to [eksctl documentation](https://eksctl.io/usage/schema/).

You can check whether the cluster is successfully created with following:

```shell
eksctl get cluster --name $EKS_CLUSTER_NAME --region $REGION_NAME
```

**NOTE:** For more `eksctl` related references on how to delete or modify the cluster, visit https://docs.aws.amazon.com/eks/latest/userguide/getting-started-eksctl.html.

Once the cluster is created successfully, let's get the credentials for your EKS cluster to access it from your machine.

```shell
aws eks update-kubeconfig --name $EKS_CLUSTER_NAME --region $REGION_NAME
```

Check whether you are able to access the nodes:
```shell
>> kubectl get nodes

NAME STATUS ROLES AGE VERSION
ip-172-31-12-100.ec2.internal NotReady <none> 10m v1.20.4-eks-6b7464
ip-172-31-34-168.ec2.internal NotReady <none> 10m v1.20.4-eks-6b7464
```
  

### Step 5: Setting up the EKS cluster to use GPUs for our workload

The good thing about using `eksctl` is that we can simply use a GPU compatible VM instance type with EKS and the AWS AMI resolvers will automatically select the correct EKS optimized accelerated AMI instance. Subsequently, `eksctl` will install the NVIDIA Kubernetes device plugin automatically ([reference](https://eksctl.io/usage/gpu-support/)) in the VMs. Therefore we do not have to do anything additional in this step.

  

### Step 6: To access models from AWS S3 and to fetch fil_backend image from ECR repository

To fetch the images from the ECR repository and to load the models from the AWS S3 we will use a config file and a secret. You need to convert the AWS credentials of your account in the base64 format and add it to the `./helm/charts/triton/values.yaml`, [similar to these instructions](https://github.com/triton-inference-server/server/tree/main/deploy/aws).

```shell
echo -n 'AWS_REGION' | base64
echo -n 'AWS_SECRET_KEY_ID' | base64
echo -n 'AWS_SECRET_ACCESS_KEY' | base64
echo -n 'AWS_SESSION_TOKEN' | base64
```

where `AWS_REGION`, `AWS_SECRET_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` can be obtained either from the environment variable or the aws credentials file. `helm` will populate the `helm/charts/triton/templates/secrets.yaml` appropriately during deployment.

  

Replace the following fields in the `helm/charts/triton/values.yml` file with the above details

```yaml
  image:
    imageName: nvcr.io/nvidia/tritonserver:21.06.1-py3 # < update this if you are using a custom image >
    pullPolicy: IfNotPresent
    modelRepositoryPath: s3://S3_BUCKET_PATH/model_repository
    numGpus: 1
    logVerboseLevel: 0
    allowGPUMetrics: True

  secret:
    region: < replace with base64 AWS_REGION >
    id: < replace with base64 AWS_SECRET_KEY_ID >
    key: < replace with base64 AWS_SECRET_ACCESS_KEY >
    session_token: < replace with base64 AWS_SESSION_TOKEN >
```

  
### Step 7: Install `istio` on your EKS cluster

Install the demo profile of `istio` by doing
    
```shell
istioctl install --set profile=demo
```

This will install the "Istio core", "Istiod", "Ingress gateways" and "Egress gateways" along with exposing all the necessary ports for gRPC and http connections for the ingress gateways.

Finally, enable sidecar injection to the triton pods in the default namespace:

```shell
kubectl label namespace default istio-injection=enabled
```

### Step 8: Install Triton service using helm.
```shell
helm install triton ./helm/charts/triton/ 
```
This will install a release named `triton`.

When finished, you can delete the triton deployment using
```shell
helm uninstall triton
```

### Step 9: Exploring inference
At this point, your triton inference cluster should be up and running or in process of coming up. Now we can submit some
test data to our running server. The process for doing this, assuming the default model, is illustrated in the jupyter
notebook [triton_inference.ipynb](./triton_inference.ipynb). 
