## Deploying an autoscaling Triton service utilizing a custom plugin for the RAPIDS cuML Forest Inference Library (FIL).
#### Adapted from Dong Meng's `gke-marketplace-app` in the [triton-inference-server](https://github.com/triton-inference-server/server/tree/master/deploy/gke-marketplace-app) repository. 

### Overview
This example will illustrate the workflow to deploy a horizontally scaled Triton service, with a non-trival custom
backend for accelerated forest model inference. It is assumed that you have an existing account with sufficient compute
and GPU quota, a correctly configured kubectl utility, and a [properly configured]() GCP bucket.

For simplicity, this process will be demonstrated using the Google Kubernetes Engine (GKE) platform. But should be
straight-forward to adapt to any desired Kubernetes environment with Istio and the stackdriver custom metrics adapter.

When referring to configuration parameters or elements that are `specific to your environment`, they will be represented
as linux-style environment variables ex. ${YOUR_CONFIG_PARAM}. 

Specific parameters used here include:
- YOUR_PROJECT_ID     : Your GCP [Project ID](https://support.google.com/googleapi/answer/7014113?hl=en).
- YOUR_GCR_PATH       : We assume a GCR model location of `gcr.io/${YOUR_PROJECT_ID}/${YOUR_GCR_PATH}/<..models..>`.
- YOUR_BUCKET_PATH    : Path to your personal, writable, GCP bucket.
- YOUR_CLUSTER_ID     : Name of your GKE cluster.
- YOUR_CLUSTER_ZONE   : Zone where the cluster will be deployed.

### Obtain the Triton FIL plugin, build the triton host container, and push to GCR
Note: as of this writing, the [FIL backend plugin](https://github.com/wphicks/triton_fil_backend) is considered 
experimental / preview-quality.
```shell
git clone git@github.com:wphicks/triton_fil_backend.git

docker build --tag gcr.io/${YOUR_PROJECT_ID}/${YOUR_GCR_PATH}/triton_fil --filename ops/Dockerfile .

docker push gcr.io/${YOUR_PROJECT_ID}/${YOUR_GCR_PATH}/triton_fil:latest
```

### Create a Triton model registry entry, or use the provided example
A sample XGBoost model, along with its .pbtext defintion is provided in ./model_repository. The layout structure and
requirements are defined in the [Triton server docs](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md),
and a brief introduction is also provided with the [FIL backend implementation](https://github.com/wphicks/triton_fil_backend#triton-inference-server-fil-backend).

This is what will be referenced for the purpose of this demo; however, feel free to add additional models, following the
same structure, and they will be included in subsequent steps.

```shell
gsutil cp -r ./model_repository gs://${YOUR_BUCKET_PATH}/triton_example/
gsutil ls gs://${YOUR_BUCKET_PATH}/triton_example/
 gs://${YOUR_BUCKET_PATH}/triton/
 gs://${YOUR_BUCKET_PATH}/triton/model_repository/
```

### Configure GKE cluster
This step is equivalent to the triton-inference-server sample, we just need to create a cluster that will host our
Triton service, and has a GPU node pool with enough nodes to illustrate horizontal scaling.

1. CLI Workflow:
```shell
gcloud beta container clusters create ${YOUR_CLUSTER_ID} \
--addons=HorizontalPodAutoscaling,HttpLoadBalancing,Istio \
--machine-type=n1-standard-8 \
--node-locations=${YOUR_CLUSTER_ZONE} \
--subnetwork=default \
--scopes cloud-platform \
--num-nodes 1

# add GPU node pools, user can modify number of node based on workloads
gcloud container node-pools create accel \
  --project ${YOUR_PROJECT_ID} \
  --zone ${YOUR_CLUSTER_ZONE} \
  --cluster ${YOUR_CLUSTER_ID} \
  --num-nodes 2 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --enable-autoscaling --min-nodes 2 --max-nodes 3 \
  --machine-type n1-standard-4 \
  --disk-size=100 \
  --scopes cloud-platform \
  --verbosity error

# so that you can run kubectl locally to the cluster
gcloud container clusters get-credentials ${YOUR_CLUSTER_ID} --project ${YOUR_PROJECT_ID} --zone ${YOUR_CLUSTER_ZONE}  

# deploy NVIDIA device plugin for GKE to prepare GPU nodes for driver install
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# make sure you can run kubectl locally to access the cluster
kubectl create clusterrolebinding cluster-admin-binding --clusterrole cluster-admin --user "$(gcloud config get-value account)"

# enable stackdriver custom metrics adaptor
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/deploy/production/adapter.yaml
```