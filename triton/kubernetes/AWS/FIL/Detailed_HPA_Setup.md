
## Deploying an autoscaling Triton service that uses a custom plugin for the RAPIDS cuML Forest Inference Library (FIL).

### Overview

This example will illustrate the workflow to deploy a [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), with the [cuML Forest Inference Library (FIL)  backend](https://github.com/triton-inference-server/fil_backend) on [Amazon Elastic Kubernetes Service (EKS)](https://aws.amazon.com/eks). [FIL](https://docs.rapids.ai/api/cuml/stable/api.html?highlight=forestinference#cuml.ForestInference) allows GPU accelerated inference for random forest and boosted decision tree models, and the FIL backend is right now available as a fully integrated part of Triton. We will use [Prometheus](https://prometheus.io/), [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter) and [dcgm-exporter](https://github.com/NVIDIA/gpu-monitoring-tools) (also known as NVIDIA GPU Monitoring Tools) to set up a Horizontal Pod Autoscaler (HPA) in EKS so that we can scale up our infrastructure based on inferencing load on the Triton Server.

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
    imageName: < update this with the image name you just pushed >
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

  

### Step 8: Configure Autoscaling

Autoscaling can be achieved in several tiers, pod autoscaling, node autoscaling etc. In this example we will set up a [Horizontal Pod Autoscaler (HPA)](https://docs.aws.amazon.com/eks/latest/userguide/horizontal-pod-autoscaler.html) that autoscales pods based on custom metrics obtained from a custom service [dcgm-exporter](https://github.com/NVIDIA/gpu-monitoring-tools) , also known as NVIDIA GPU Monitoring Tools. 
With increasing inferencing load in the FIL inferencing system, the GPU utilization will go up. As a result, the SLAs may be hampered. If we have a HPA in place, our cluster can then horizontally scale up to add more pods with GPU access to handle this additional load.

##### We will need to setup a few things in order for this setup to work:

-  [nvidia-device-plugin](https://github.com/NVIDIA/k8s-device-plugin): We already have the NVIDIA k8s device plugin installed automatically by AWS.
-  [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) - This collects the GPU metrics, store them and display in Grafana dashboard.
-  [prometheus-adapter](https://github.com/kubernetes-sigs/prometheus-adapter) - Collects metrics from the Prometheus server and exposes them to a custom k8s metrics api for use in pod autoscaling.
-  [dcgm-exporter](https://github.com/NVIDIA/gpu-monitoring-tools) - Esentially a daemonset with some services to reveal GPU metrics on each node.

  

#### Step 8.a: Install Prometheus related components

Installing the Prometheus stack is made easy with the help of `Helm` charts. We will first add the  Prometheus kube stack Helm chart to our local Helm installation. We then make some customizations before installing it. In addition, we will also install the prometheus-adapter stack.

Do the following to get the default configuration values of the kube-prometheus-stack chart in a `yaml` file to modify the settings.

```shell
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm inspect values prometheus-community/kube-prometheus-stack > ./kube-prometheus-stack.values.yaml
```

  

Next, we’ll need to edit the values file to change the port at which the Prometheus server service is available. In the `prometheus` instance section of the chart yaml file, change the service type from ClusterIP to NodePort. This will allow the Prometheus server to be accessible at your machine ip address at port 30090 as http://<machine-ip>:30090/ using `kubectl port-forward` if you are on a remote machine.

From:

```yaml
# Port to expose on each node
# Only used if service.type is 'NodePort'
#
nodePort: 30090

# Loadbalancer IP
# Only use if service.type is "loadbalancer"
loadBalancerIP: ""
loadBalancerSourceRanges: []
# Service type
#
type: ClusterIP
```

To:

```yaml
# Port to expose on each node
# Only used if service.type is 'NodePort'
#
nodePort: 30090

# Loadbalancer IP
# Only use if service.type is "loadbalancer"
loadBalancerIP: ""
loadBalancerSourceRanges: []
# Service type
#
type: NodePort
```

Also, modify the `prometheusSpec.serviceMonitorSelectorNilUsesHelmValues` settings to `false` below to allow Prometheus to scrape metrics from all namespaces :

```yaml
# If true, a nil or {} value for prometheus.prometheusSpec.serviceMonitorSelector will cause the
# prometheus resource to be created with selectors based on values in the helm deployment,
# which will also match the servicemonitors created
#
serviceMonitorSelectorNilUsesHelmValues: false
```

  

Lastly, we also add the following configMap to the section on `additionalScrapeConfigs` in the Helm chart. Make sure that in the `additionalScrapeConfigs.kubernetes_sd_configs.role.namespaces.names` you specify the namespace where the `dcgm-exporter` service will be deployed so that Prometheus can scrape metrics information from that service. Here we will deploy the dcgm exporter in the `default` namespace, so we keep the value `default`:

```yaml
# AdditionalScrapeConfigs allows specifying additional Prometheus scrape configurations. Scrape configurations
# are appended to the configurations generated by the Prometheus Operator. Job configurations must have the form
# as specified in the official Prometheus documentation:
# https://prometheus.io/docs/prometheus/latest/configuration/configuration/#scrape_config. As scrape configs are
# appended, the user is responsible to make sure it is valid. Note that using this feature may expose the possibility
# to break upgrades of Prometheus. It is advised to review Prometheus release notes to ensure that no incompatible
# scrape configs are going to break Prometheus after the upgrade.
#
# The scrape configuration example below will find master nodes, provided they have the name .*mst.*, relabel the
# port to 2379 and allow etcd scraping provided it is running on all Kubernetes master nodes
#
additionalScrapeConfigs:
- job_name: gpu-metrics
  scrape_interval: 1s
  metrics_path: /metrics
  scheme: http
  kubernetes_sd_configs:
  - role: endpoints
    namespaces:
      names:
      - default
      # - <the namespace where dcgm exporter will be installed. Needs to be in the same namespace.>
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_node_name]
    action: replace
    target_label: kubernetes_node
```

Next, we will finally install the Prometheus stack and the Prometheus adapter.

```shell
# Install Prometheus stack with the modifications
helm install prometheus-community/kube-prometheus-stack \
            --create-namespace --namespace prometheus \
            --generate-name \
            --values ./kube-prometheus-stack.values.yaml 

# Get the Prometheus Service
PROM_SERVICE=$(kubectl get svc -nprometheus -lapp=kube-prometheus-stack-prometheus -ojsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')

# Install Prometheus Adapter
helm install prometheus-adapter prometheus-community/prometheus-adapter \
--namespace prometheus \
--set rbac.create=true,prometheus.url=http://${PROM_SERVICE}.prometheus.svc.cluster.local,prometheus.port=9090
```


Once you install the above charts, in around 1-2minutes, you should be able to observe metrics from the custom endpoint

```shell
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq -r .
```


### Step 9: Deploy the dcgm exporter stack.

**Note 1:** It is mandatory that the Prometheus stack is deployed beforehand.

**Note 2:** The instructions to configure dcgm are modified from [official NVIDIA instructions](https://docs.nvidia.com/datacenter/cloud-native/kubernetes/dcgme2e.html#gpu-telemetry) .

The dcgm-exporter as of now may run into a few problems when deploying. Fortunately, we have some gotchas. We will have to make sure we customize our dcgm deployment to take care of those.

- Firstly, in the current version of dcgm-exporter, the GPU utilization metric `DCGM_FI_DEV_GPU_UTIL` is turned off by default, probably because it is resource intensive to collect (see https://github.com/NVIDIA/gpu-monitoring-tools/issues/143). We need to enable this flag to obtain GPU utilization metrics for autoscaling.
- Secondly, there is a liveness bug where the pods after deployment will consistently enter a CrashLoopBackoff status (https://github.com/NVIDIA/gpu-monitoring-tools/issues/120). To get around this, we need to modify the dcgm-exporter k8s config by taking details from [here](https://github.com/NVIDIA/gpu-monitoring-tools/issues/120#issuecomment-801574290) and customize them slightly.

To fix the first problem and re-enable the metric `DCGM_FI_DEV_GPU_UTIL`, we will create a custom docker image and use that image in the dcgm-exporter deployment. Then to mitigate the second issue, we will modify the helm chart of dcgm-exporter before deploying.

First let us create a docker image with the following Dockerfile to expose our required metrics:

```Dockerfile
FROM nvcr.io/nvidia/k8s/dcgm-exporter:latest
RUN sed -i -e '/^# DCGM_FI_DEV_GPU_UTIL.*/s/^#\ //' /etc/dcgm-exporter/default-counters.csv
ENTRYPOINT ["/usr/local/dcgm/dcgm-exporter-entrypoint.sh"]
```

**Note:** You can choose to expose any other metric of interest in the above Dockerfile as well.

Let's suppose, you build and push the image to [Dockerhub](https://hub.docker.com/) as `anirbandas/dcgm-exporter:latest`.

Next, clone the [NVIDIA gpu-monitoring-tools](https://github.com/NVIDIA/gpu-monitoring-tools) repository which contains the dcgm-exporter.

```shell
git clone https://github.com/NVIDIA/gpu-monitoring-tools
```

Then edit the `./gpu-monitoring-tools/deployment/dcgm-exporter/values.yaml` file

From :
```yaml
image:
  repository: nvcr.io/nvidia/k8s/dcgm-exporter
  pullPolicy: IfNotPresent
  # Image tag defaults to AppVersion, but you can use the tag key
  # for the image tag, e.g:
  tag: 2.1.8-2.4.0-rc.3-ubuntu18.04
  
# Comment the following line to stop profiling metrics from DCGM
arguments: ["-f", "/etc/dcgm-exporter/dcp-metrics-included.csv"]
```

To:
```yaml
image:
  repository: anirbandas/dcgm-explorer # or whereever you pushed your image
  pullPolicy: IfNotPresent
  # Image tag defaults to AppVersion, but you can use the tag key
  # for the image tag, e.g:
  tag: latest

# The following line will use the default metrics flags
arguments: ["-k"]
```

Also, do not forget to edit the `livenessProbe` and `readinessProbe` fields of the daemonset configuration at `./gpu-monitoring-tools/deployment/dcgm-exporter/templates/daemonset.yaml` to get rid of the `CrashLoopBackoff` error. We change the config as follows.

From :
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: {{ .Values.service.port }}
  initialDelaySeconds: 5
  periodSeconds: 5
readinessProbe:
  httpGet:
    path: /health
    port: {{ .Values.service.port }}
  initialDelaySeconds: 5
```

To:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: {{ .Values.service.port }}
  initialDelaySeconds: 20
  periodSeconds: 5
readinessProbe:
  httpGet:
    path: /health
    port: {{ .Values.service.port }}
  initialDelaySeconds: 20
```
  

You should then be able to deploy the dcgm-exporter as:

```shell
helm install dcgm-exporter gpu-monitoring-tools/deployment/dcgm-exporter/
```

To check whether the GPU metrics are available in the Prometheus custom metrics api endpoint:

```shell
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq -r . | grep DCGM_FI_DEV_GPU_UTIL
```

You want to be able to see something like:

```shell
"name": "namespaces/DCGM_FI_DEV_GPU_UTIL",
"name": "services/DCGM_FI_DEV_GPU_UTIL",
"name": "jobs.batch/DCGM_FI_DEV_GPU_UTIL",
"name": "pods/DCGM_FI_DEV_GPU_UTIL",
```

If you are upto this point, then you are golden. You can now pull the custom metrics from both Prometheus and Grafana dashboards.



### Step 10: Configure the HPA

Now that everything else is in place, and we have the GPU metrics available in the custom metrics server, we will configure our HPA before we deploy the Triton service.

Our `hpa` configuration file is located at [./helm/charts/triton/templates/hpa.yml](./helm/charts/triton/templates/hpa.yml), and it should look like the following before deployment.

```yaml
apiVersion: autoscaling/v2beta1
kind: HorizontalPodAutoscaler
metadata:
  name: triton-hpa
  namespace: {{ .Release.Namespace }}
  labels:
    app: triton-hpa
spec:
  minReplicas: {{ .Values.minReplicaCount }}
  maxReplicas: {{ .Values.maxReplicaCount }}
  metrics:
  - type: Object
    object:
      metricName: DCGM_FI_DEV_GPU_UTIL
      targetValue: {{ .Values.HPATargetAverageValue }}
      target:
        kind: Service
        name: {{ .Values.DCGMExporterService }}
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ template "triton-inference-server.name" . }}
```
  
Update `./helm/charts/triton/Values.yaml` with the correct dcgm-exporter service name as

```yaml
.
.
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

initReplicaCount: 1
minReplicaCount: 2
maxReplicaCount: 6
HPATargetAverageValue: 80
DCGMExporterService: dcgm-exporter # <replace with the DCGM exporter service name if different>

image:
.
.
```

  

### Step 11: Deploy the Triton Inferencing Server

Once everything else is configured, deploying triton service is as easy as the following:

```shell
helm install triton ./helm/charts/triton/
```

  

### Step 12: Check if the HPA is able to access the metrics

It is a good idea to check if the HPA is able to get the metrics from the metrics server. If there is an error the HPA would not autoscale your pods. 

```shell
kubectl describe hpa triton-hpa
```

If everything else is configured correctly, you will be able to see something similar to the following:

```shell
Name:                                                                         triton-hpa
Namespace:                                                                    default
Labels:                                                                       app=triton-hpa
                                                                              app.kubernetes.io/managed-by=Helm
Annotations:                                                                  meta.helm.sh/release-name: triton
                                                                              meta.helm.sh/release-namespace: default
CreationTimestamp:                                                            Wed, 07 Jul 2021 18:43:06 -0400
Reference:                                                                    Deployment/triton-inference-server
Metrics:                                                                      ( current / target )
  "DCGM_FI_DEV_GPU_UTIL" on Service/dcgm-exporter-1625697000 (target value):  0 / 800m
Min replicas:                                                                 2
Max replicas:                                                                 6
Deployment pods:                                                              2 current / 2 desired
Conditions:
  Type            Status  Reason               Message
  ----            ------  ------               -------
  AbleToScale     True    ScaleDownStabilized  recent recommendations were higher than current one, applying the highest recent recommendation
  ScalingActive   True    ValidMetricFound     the HPA was able to successfully calculate a replica count from Service metric DCGM_FI_DEV_GPU_UTIL
  ScalingLimited  False   DesiredWithinRange   the desired count is within the acceptable range
Events:
  Type    Reason             Age   From                       Message
  ----    ------             ----  ----                       -------
  Normal  SuccessfulRescale  3m3s  horizontal-pod-autoscaler  New size: 2; reason: Current number of replicas below Spec.MinReplicas
  ```
  

### Step 12: Exploring inference

At this point, your triton inference cluster should be up and running or in process of coming up. Now we can submit some test data to our running server. The process for doing this, assuming the default model, is illustrated in the jupyter notebook [triton_inference.ipynb](./triton_inference.ipynb). 

  

### Step 13. For cluster autoscaling you can follow instructions in https://github.com/awsdocs/amazon-eks-user-guide/blob/master/doc_source/cluster-autoscaler.md .



