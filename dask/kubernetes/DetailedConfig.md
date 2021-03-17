# [Detailed Google Kubernetes Engine (GKE) Guide](#anchor-start)
### Baseline

For all steps referring to the Google Cloud Platform (GCP) console window, components can be selected from the 'Huburger Button'
on the top left of the console.
![Hamburger Bars](./images/gcp_hamburger_bar.png)

## [Create a GKE Cluster](#anchor-create-cluster)
__Verify Adequate GPU Quotas__

Depending on your account limitations, you may have restricted numbers and types of GPUs that can be allocated within a given zone or region.

- Navigate to your [GCP Console](https://console.cloud.google.com/)
- Select "IAM & Admin" $\rightarrow$ "Quotas"  
    ![IAM & Admin](./images/gcp_iam_admin.png) 
    - Filter by the type of GPU you want to add ex ‘T4’, ‘V100’, etc...
    - Select __ALL QUOTAS__ under the __Details__ column
    - If you find that you’re not given the option to assign GPUs during cluster configuration, re-check that you have an allocated quota.

## [Configure the Cluster hardware](#anchor-configure-cluster)
Once you’ve verified that you can allocate the necessary hardware for your cluster, you’ll need to go through the process 
of creating a new cluster, and configuring node pools that will host your services, and run MLflow jobs.

__Allocate the Appropriate Hardware__

- Navigate to your [GCP Console](https://console.cloud.google.com/)
- Select __Kubernetes Engine__ $\rightarrow$ __Clusters__
- Create a new cluster, we'll call it 'rapids-dask'
- Create a node-pool: __gpu-pool__
    - This is a good practice that can help reduce overall costs, so that you are not running CPU only
    tasks on GPU enable nodes. GKE will automatically taint GPU nodes so that they will be unavailble for
    tasks that do not request a GPU.
    - __GPU Pool__   
        ![GCP Node Pool Configuration 3](./images/gcp_node_pools_3.png)  
        ![GCP Node Pool Configuration 4](./images/gcp_node_pools_5.png)  
- Click __Create__ and wait for your cluster to come up.

## [Configure Kubectl](#anchor-kubectl)
__Obtain Kubectl Cluster Credentials from GKE.__
- First, be sure that [Kubectl is installed](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- Once your cluster appears to be up, running, and reported green by GKE, we need to use __glcoud__ to configure kubectl
with the correct credentials.
   ```shell script
   gcloud container clusters get-credentials rapids-dask --region us-east1-c
   ```
   - Once this command completes, your kubektl's default configuration should be pointing to your GKE cluster instance,
   and able to interact with it.
       ```shell script
      kubectl config get-contexts
      
      CURRENT   NAME                 CLUSTER            AUTHINFO            NAMESPACE
      *         gke_[YOUR_CLUSTER]   gke_[YOUR_CLUSTER] gke_[YOUR_CLUSTER]  default
       ```
        
       ```shell script
       kubectl get all
       ```
__Create an Up to Date NVIDIA Driver Installer.__  
As of this writing, this step is necessary to ensure that a CUDA 11 compatible driver (450+) is installed on your worker nodes.

```shell script
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-nvidia-v450.yaml
```
