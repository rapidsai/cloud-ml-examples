# [Detailed Guide to use Dask on Azure Kubernetes Service (AKS)](#anchor-start)

For all the next steps, we will be using the [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli), however the same can be achieved through the [Azure Portal](https://portal.azure.com/#home). 

### [Step 0: Install and authenticate with Azure CLI](#anchor-install-azurecli)
- Install the `az` cli using 
    ```
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    ````
    on the computer from where you will be running these examples from. You can remove the `sudo` if running inside a Docker container. 

- Once `az` is installed, make sure you configure the local `az` cli to work with your Azure credentials, run `az login` and authenticate from Microsoft's website.

- For more details follow the steps [Here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli). 


### [Step 1: Install Kubectl](#anchor-install-kubectl)

- Install `kubectl` to access your cluster from the local machine from the following link : [https://kubernetes.io/docs/tasks/tools/] depending on your operating system. 


### [Step 2: Set some environment variables](#anchor-set-env-variables)
We will set some environment variables beforehand, namely to help us deploy some resources quickly.  We will continue using common values for the rest of the deployments. 
```bash
REGION_NAME=<your preferred location>
RESOURCE_GROUP=<your preferred resource group name>
SUBNET_NAME=<optional subnet name>
VNET_NAME=<optional vnet name>
VERSION=$(az aks get-versions --location $REGION_NAME --query \
                'orchestrators[?!isPreview] | [-1].orchestratorVersion' \--output tsv)
AKS_CLUSTER_NAME=<your cluster name>
VM_SIZE=Standard_NC12s_v3 # or any other VM size. We use VMs with GPU
```

- We would need a resource group. You can create a resource group in Azure `az group create  --name $RESOURCE_GROUP --location $REGION_NAME`, or use an existing one.

- Secondly, we get the latest non preview Kubernetes version for the specific region and store it in an env variable `VERSION`. At the time of writing this article, the latest version is 1.20.5. Optionally you can directly set the Kubernetes version in the environment variable `VERSION`.

- **NOTE 1**: There are two network modes to choose from when deploying an AKS cluster. The default one is *Kubenet networking* which we will use here. 

- **NOTE 2**: Depending on your account limitations, the number and type of VMs that you can spin up may vary. Also there may be zone limitations. Make sure you spin up VMs with GPUs: NVIDIA Pascal™ or better with compute capability 6.0+. To give some examples of types of VMs you can use, the Azure [NC series](https://docs.microsoft.com/en-us/azure/virtual-machines/nc-series) / [NCv3 series](https://docs.microsoft.com/en-us/azure/virtual-machines/ncv3-series) VMs provide single or multi-gpu capabilities. In this setup guide for Kubernetes, we are using `Standard_NC12s_v3` VMs which have 2 NVIDIA V100 GPUs each.


### [Step 3: Create the cluster and get Kubernetes credentials](#anchor-create-aks-cluster)

Once you verify that you are allowed to use the necessary VM sizes in your preferred location, now its time to create a managed kubernetes cluster, namely an AKS cluster. The process is pretty simple. Also, after you successfully deploy a cluster with a node-pool of some nodes, you will be able to run workers as pods on the kubernetes cluster using [dask-kubernetes](https://github.com/dask/dask-kubernetes).

- Let's first run the following command to create a AKS cluster using the latest kubernetes version. It will take a few minutes before it completes. Grab a coffee :coffee: :coffee: .
    ```bash
    az aks create \
        --resource-group $RESOURCE_GROUP \
        --name $AKS_CLUSTER_NAME \
        --node-count 2 \
        --location $REGION_NAME \
        --kubernetes-version $VERSION \
        --node-vm-size $VM_SIZE \
        --generate-ssh-keys
    ```
- Once the cluster is created successfully, let's get the credentials for your AKS cluster to access it from your machine.
    ```bash
    az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER_NAME
    ```
- Check whether you are able to access the nodes: 
    ```bash
    kubectl get nodes

    NAME                                STATUS   ROLES   AGE     VERSION
    aks-nodepool1-98672075-vmss000000   Ready    agent   4m56s   v1.20.5
    aks-nodepool1-98672075-vmss000001   Ready    agent   4m12s   v1.20.5
    ```

### [Step 3: Set up the AKS cluster to use GPUs for our workload](#anchor-setup-gpu)
Once you have an AKS cluster up and running with nodes which have GPU capabilities, you need to install the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) which allows allocation of GPUs to pods. 
- First create a namespace using: 
    ```
    kubectl create namespace gpu-resources
    ```
- Create a file named *nvidia-device-plugin-ds.yaml* and paste the following manifest. This instruction set is taken from [Microsoft's official instructions](https://docs.microsoft.com/en-us/azure/aks/gpu-cluster). You can follow that as well. 
    ```yaml
    apiVersion: apps/v1
    kind: DaemonSet
    metadata:
    name: nvidia-device-plugin-daemonset
    namespace: gpu-resources
    spec:
    selector:
        matchLabels:
        name: nvidia-device-plugin-ds
    updateStrategy:
        type: RollingUpdate
    template:
        metadata:
        # Mark this pod as a critical add-on; when enabled, the critical add-on scheduler
        # reserves resources for critical add-on pods so that they can be rescheduled after
        # a failure.  This annotation works in tandem with the toleration below.
        annotations:
            scheduler.alpha.kubernetes.io/critical-pod: ""
        labels:
            name: nvidia-device-plugin-ds
        spec:
        tolerations:
        # Allow this pod to be rescheduled while the node is in "critical add-ons only" mode.
        # This, along with the annotation above marks this pod as a critical add-on.
        - key: CriticalAddonsOnly
            operator: Exists
        - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
        containers:
        - image: mcr.microsoft.com/oss/nvidia/k8s-device-plugin:1.11
            name: nvidia-device-plugin-ctr
            securityContext:
            allowPrivilegeEscalation: false
            capabilities:
                drop: ["ALL"]
            volumeMounts:
            - name: device-plugin
                mountPath: /var/lib/kubelet/device-plugins
        volumes:
            - name: device-plugin
            hostPath:
                path: /var/lib/kubelet/device-plugins
    ```
    Finally apply the NVIDIA Device plugin so that the pods can see the GPU :
    ```bash
    kubectl apply -f nvidia-device-plugin-ds.yml
    ```

### [Step 4: Create Azure Container Registry for pulling and pushing worker and scheduler docker images](#anchor-setup-azure-container-repository)
We will also need a container registry for the pod images. Here, we will use the container repository provided by Azure (ACR). 

- Create an Azure container repository env variable which will be useful later.
    ```bash
    ACR_NAME=<your repo name>
    ```
- Create an Azure Container Registry in the same region and under the same resource group with the Standard SKU with the following: 
    ```bash
    az acr create \
        --resource-group $RESOURCE_GROUP \
        --location $REGION_NAME \
        --name $ACR_NAME \
        --sku Standard
    ```

### [Step 5: Authenticate AKS to pull images from ACR using secrets](#anchor-setup-aks-acr-authentication)
We need to authenticate AKS to pull the images from ACR. For this purpose we will pass a secret in the pod configurations. We need to perform the following steps for that

- Admin enable ACR 
    ``` 
    az acr update -n $ACR_NAME --admin-enabled true
    ```
- Get the username and password using of ACR using 
    ```
    >> az acr credential show --name $ACR_NAME

        {
    "passwords": [
        {
        "name": "password",
        "value": "<some password 2>"
        },
        {
        "name": "password2",
        "value": "<some password 2>"
        }
    ],
    "username": "<your user name>"
    }
    ```
    Note down the password and the username. Any of the two passwords will work.

- Docker login and create a secret with `docker login`
    ```bash
    docker login $ACR_NAME.azurecr.io
    kubectl create secret docker-registry aks-secret --docker-server=$ACR_NAME.azurecr.io \
        --docker-username=$ACR_NAME --docker-password=<passwd> --docker-email=<any-email>
    ```
- **IMPORTANT:** The password/credentials for ACR expires every 3 hours. Make sure you renew them from the Azure portal if you want AKS to pull the images during pod creation. Then update the docker secret accordingly before scaling up or starting the cluster. 

- And then pass the secret to a pod manifest (all pod specification `yaml` files are in `./podspecs` directory, replace necessary details in those files) like the following:
    ```yaml
    kind: Pod
    spec:
    restartPolicy: Never
    containers:
        - image: <user>.azurecr.io/<image-path>
        imagePullPolicy: IfNotPresent
        name: dask-scheduler
        resources:
            limits:
            cpu: "4"
            memory: 25G
            requests:
            cpu: "4"
            memory: 25G
    imagePullSecrets:
        - name: "aks-secret"
    ```


### [Step 6: Build and push the pod images to ACR](#anchor-build-podimages)
Now we build and push the pod images to ACR. We have a `Dockerfile` in the current directory. From the current directory do the following:

```bash
docker build -t $ACR_NAME.azurecr.io/aks-mnmg/dask-unified:21.06 .
docker push $ACR_NAME.azurecr.io/aks-mnmg/dask-unified:21.06
```



### [Step 7: Install dask-kubernetes python library if not already present](#anchor-install-daskcloudprovider)
Install [dask-kubernetes](https://kubernetes.dask.org/en/latest/) if not already installed.

