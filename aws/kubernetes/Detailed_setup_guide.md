# [Detailed Guide to use Dask on AWS Elastic Kubernetes Service (EKS)](#anchor-start)

For all the next steps, we will be using the following two command line interfaces:
1. [AWS CLI](https://aws.amazon.com/cli/) - the general cli to manage AWS resources, perform authentication, get cluster credentials etc.
2. [eksctl](https://eksctl.io/) - the specialized cli to manage EKS clusters in AWS. We will use this to provision the Kubernetes cluster and associated resources using simple `yaml` files. 

**NOTE:** The rest of the instructions assume that you as a user have all the necessary IAM permissions in AWS to spin up an EKS cluster. However, if that is not the case, the following page, [How Amazon EKS works with IAM](https://docs.aws.amazon.com/eks/latest/userguide/security_iam_service-with-iam.html) in EKS official documentation provides a great starting point to understand what is necessary. Your role would ideally need some specific policies attached to it to create EC2 instances, manage EKS resources, autoscaling, logging, etc. 

Additionally, [terraform for EKS](https://github.com/terraform-aws-modules/terraform-aws-eks/) also has a [compact documentation](https://github.com/terraform-aws-modules/terraform-aws-eks/blob/master/docs/iam-permissions.md) for the minimum IAM requirements to create an EKS cluster that can be used as a reference. Please consult with the AWS administrator of your account to figure out the permissions.

Once you have obtained the permissions necessary, lets follow the steps to set up a Kubernetes cluster in AWS.

### [Step 1: Install and authenticate with AWS CLI and eksctl](#anchor-install-awscli)
- Install the `aws` cli from the instructions in : https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html 

- Once `aws` cli is installed, make sure you configure the local `aws` cli to work with your AWS credentials. Run `aws configure` and follow the steps [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html) to provide the AWS Access Key ID, AWS Secret Access Key, Default region name etc. 

- Next, install `eksctl` from https://eksctl.io/introduction/#installation. `eksctl` will use the `aws` credentials automatically, therefore you won't have to authenticate it explicitly.

### [Step 2: Install Kubectl](#anchor-install-kubectl)

- Install `kubectl` to access your cluster from the local machine from the following link https://kubernetes.io/docs/tasks/tools/ depending on your operating system. 


### [Step 3: Set some environment variables](#anchor-set-env-variables)
We will set some environment variables beforehand, namely to help us deploy some resources quickly.  We will continue using common values for the rest of the deployments. 
```bash
REGION_NAME=<your preferred location>
EKS_CLUSTER_NAME=<your cluster name>
INSTANCE_TYPE=g4dn.12xlarge # or any other VM size. We use VMs with GPU
```

- **NOTE:**: Depending on your account limitations, the number and type of VMs that you can spin up may vary. Also there may be zone limitations. Make sure you spin up VMs with GPUs: NVIDIA Pascal™ or better with compute capability 6.0+. To give some examples of types of VMs you can use, the AWS [P3 Instances](https://aws.amazon.com/ec2/instance-types/p3/) with V100 GPUs, AWS [G4 Instances](https://aws.amazon.com/ec2/instance-types/g4/) with T4 GPUs can be used. All these VM types provide single or multi-gpu capabilities. In this setup guide for Kubernetes, we are using `g4dn.12xlarge` VMs which have 4 NVIDIA T4 GPUs each.


### [Step 4: Create the cluster and get Kubernetes credentials](#anchor-create-eks-cluster)

Once you verify that you are allowed to use the necessary VM types in your preferred location, now it's time to create a managed Kubernetes cluster, namely an EKS cluster. The process is pretty simple with `eksctl`. After you successfully deploy a cluster with a node-group of some nodes, you will be able to run workers as pods on the Kubernetes cluster using [dask-kubernetes](https://github.com/dask/dask-kubernetes).

- Lets first create a minimal yaml configuration file with name `eksctl_config.yaml` with the following in it: 
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

    **NOTE:** TO use RAPIDS, you must use a GPU enabled instance in AWS. 

- With the `yaml` file, creating the EKS cluster is as simple as running the following command: 
    ```bash
    eksctl create cluster -f eksctl_config.yaml
    ```
    This will take a few minutes before it completes. Grab a coffee :coffee: :coffee: . For more `yaml` configurations, you can refer to [eksctl documentation](https://eksctl.io/usage/schema/).

- You can check the whether the cluster is successfully created with following:
    ```bash
    eksctl get cluster --name $EKS_CLUSTER_NAME --region $REGION_NAME
    ```
    **NOTE:** For more `eksctl` related references on how to delete or modify the cluster, visit https://docs.aws.amazon.com/eks/latest/userguide/getting-started-eksctl.html.

- Once the cluster is created successfully, let's get the credentials for your EKS cluster to access it from your machine. 
    ```bash
    aws eks update-kubeconfig --name $EKS_CLUSTER_NAME  --region $REGION_NAME
    ```
- Check whether you are able to access the nodes: 
    ```bash
    kubectl get nodes

    NAME                            STATUS     ROLES    AGE   VERSION
    ip-172-31-12-100.ec2.internal   NotReady   <none>   10m   v1.20.4-eks-6b7464
    ip-172-31-34-168.ec2.internal   NotReady   <none>   10m   v1.20.4-eks-6b7464
    ```

### [Step 5: Set up the EKS cluster to use GPUs for our workload](#anchor-setup-gpu)
The good thing about using `eksctl` is that we can simply use a GPU compatible VM instance type with EKS and the AWS AMI resolvers will automatically select the correct EKS optimized accelerated AMI instance. Subsequently, `eksctl` will install the NVIDIA Kubernetes device plugin automatically ([reference](https://eksctl.io/usage/gpu-support/)) in the VMs. Therefore we do not have to do anything additional in this step.

### [Step 6: Install dask-kubernetes python library if not already present](#anchor-install-daskcloudprovider)
Install [dask-kubernetes](https://kubernetes.dask.org/en/latest/) if not already installed.
    ```
    pip install dask-kubernetes
    ```
