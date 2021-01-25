# RAPIDS on AzureML

These are a few examples to get started on Azure. We'll look at how to set up the environment locally and on Azure to run the notebooks provided. 

Index
1. [Create an Azure Machine Learning Service Workspace](1. Create an Azure Machine Learning Service Workspace)
2. [RAPIDS MNMG example using dask-clouprovider](#2.-RAPIDS-MNMG-example-using-dask-clouprovider)
3. [RAPIDS Hyperparameter Optimization on AzureML](#)

# 1. Create an Azure Machine Learning Service Workspace

### 1(a) Resource Groups and Workspaces

An [Azure Machine Learning service workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace) will manage experiments and coordinate storage, databases and computing resources for machine learning applications. 

1. First create an [Azure subscription](https://azure.microsoft.com/en-us/free/) or access existing information from the [Azure portal](https://portal.azure.com/).

2. Next you will need to access a [Resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/overview#resource-groups) or create a new one in Azure portal: 

- Sign in to the Azure portal and navigate to Resource groups page by clicking on **Resource groups** in the portal:

![Portal](./img/Portal.JPG)

- Select one of the available Resource groups or create a new one by clicking on the **Add** button:

![ResourceGroup](./img/ResourceGroup.JPG)

- You can also select **+ Create a resource** in the upper-left corner of Azure portal and search for Resource group

Select a a *Subscription* with GPU resources, enter a name for the *Resource group* and select a *Region* with GPU resources. Check these pages for the [List](https://azure.microsoft.com/en-us/global-infrastructure/services/?products=machine-learning-service) of supported regions and [information](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu) on GPU optimized VM sizes. Pick a region that is closest to your location or contains your data. 

 3. Next we will create a Machine Learning service workspace: navigate to your Resource groups page and click on the **Add** button, this will take you to the [Azure Marketplace](https://azuremarketplace.microsoft.com/). Use the search bar to find **Machine Learning** or select **AI + Machine Learning** category on the left:  

![MarketPlace](./img/MarketPlace.JPG)

- Click on *Machine Learning* and this will direct you to the page below:

![MLWorkspace](./img/MLWorkspace.JPG)

- Enter a unique *Workspace Name* that indentifies your workspace, select your Azure *Subscription*, use an existing *Resource group* in your subscription and select a *Location* with adequate GPU quota.

After entering the information, select **Review + Create**. The deployment success message will appear and and you can view the new workspace by clicking on **Go to resource**. 

4. After creating the workspace, download the **config.json** file that includes information about workspace configuration. 

![Config](./img/Config.JPG)

This file will be used with [Azure Machine Learning SDK for Python](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py) in the notebook example to load the workspace and contains a dictionary list with key-values for:

* Workspace name
* Azure region
* Subscription id
* Resource group


# 2. RAPIDS MNMG example using dask-clouprovider

This [Azure MNMG notebook](#) will use [dask-cloudprovider](#) to run a multi-node muli-GPU example on Azure. For this, we will make use of [AzureVMCluster](#) function to set-up a cluster and run an example.


## 2(a) Set up environment on local computer

We recommend using RAPIDS docker image on your local system and using the same image in the notebook so that the libraries can match accurately. You can achieve this using conda environments for RAPIDS too.

In the example notebook we are using `rapidsai/rapidsai-core:cuda10.2-runtime-ubuntu18.04-py3.8` docker image, to pull and run this use the following command. The `-v` flag sets the volume you'd like to mount on the docker container. This way, the changes you make within the docker container are present on your local system to. Make sure to change `local/path` to the path which contains this repository.

    `docker run --runtime nvidia --rm -it -p 8888:8888 -p 8787:8787 -v /local/path:/docker/path rapidsai/rapidsai-core:cuda10.2-runtime-ubuntu18.04-py3.8`


## 2(b) Setup Azure environment

We need to setup a Virtual Network and Security Group to run this example.  You can use either the command line or the Azure Portal to set these up. 

Below, we'll be looking at how you can use command line to set it up. These commands need to be executed within the docker container.

Note: Be sure to set up all the resources in the same region

1. To setup the azure authentication, run `az login`

2. You can make use of the resoruce group you've set up earlier.

3. To create a virtual network -  `az network vnet create -g <resource group name> --location <location -n <vnet name> --address-prefix 10.0.0.0/16 --subnet-name <subnet name> --subnet-prefix 10.0.0.0/24`

4. We can now set up the Security group and add a rule for the dask cloud provider run.

```
az network nsg create -g <resource group name> --name <security group name> --location <region>
az network nsg rule create -g <resource group name> --nsg-name <security group name> -n MyNsgRuleWithAsg \
      --priority 500 --source-address-prefixes Internet --destination-port-ranges 8786 8787 \
      --destination-address-prefixes '*' --access Allow --protocol Tcp --description "Allow Internet to Dask on ports 8786,8787."
```

For more details, visit [Microsoft Azure - dask cloud provider](https://cloudprovider.dask.org/en/latest/azure.html#overview)

5. Once you have set up the resources, start a jupyter notebook on the docker container using the following command

```
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.token=''
```

6. Navigate to the notebook `azure_mnmg.ipynb` under `azure/notebooks`

7. Update the notebook with the names of resources appropriately and run it.

# 3. RAPIDS Hyperparameter Optimization on AzureML

This example will walk you through how to launch RAPIDS-accelerated hyperparameter optimization jobs on Microsoft Azure ML. Azure ML will train and evaluate models with many different variations of key parameters in order to find the combination that yields the highest accuracy. You'll start by launching a Jupyter notebook locally, which will launch all of the jobs and walk you through the process in more detail.

## 3(a) Set up environment on local computer

Install the [Azure Machine Learning Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py) (if you are running in your own environment. SDK is already installed in [Azure Notebooks](https://notebooks.azure.com/) or other Microsoft managed environments), this link includes additional instructions to [setup environment](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#local) on your local computer. 

After setting up a conda environment, clone the [clould-ml-examples repository](https://github.com/rapidsai/cloud-ml-examples.git) by running the following command in a `local_directory`: 

git clone https://github.com/rapidsai/cloud-ml-examples.git 

### 3(b) Notebooks and Scripts

Navigate to the azure/notebooks subdirectory. This will include hyperparameter optimizaiton notebooks: HPO-RAPIDS.ipynb and HPO-SKLearn.ipynb. Copy the **config.json** file (that you downloaded after creating a ML workspace) in the directory that contains these notebooks (azure/notebooks). You will load the information from this file in the `Initialize workspace` step of the notebook.

Activate the conda environment, where the Azure ML SDK was installed and launch the Jupyter Notebook server with the following command:

jupyter notebook

Open your web browser, navigate to http://localhost:8888/ and access `HPO-RAPIDS.ipynb` from your local machine. Follow the steps in the notebook for hyperparameter tuning with RAPIDS on GPUs.

