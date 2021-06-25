# RAPIDS on AWS 
This directory contains a few examples to get started on using RAPIDS on AWS. The sections in this README are the following:

1. Instructions for Running RAPIDS + SageMaker HPO
2. RAPIDS MNMG with Amazon Elastic Kubernetes Service (EKS) using Dask Kubernetes

## 1. Instructions for Running RAPIDS + SageMaker HPO 

0. Upload train/test data to S3 

   - We offer the dataset for this demo in a public bucket hosted in either the `us-east-1` or `us-west-2` regions:
   > https://s3.console.aws.amazon.com/s3/buckets/sagemaker-rapids-hpo-us-east-1/   
   > https://s3.console.aws.amazon.com/s3/buckets/sagemaker-rapids-hpo-us-west-2/
   
   
1. Create a SageMaker Notebook Instance

   - Sign in to the Amazon SageMaker console at 
   > https://console.aws.amazon.com/sagemaker/
   
   - Choose **Notebook Instances**, then choose 'Create notebook instance'.
   - Note that this notebook does not currently work in SageMaker Studio, due to lack of docker support. We plan to release a version with documentation on how to work around this limitation.
   
<img src='img/sagemaker_notebook_instance.png'>

2. On the Create notebook instance page, provide the following information (if a field is not mentioned, leave the default values):
   - For **Notebook instance name**, type a name for your notebook instance.
   - For **Instance type**, we recommend you choose a lightweight instance (e.g., ml.t2.medium) since the notebook instance will only be used to build the container and launch work.
   - For **IAM role**, choose Create a new role, then choose Create role.
   - For **Git repositories**, choose 'Clone a public Git repository to this notebook instance only' and add the cloud-ml-examples repository to the URL
   > https://github.com/rapidsai/cloud-ml-examples 
   - Choose 'Create notebook instance'. 
   
   - In a few minutes, Amazon SageMaker launches an ML compute instance — when its ready you should see several links appear in the Actions tab of the **Notebook Instances** section, click on **Open JupyerLab** to launch into the notebook.   
   > Note: If you see Pending to the right of the notebook instance in the Status column, your notebook is still being created. The status will change to InService when the notebook is ready for use.

3. Run Notebook
   - Once inside JupyterLab you should be able to navigate to the notebook in the root directory named **rapids_sagemaker_hpo.ipynb**


## 2. RAPIDS MNMG with Amazon Elastic Kubernetes Service (EKS) using Dask Kubernetes

For detailed instructions of setup and example notebooks to run RAPIDS with Amazon Elastic Kubernetes Service using Dask Kubernetes, navigate to the `kubernetes` subdirectory.

- Detailed instructions to set up RAPIDS with EKS using Dask Kubernetes is in the markdown file [Detailed_setup_guide.md](./kubernetes/Detailed_setup_guide.md) . Go through this before you try to run any other notebooks.
- Shorter example notebook using Dask + RAPIDS + XGBoost in [MNMG_XGBoost.ipynb](./kubernetes/MNMG_XGBoost.ipynb)
- Full example with performance sweeps over multiple algorithms and larger dataset in [Dask_cuML_Exploration_Full.ipynb](./kubernetes/Dask_cuML_Exploration_Full.ipynb)