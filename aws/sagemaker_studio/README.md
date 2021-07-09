# Instructions for Running RAPIDS + SageMaker Studio

0. Upload train/test data to S3 

   - We offer a dataset for the HPO demo in a public bucket hosted in either the `us-east-1` or `us-west-2` regions:
   > https://s3.console.aws.amazon.com/s3/buckets/sagemaker-rapids-hpo-us-east-1/   
   > https://s3.console.aws.amazon.com/s3/buckets/sagemaker-rapids-hpo-us-west-2/

1. Publish a container to the Amazon Elastic Container Registry (ECR)

   - Because SageMaker Studio runs from a Docker container, you cannot create and upload a Docker container from a SageMaker Studio instance. However, in order to use a custom container with an Estimator, you must first publish it to the ECR. In order to work around this, we recommend running the studio_hpo_container_setup.ipynb notebook within a notebook instance. To create a notebook instance: 
   
   - Sign in to the Amazon SageMaker console at 
   > https://console.aws.amazon.com/sagemaker/
   
   - Choose **Notebook Instances**, then choose 'Create notebook instance'.
   
   - For **Notebook instance name**, type a name for your notebook instance.
   - For **Instance type**, we recommend you choose a lightweight instance (e.g., ml.t2.medium) since the notebook instance will only be used to build the container and launch work.
   - For **IAM role**, choose Create a new role, then choose Create role.
   - For **Git repositories**, choose 'Clone a public Git repository to this notebook instance only' and add the cloud-ml-examples repository to the URL
   > https://github.com/rapidsai/cloud-ml-examples 
   - Choose 'Create notebook instance'. 
   
   - In a few minutes, Amazon SageMaker launches an ML compute instance — when its ready you should see several links appear in the Actions tab of the **Notebook Instances** section, click on **Open JupyerLab** to launch into the notebook.   
   > Note: If you see Pending to the right of the notebook instance in the Status column, your notebook is still being created. The status will change to InService when the notebook is ready for use.
   - Once inside JupyterLab you should be able to navigate to the notebook in the **aws** directory named **studio_container_setup.ipynb**. Run all of its cells. 
   - After running, you may shut down the notebook instance. 
   
2. Create/open a SageMaker Studio session   

   - Choose **Amazon SageMaker Studio**, and set up a domain if one does not already exist in the region. See the Quick start procedure for details: 
   > https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html
   - Add a user to the SageMaker Studio Control Panel (if one does not already exist), and Open Studio to start a session.
   
3. Within the SageMaker Studio session, clone this repository

   - Click the Git icon on the far left of the screen (second button, below the folder icon), select Clone a Repository, and paste: 
   > https://github.com/rapidsai/cloud-ml-examples 
   
   - After cloning, you should see the directory **cloud-ml-examples** in your file browser. 

4. Run desired notebook

   - Within the root directory **cloud-ml-examples**, navigate to **aws/sagemaker_studio**, and open and run your desired notebook. 
