## [Databricks-Requirements]
#### The following are configuration requirements for both publishing to the Databrick's tracking server, as well as initializing remote jobs.
1. Set environment variables expected by MLFlow, which define your Databricks cluster.
    1. **NOTE**: If you make any changes to the project repo, they will not be reflected in your mlflow deployment until
    they are committed to the working branch.
    1. **Note**: While the MLFLow Python API interface should allow these values to be set programatically, that workflow
    does not appear to work; as of version 1.8.0, `set_tracking_uri` and `set_experiment` do not produce the expected behavior when targeting Databricks. You will likely see raised errors claiming that active and selected experiments do not match, and/or spurrious authentication errors.
    1. ```bash
        $ export MLFLOW_EXPERIMENT_NAME=/Users/[YOUR USER NAME]/[EXPERIMENT NAME]
        $ export MLFLOW_TRACKING_URI=databricks
        $ export DATABRICKS_HOST=https://[CLUSTER_ID].cloud.databricks.com
        $ export DATABRICKS_TOKEN="[ACCESS TOKEN]"
       ```
       1. Here, ACCESS TOKEN can be created utilizing Databrick's
       [Token Creation Process](https://docs.databricks.com/dev-tools/api/latest/authentication.html#:~:text=Generate%20a%20personal%20access%20token,-This%20section%20describes&text=in%20the%20upper%20right%20corner,the%20Generate%20New%20Token%20button.).
       2. See the [Databricks documentation](https://docs.databricks.com/applications/mlflow/access-hosted-tracking-server.html) for additional information.

### Train Locally and Publish to Local or Databrick's MLFlow Tracking Server.
**NOTE**: Publishing to a local 'mlruns' directory does not support much of the MLFlow 
[Client or Tracking APIs](https://www.mlflow.org/docs/latest/tracking.html). 
1. Create a new conda environment, and configure your [databricks-cli](https://docs.databricks.com/dev-tools/cli/index.html) client.
    1. `$ conda create -f envs/conda.yaml`
    1. `$ databricks configure`
1. Train the model
    1. `$ cd databricks-mlflow-hyperopt`
    1. The example project is described in [MLProject](https://www.mlflow.org/docs/latest/projects.html) file.
        1. This can be edited to allow additional command line variables, specify conda environments, and training
        parameters (see link for additional information).
    1. Publish to local tracking server
        1. Here we instruct mlflow to run our training routine locally, and publish the result's to the local file system.
        1. ```shell script
            mlflow run file:///$PWD -b local\
                                    -P conda-env=$PWD/envs/conda.yaml\
                                    -P fpath=$PWD/airline_100000.orc
           ````
    1. Publish to Databrick's tracking server
        1. Here, we use mlflow to run our training routine locally, publish the resulting metrics to our configured
        Databricks account, and save our RAPIDS model.
        1. Export the [required environment](#databricks-requirements) variables for MLFlow
        1. ```shell script
            mlflow run file:///$PWD -b local\
                                    -P conda-env=$PWD/envs/conda.yaml\
                                    -P fpath=$PWD/airline_100000.orc
           ```
1. Deploy your model
    1. Locate the model's 'Full Path' identity. 
        1. Local storage
            1. `$ mlflow ui`
            1. Locate the model path using the mlflow ui at localhost:5000
        1. Databricks
            1. Locate your saved experiment in the Databricks tracking UI at: `/Users/[YOUR USER NAME]/[EXPERIMENT NAME]`
            1. Ex. `dbfs:/databricks/mlflow/[EXPERIMENT ID #]/[EXPERIMENT RUN HASH]/artifacts/`
    1. Select the successful run and find the 'Full Path' element
        1. ![Example 1](imgs/example.png)
    1. Deploy your model
        1. `mlflow models serve -m [PATH_TO_MODEL] -p 55755`
        1. **Note**: If you have not defined the environment variables described above, this will fail to pull your model
        from Databricks.
    1. From Databricks MLFlow UI
        1. Locate your saved experiment in the Databricks tracking UI at: `/Users/[YOUR USER NAME]/[EXPERIMENT NAME]`
        1. Ex. `dbfs:/databricks/mlflow/[EXPERIMENT ID #]/[EXPERIMENT RUN HASH]/artifacts/`
        1. `mlflow models serve -m [PATH_TO_MODEL] -p 55755` 
1. Query the deployed model with test data `test_call.sh` example script.
    1. `bash test_call.sh`
       
## Deploy Models Using MLFlow with Hyperopt and RAPIDS -1.13, on Databricks.
#### Currently, this approach does not support SparkTrials integration with Hyperopt.
1. The example `train.py` file supports two simple path specifications: local, and http: prefix.
    1. Training with Databrick's local file system
        1. Upload your data and conda environment spec to dbfs
    1. Training with externally accessible data (ex. http)
        1. Pass the URL of your dataset to the example training code: `http://your-endpoint.com/datafile.orc` 
1. Define a cluster configuration, or use the [sample provided](databricks/training_cluster.json)
1. Initiate Databricks job
    1. ```shell script
        mlflow run file:///$PWD -b databricks\
                                --backend-config=./databricks/training_cluster.json\
                                -P conda-env=[CONDA SPEC PATH OR URL]\
                                -P fpath=[DATH PATH]
       ``` 