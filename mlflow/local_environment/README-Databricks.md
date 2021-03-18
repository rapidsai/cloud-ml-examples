### Utilizing Databricks' for MLFlow Tracking and Job Training

#### Assumptions and Naming Conventions
- All shell commands are assumed to be run within the `/cloud-ml-examples/mlflow/docker_environment` directory.

- There are a number of configuration parameters that will be specific to your _environment_ and _deployment_:
    - `DATBRICKS HOST` : URI of your Databricks service, will be of the form: `https://<cluster_id>.cloud.databricks.com` 
    - `DATABRICKS TOKEN` : Access token used to authenticate with your Databricks service account
        - [Token Creation Process](https://docs.databricks.com/dev-tools/api/latest/authentication.html#:~:text=Generate%20a%20personal%20access%20token,-This%20section%20describes&text=in%20the%20upper%20right%20corner,the%20Generate%20New%20Token%20button.).
        See the [Databricks documentation](https://docs.databricks.com/applications/mlflow/access-hosted-tracking-server.html) for additional information.
    - `EXPERIMENT NAME` : MLflow experiment name, which will be used to register with the tracking server and indicates
    the subdirectory of your user environment to write to. 
    - `YOUR USER NAME` : Databricks username 
       
#### Databricks-Requirements
1. Set environment variables expected by MLFlow, which indicate the location of your Databricks cluster and how to connect to it.
    - **Note**: If you make any changes to the project repo, they will not be reflected in your mlflow deployment until
    they are committed to the working branch.
    - **Note**: While the MLFLow Python API interface should allow these values to be set programatically, that workflow
    does not appear to work; as of version 1.8.0, `set_tracking_uri` and `set_experiment` do not produce the expected behavior when targeting Databricks. You will likely see raised errors claiming that active and selected experiments do not match, and/or spurrious authentication errors.
        ```bash
        $ export MLFLOW_EXPERIMENT_NAME=/Users/[YOUR USER NAME]/[EXPERIMENT NAME]
        $ export MLFLOW_TRACKING_URI=databricks
        $ export DATABRICKS_HOST=https://[CLUSTER_ID].cloud.databricks.com
        $ export DATABRICKS_TOKEN="[ACCESS TOKEN]"
        ```

#### Train Locally and Publish to Databrick's MLFlow Tracking Server.
[Client and Tracking APIs](https://www.mlflow.org/docs/latest/tracking.html). 
- Create a new conda environment, and configure your [databricks-cli](https://docs.databricks.com/dev-tools/cli/index.html) client.
    - `$ conda create -f envs/conda.yaml`
    - `$ databricks configure`
- Train the model
    - `$ cd mlflow`
    - Publish to Databrick's tracking server
        - Here, we use mlflow to run our training routine locally, publish the resulting metrics to our configured
        Databricks account, and save our RAPIDS model.
        - Export the [required environment](#databricks-requirements) variables for MLFlow
            ```shell script
            mlflow run file:///$PWD -b local -e hyperopt \
                                    -P conda-env=$PWD/envs/conda.yaml\
                                    -P fpath=https://rapidsai-cloud-ml-sample-data.s3-us-west-2.amazonaws.com/airline_small.parquet
            ```
- Deploy your model
    - Locate the model's 'Full Path' identity. 
        - Databricks
            - Locate your saved experiment in the Databricks tracking UI at: `/Users/[YOUR USER NAME]/[EXPERIMENT NAME]`
            - Ex. `dbfs:/databricks/mlflow/[EXPERIMENT ID #]/[EXPERIMENT RUN HASH]/artifacts/`
    - Select the successful run and find the 'Full Path' element
        1. ![Example 1](images/example.png)
    - Deploy your model
        - If you have not defined the environment variables described above, this will fail to pull your model
        from Databricks.
        - `mlflow models serve -m [PATH_TO_MODEL] -p 55755`
    - From Databricks MLFlow UI
        - Locate your saved experiment in the Databricks tracking UI at: `/Users/[YOUR USER NAME]/[EXPERIMENT NAME]`
        - Ex. `dbfs:/databricks/mlflow/[EXPERIMENT ID #]/[EXPERIMENT RUN HASH]/artifacts/`
        - `mlflow models serve -m [PATH_TO_MODEL] -p 55755` 
- Query the deployed model with test data `test_call.sh` example script.
    - `bash test_call.sh`
       
## Train Models Using MLFlow with Hyperopt and RAPIDS on Databricks.
#### Currently, this approach does not support SparkTrials integration with Hyperopt.
- Define a cluster configuration, or use the [sample provided](cluster_definitions/training_cluster.json)
- Initiate Databricks job
   ```shell script
    mlflow run file:///$PWD -b databricks \
                            --backend-config=./databricks/training_cluster.json \
                            -P conda-env=[CONDA SPEC PATH OR URL] \
                            -P fpath=[DATH PATH]
   ``` 
