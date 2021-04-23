### Train and Publish Locally With MLFlow
#### Jupyter Notebook Workflow
[Jupyter Notebook](notebooks/rapids_mlflow_databricks_train_deploy.ipynb)

#### To reproduce this workflow, utilizing Databricks MLFlow tracking server, see:
- [Databricks MLFlow CLI](README-Databricks.md)

#### CLI Based Workflow
- Create a new conda environment.
    - `$ conda create -f envs/conda.yaml`
- Train the model
    - `$ cd mlflow`
    - MLflow project configuration is described in our [MLProject](https://www.mlflow.org/docs/latest/projects.html) file.
        - This can be edited to allow additional command line variables, specify conda environments, and training
        parameters (see link for additional information).
    - Publish to local tracking server
        - Here use mlflow to run our training routine locally, and publish the results to the local file system.
        - In your shell, run:
            ```shell script
            # Downlad the file
            wget -N https://rapidsai-cloud-ml-sample-data.s3-us-west-2.amazonaws.com/airline_small.parquet
            # Launch the job
            mlflow run . -b local -e hyperopt \
                     -P conda-env=$PWD/envs/conda.yaml \
                     -P fpath=airline_small.parquet
            ```

- Deploy your model
    - Locate the model's 'Full Path'
        - `mlflow ui`
        - Locate the model path using the mlflow ui at localhost:5000
    - Select the successful run and find the 'Full Path' element
        - ![](images/example.png)
    - Deploy your model
        - `$ mlflow models serve -m [PATH_TO_MODEL] -p 55755`

1. Query the deployed model with test data `src/sample_server_query.sh` example script.
    1. `bash src/sample_server_query.sh`

