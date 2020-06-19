### Train and Publish Locally With MLFlow
#### Jupyter Notebook Workflow
[Jupyter Notebook](notebooks/rapids_mlflow_databricks_train_deploy.ipynb)

#### CLI Based Workflow
1. Create a new conda environment.
    1. `$ conda create -f envs/conda.yaml`
1. Train the model
    1. `$ cd mlflow\_project`
    1. The example project is described in [MLProject](https://www.mlflow.org/docs/latest/projects.html) file.
        1. This can be edited to allow additional command line variables, specify conda environments, and training
        parameters (see link for additional information).
    1. Publish to local tracking server
        1. Here we instruct mlflow to run our training routine locally, and publish the result's to the local file system.
        1. ```shell script
          $ mlflow run file:///$PWD -b local\
                                   -P conda-env=$PWD/envs/conda.yaml\
                                   -P fpath=https://rapidsai-cloud-ml-sample-data.s3-us-west-2.amazonaws.com/airline_small.parquet
           ```

1. Deploy your model
    1. Locate the model's 'Full Path' identity. 
        1. Local storage
            1. `$ mlflow ui`
            1. Locate the model path using the mlflow ui at localhost:5000
    1. Select the successful run and find the 'Full Path' element
        1. ![Example 1](imgs/example.png)
    1. Deploy your model
        1. `$ mlflow models serve -m [PATH_TO_MODEL] -p 55755`

1. Query the deployed model with test data `src/sample_server_query.sh` example script.
    1. `$ bash src/sample_server_query.sh`


#### CLI Based Workflow + Databricks
[Databricks MLFlow CLI](README-Databricks.md)
