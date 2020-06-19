## Databricks Notebooks with MLFlow, RAPIDS, and Hyperopt
#### Currently, Databricks does not directly support modifying the conda environment, as a result, there are some incompatibilities with RAPIDS 0.14
### Upload RAPIDS 0.13 Init Script
1. Copy `src/rapids_install_cuml0.13_cuda10.0_ubuntu16.04.sh` onto your Databricks dbfs file system.
    1. This will become the base init script that is run at cluster start up.
    1. Example:
    ```shell script
    $ dbfs configure
       ... configure your dbfs client for your account ...
    $ dbfs cp src/rapids_install_cuml0.13_cuda10.0_ubuntu16.04.sh dbfs:/databricks/init_scripts/
    ```
   
### Create a Cluster
1. Create your cluster
    1. Select a GPU enabled Databricks runtime. Ex: 6.6 ML 
        1. Currently 'Use your own Docker container' is not available for ML instances.
    1. Select a GPU enabled worker and driver type
        1. **Note** Selected GPU must be Pascal generation or greater. p2.X is not supported.
        1. Recommended: `g4dn.xxxx` (NVIDIA T4) or `p3.xxxx` (NVIDIA V100)
    1. Select `Advanced` -> `init_scripts`
        1. For our example set `init_scripts` to `dbfs:/databricks/init_scripts/rapids_install_cuml0.13_cuda10.0_ubuntu16.04.sh'
    1. Launch your cluster
        1. At this point, you should have RAPIDS 0.13 installed in the databricks-ml-gpu conda environment, and can import cudf/cuml modules.         

## Databricks Jobs with MLFlow, RAPIDS, and Hyperopt
### MLFlow and RAPIDS
1. RAPIDS Attempts to maintain compatibility with the SKlearn API. This means that, in general, you will be able to
utilize cuML models with the MLFlow Sklearn interface, including model training, saving artifacts/models, and deploying
saved models.
    1. Ex. 
    ```python
   import mlflow
   from cuml.ensemble import RandomForestClassifier
   
   model = RandomForestClassifier()
   mlflow.sklearn.log_model(model, "cuml_model", conda_env='conda.yaml')
    ```
