# RAPIDS on Databricks

This directory contains sample notebooks for running RAPIDS on Databricks.

The `rapids_intro.ipynb` notebook has been tested with the latest RAPIDS version (0.19) by building a custom container, and contains basic examples to get started with cuDF and cuML.

The `rapids_airline_hyperopt.ipynb` example walks through the optimization of a random forest model using cuML and hyperopt. It includes init scripts to install an earlier version of RAPIDS (0.13) on DataBricks ML Runtime.

## 1. Use a custome image on Databricks

## Build the RAPIDS container

```console
$ docker build --tag <username>/rapids_databricks:latest --build-arg RAPIDS_IMAGE=rapidsai/rapidsai-core:22.06-cuda11.0-base-ubuntu18.04-py3.8 ./docker
```

Push this image to a Docker registry (DockerHub, Amazon ECR or Azure ACR).

## Configure and create a cluster

* Create your cluster:
    1. Select a standard Databricks runtime. In this example 8.2 version, since we're using a container with CUDA 11.
        *  This needs to be a Databricks runtime version that supports Databricks Container Services.
    2. Select "Use your own Docker container".
    3. In the Docker Image URL field, enter the image that you created above.
    4. Select a GPU enabled worker and driver type.
        * **Note** Selected GPU must be Pascal generation or greater.
    5. Create and launch your cluster.

## Launching the notebook example

1. Upload the `rapids_intro.ipynb` notebook to your workspace.
2. Execute the cells to import cuDF and cuML, and walk through simple examples on the GPU.

## 2. Use an init script on Databricks

**The example below has been tested with an earlier version of RAPIDS (0.13). To use the latest version of RAPIDS, follow the steps mentioned above.**

### Upload RAPIDS 0.13 Init Script to DBFS
* Copy `src/rapids_install_cuml0.13_cuda10.0_ubuntu16.04.sh` onto your Databricks dbfs file system.
    * This will become the base init script that is run at cluster start up.
    * Example:
    ```shell script
    $ dbfs configure
       ... configure your dbfs client for your account ...
    $ dbfs cp src/rapids_install_cuml0.13_cuda10.0_ubuntu16.04.sh dbfs:/databricks/init_scripts/
    ```

### Create and Configure a Cluster
* Create your cluster:
    1. Select a GPU enabled Databricks runtime. Ex: 6.6 ML
        * Currently 'Use your own Docker container' is not available for ML instances.
    2. Select a GPU enabled worker and driver type
        * **Note** Selected GPU must be Pascal generation or greater. p2.X is not supported.
        * Recommended: `g4dn.xxxx` (NVIDIA T4) or `p3.xxxx` (NVIDIA V100) for AWS users
    3. Select `Advanced` -> `init_scripts`
        * Add an init scripts with the location `dbfs:/databricks/init_scripts/rapids_instal_cuml0.13_cuda10.0_ubuntu16.04.sh`
* Launch your cluster
  * At this point, you should have RAPIDS 0.13 installed in the databricks-ml-gpu conda environment, and can import cudf/cuml modules.

![Setting up init script](imgs/init_script_config.png)

### Launching the notebook

1. Upload the `rapids_airline_hyperopt.ipynb` notebook to your workspace.
2. Uncomment the "data download" cell and configure it to point to a path of your choice for data download. By default, it will use a smaller (200k row) dataset. This executes fast but doesn't demonstrate the full speedups possible with larger datasets.
3. Execute all of the cells to launch your hyperopt job.
4. Optionally, check out stats in the runs page and Experiment UI.


## More on Integrating Databricks Jobs with MLFlow and RAPIDS

You can find more detail in [this blog post on MLFlow + RAPIDS](https://medium.com/rapids-ai/managing-and-deploying-high-performance-machine-learning-models-on-gpus-with-rapids-and-mlflow-753b6fcaf75a).
