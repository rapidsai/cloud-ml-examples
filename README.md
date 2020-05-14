# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;RAPIDS Cloud Machine Learning Services Integration</div>

RAPIDS is a suite of open-source libraries that bring GPU acceleration
to data science pipelines. Users building cloud-based hyperparameter
optimization experiments can take advantage of this acceleration
throughout their workloads to build models faster, cheaper, and more
easily on the cloud platform of their choice.

This repository provides example notebooks and "getting started" code
samples to help you integrate RAPIDS with the hyperparameter
optimization services from Azure ML, AWS Sagemaker, and Google
Cloud. The directory for each cloud contains a step-by-step guide to
launch an example hyperparameter optimization job.

Each example job will use RAPIDS
[cuDF](https://github.com/rapidsai/cudf) to load and preprocess 20
million rows of airline arrival and departure data and build a model
to predict whether or not a flight will arrive on time. It
demonstrates both [cuML](https://github.com/rapidsai/cuml) Random
Forests and GPU-accelerated XGBoost modeling.

## Microsoft Azure ML
[Azure ML Step-by-step.](https://github.com/rapidsai/cloud-ml-examples/blob/master/azure/README.md "Azure Deployment Guide")

## AWS SageMaker
[AWS SageMaker Step-by-step.](https://gitlab-master.nvidia.com/drobison/aws-sagemaker-gtc-2020/tree/master/docs/aws/README.md "AWS Deployment Guide")

## Google Cloud AI Platform
[Google Cloud AI Step-by-step](https://gitlab-master.nvidia.com/drobison/aws-sagemaker-gtc-2020/tree/master/gcp/README.md "GCP Deployment Guide")

## Bring Your Own Cloud (Dask and Ray)

In addition to public cloud HPO options, the respository also includes
"BYOC" sample notebooks that can be run on the public cloud or private
infrastructure of your choice. These leverage [Ray Tune](ray) or
[Dask-ML](dask) for distributed infrastructure, while demonstrating
the same airline classifier HPO workload.