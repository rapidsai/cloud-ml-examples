# RAPIDS Hyperparameter Optimization (HPO) with Dask ML

[Dask-ML](https://ml.dask.org/) provides machine learning utilities
built on top of the scalable Dask platform. Dask already offers
[first-class integration with RAPIDS](https://rapids.ai/dask.html),
and Dask-ML is no exception.

The Dask-ML [hyperparameter search
tools](https://ml.dask.org/hyper-parameter-search.html) make it easy
to take advantage of grid search, randomized search, or hyperband HPO
algorithms. It particularly excels at incorporating cross-validation
into the HPO process for more stable accuracy estimates and at
allowing intelligent reuse of intermediate results from the Dask task
graph.

## RAPIDS + Dask-ML sample notebooks

This sample notebook shows how to use Ray Tune to optimize XGBoost and
cuML Random Forest classifiers over a large dataset of airline arrival
times. By design, it is very similar to the RAPIDS examples provided
for other cloud and bring-your-own-cloud HPO offerings.

You need both Jupyter and RAPIDS 0.13 or later installed to begin. See
https://rapids.ai/start.html for instructions. We recommend using 0.14
nightly packages for the latest updates. Dask-ML can be installed via conda or pip, following the instructions from: https://ml.dask.org/install.html.

## RAPIDS + Dask-Kubernetes sample notebooks
Dask_cuML_Exploration and Dask_cuML_Exploration_Full provide a template
for launching a dask cluster on top of your kubernetes environment, loading
the NYC-Taxi dataset, and generating performance metrics for the available
cuML Multi-Node Multi-GPU (MNMG) algorithms in your environment.

See: [Dask-Kubernetes](https://kubernetes.dask.org/en/latest/) and cuML's
[API documentation](https://docs.rapids.ai/api/cuml/stable/api.html#multi-node-multi-gpu-algorithms) for additional information.

   
