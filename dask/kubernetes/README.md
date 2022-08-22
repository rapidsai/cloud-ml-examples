# Exploring cuML algorithms with dask-kubernetes

This guide aims to showcase a joint working example of [`cuML`](https://docs.rapids.ai/api/cuml/stable/),
[`dask-kubernetes`](https://kubernetes.dask.org/en/latest/index.html) and [Kubernetes cluster](https://kubernetes.io/).

## Prerequisite

A Kubernetes cluster capable of supporting 1 dask-scheduler and 8 dask-cuda-workers.
See `spec/sched-spec.yaml` and `spec/worker-spec.yaml` for resource requirements.

Visit [`rapidsai/deployment`](docs.rapids.ai/deployment) for setting up your cluster from various
cloud service provider (CSP).

## Launch Client

`Dockerfile` contains the image capable of running the notebooks in this folder.
Build the image:

```bash
docker build -t rapids-dask-kubernetes-client:22.06 .
```

Launch the container.
The container will automatically start a jupyter server and a dask dashboard.
Expose these services to local ports.

```bash
docker run --gpus all --rm -it --shm-size=1g --ulimit memlock=-1 \
  -p 8888:8888 \
  -p 8787:8787 \
  -p 8786:8786 \
  rapids-dask-kubernetes-client:22.06
```

> **note**
> The three ports exposed here are for jupyter-lab, dask-dashboard, dask-scheduler
> respectively.

## Execute Notebooks

Enter `cloud-ml-examples/dask/kubernetes` and explore the notebooks.

- `Dask_cuML_Exploration.ipynb`,
performs performance sweep of `RandomForesetRegressor` and XGBoost over fil model.
- `Dask_cuML_Exploration_Full.ipynb`,
extended version of above and performs performance sweep of more cuML APIs.

### Setup Cluster via dask-kubernetes

The first few cells of the notebooks launches a dask-cluster on your kubernetes cluster.
The scheduler and worker pods specifications are defined in `spec/sched-spec.yaml` and `spec/worker-spec.yaml`.
The specs are loaded via `create_pod_from_yaml` and passed to `KubeCluster`,
which will create the scheduler pod and services exposing the scheduler pods.
Worker pods are then created by `scale_worker` function.
