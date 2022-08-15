# Exploring cuML algorithms with dask-kubernetes in Google Cloud Platform (GCP)

This guide aims to showcase a joint working example of [`cuML`](https://docs.rapids.ai/api/cuml/stable/),
[`dask-kubernetes`](https://kubernetes.dask.org/en/latest/index.html) and [Google Cloud Platform](https://cloud.google.com/).

## Prerequisite

Install gcloud in your environment:
- [gcloud] (https://cloud.google.com/sdk/docs/install)

Login to your gcloud account with

```bash
gcloud init
```

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
docker run --gpus all --rm -it --shm-size=1g --ulimit memlock=-1 -p 8888:8888 -p 8787:8787 -p 8786:8786 -v $HOME/.config/gcloud:/root/.config/gcloud cloud-ml-examples rapids-dask-kubernetes-client:22.06
```

> **Note**
> The config files for `gcloud` is bind to the container for reuse,
> reconfigure if necessary.

## Setup Cluster

Let's setup the cluster to supply the resources required by the examples.
In total,
we will use at most 1 dask-scheduler pod and 8 dask-cuda-workers.
See `spec/sched-spec.yaml` and `spec/worker-spec.yaml` for resource requirements.

```bash
gcloud container clusters create rapids-dask-kubernetes \
  --accelerator type=nvidia-tesla-a100,count=8 \
  --zone us-central1-c \
  --num-nodes=1 \
  --machine-type a2-highgpu-8g
```

> **Note**
> GPU availability in different region and zone may vary. See latest list of available
> nodes at https://cloud.google.com/compute/docs/regions-zones#available

Install Nvidia driver:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

## Execute Notebooks

Enter `cloud-ml-examples/dask/kubernetes` and explore the notebooks.

- `Dask_cuML_Exploration.ipynb`, performs performance sweep of `RandomForesetRegressor` and XGBoost over fil model.
- `Dask_cuML_Exploration_Full.ipynb`, extended version of above and performs performance sweep of more cuML APIs.
- `Mortgage_Data_Conversion.ipynb`, converts the `Morgage` dataset from csv to parquet.
