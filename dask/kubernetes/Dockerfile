FROM rapidsai/rapidsai-core:22.06-cuda11.5-runtime-ubuntu20.04-py3.9

# Install required package for notebook and cluster control 
RUN mamba install -n rapids -c conda-forge --freeze-installed -y kubernetes google-cloud-sdk gcsfs seaborn dask-kubernetes

# Install gke-gcloud-auth-plugin, see https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke
RUN gcloud components install gke-gcloud-auth-plugin

RUN git clone --depth 1 https://github.com/rapidsai/cloud-ml-examples.git /rapids/notebooks/cloud-ml-examples
