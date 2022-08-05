FROM rapidsai/rapidsai-core-nightly:22.08-cuda11.5-runtime-ubuntu20.04-py3.9 

# Install required package for notebook and kubectl 
RUN mamba install -n rapids -c conda-forge --freeze-installed -y kubernetes google-cloud-sdk gcsfs seaborn dask-kubernetes

# Install kubectl autocompletion (optional)
RUN apt-get update && apt-get install bash-completion && source /etc/profile.d/bash_completion.sh
RUN source <(kubectl completion bash) 

# Install gke-gcloud-auth-plugin, see https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke
RUN gcloud components install gke-gcloud-auth-plugin

RUN git clone --depth 1 --branch main https://github.com/isVoid/cloud-ml-examples.git && cd cloud-ml-examples/dask/kubernetes
