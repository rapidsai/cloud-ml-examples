## **Pack and Deploy Conda Environments for RAPIDS on Google Cloud Platform (GCP)**
This section describes the process required to:
1. Package and deploy a RAPIDS conda environment via helper script
1. Package and deploy a RAPIDS conda environment manually
    1. Initialize a RAPIDS conda environment.
    1. Package the environment using conda-pack.
    1. Unpack into a second second environment with conda-unpack.
1. Unpack an existing conda environment via cloud storage.

### **Package and Deploy Using the Helper Script**
#### Pack Environment
1. `common/code/create_packed_conda_env`
```bash
... processing ...
Packing conda environment
Collecting packages...
Packing environment at '[CONDA ENV]/rapids0.19_py3.7' to 'rapids0.19_py3.7.tar.gz'
[########################################] | 100% Completed |  1min 51.1s
```

#### Unpack Environment on Target System
1. Copy your environment tarball (rapids_py37.tar.gz) to the target system
1. Unpack in desired environment
    1. `common/code/create_packed_conda_env --action unpack` 
1. Alternatively, the environment can be manually unpacked as
    ```bash
      CONDA_ENV="rapids0.19_py3.7"
      TARBALL="$CONDA_ENV.tar.gz"
      UNPACK_TO="$CONDA_ENV"
   
      mkdir -p "$UNPACK_TO"
      tar -xzf $TARBALL -C "$UNPACK_TO"
      source "$UNPACK_TO/bin/activate"
      conda-unpack
      python -m ipykernel install --user --name $CONDA_ENV
    ```

### **Package and Deploy Manually**
#### Pack Environment
1. Create a clean conda environment to work from
    1. `conda create --name=rapids_env python=3.7`
1. Install conda-pack
    1. `conda install -c conda-forge conda-pack`
1. Install RAPIDS
    1. Select the package level to install from [RAPIDS.ai](rapids.ai/start.html)
    1. Ex. For a full install on Ubuntu 18.04, with CUDA 11.0
        ```bash
        conda install -c rapidsai -c nvidia -c conda-forge 
            -c defaults rapids=0.19 python=3.7 cudatoolkit=11.0
        ```
1. Pack your environment
    1. `conda-pack -n rapids_env -o rapids_py37.tar.gz`
    
#### Unpack Environment on Target System 
1. Copy your environment tarball (rapids_py37.tar.gz) to the target system
1. Extract the tarball to the desired environment
    1. Ex. Local
    ```bash
        mkdir -p ./rapids_env
        tar -xzf rapids_py37.tar.gz -C ./rapids_env
        source ./rapids_env/bin/activate
    ```
   1. Ex. Anaconda
    ```bash
        mkdir -p $HOME/anaconda3/envs/rapids_py37
        tar -xzf rapids_py37.tar.gz -C $HOME/anaconda3/envs/rapids_py37
        source $HOME/anaconda3/envs/rapids_py37/bin/activate 
    ```
1. Cleanup environment prefixes
    1. `conda-unpack`

### **Unpack an Existing Environment Via Cloud Storage**
#### Unpacking on a Target Environment
1. Upload your packed conda environment to a GCP storage bucket.
1. Pull and unpack your environment manually or via script as
    ```bash
      GCP_STORAGE="https://storage.googleapis.com/$YOUR_BUCKET/rapids_py37.tar.gz"
      CONDA_ENV="rapids0.19_py3.7"
      TARBALL="$CONDA_ENV.tar.gz"
      UNPACK_TO="$CONDA_ENV"
   
      wget -q --show-progress $GCP_STORAGE
      mkdir -p "$UNPACK_TO"
      tar -xzf $TARBALL -C "$UNPACK_TO"
      source "$UNPACK_TO/bin/activate"
      conda-unpack
      python -m ipykernel install --user --name $CONDA_ENV
    ```