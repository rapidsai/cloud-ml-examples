#!/usr/bin/env bash

set -x
set -e

/databricks/python/bin/python -V
. /databricks/conda/etc/profile.d/conda.sh
conda activate /databricks/python

INSTALL_FILE="/opt/rapids_initialized.log"
if [[ -f "$INSTALL_FILE" ]]; then
    TEST=$(cat "$INSTALL_FILE")

    if (( $TEST == 1 )); then
        echo "Node was previously configured. Exiting."
        exit 0
    fi
fi

cat > rapids0.13_cuda10.0_ubuntu16.04.yml <<EOF
name: databricks-ml-gpu
channels:
  - rapidsai
  - nvidia
  - conda-forge
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=1_llvm
  - arrow-cpp=0.15.0=py37h090bef1_2
  - bokeh=2.0.1=py37hc8dfbb8_0
  - boost-cpp=1.70.0=h8e57a91_2
  - brotli=1.0.7=he1b5a44_1002
  - bzip2=1.0.8=h516909a_2
  - c-ares=1.15.0=h516909a_1001
  - ca-certificates=2020.4.5.1=hecc5488_0
  - certifi=2020.4.5.1=py37hc8dfbb8_0
  - click=7.1.2=pyh9f0ad1d_0
  - cloudpickle=1.4.1=py_0
  - cudatoolkit=10.0.130=0
  - cudf=0.13.0=py37_0
  - cudnn=7.6.0=cuda10.0_0
  - cuml=0.13.0=cuda10.0_py37_0
  - cupy=7.5.0=py37h658377b_0
  - cytoolz=0.10.1=py37h516909a_0
  - dask=2.17.2=py_0
  - dask-core=2.17.2=py_0
  - dask-cudf=0.13.0=py37_0
  - distributed=2.17.0=py37hc8dfbb8_0
  - dlpack=0.2=he1b5a44_1
  - double-conversion=3.1.5=he1b5a44_2
  - fastavro=0.23.4=py37h8f50634_0
  - fastrlock=0.4=py37h3340039_1001
  - freetype=2.10.2=he06d7ca_0
  - fsspec=0.6.3=py_0
  - gflags=2.2.2=he1b5a44_1002
  - glog=0.4.0=h49b9bf7_3
  - grpc-cpp=1.23.0=h18db393_0
  - heapdict=1.0.1=py_0
  - icu=64.2=he1b5a44_1
  - jinja2=2.11.2=pyh9f0ad1d_0
  - joblib=0.15.1=py_0
  - jpeg=9c=h14c3975_1001
  - ld_impl_linux-64=2.33.1=h53a641e_7
  - libblas=3.8.0=16_openblas
  - libcblas=3.8.0=16_openblas
  - libcudf=0.13.0=cuda10.0_0
  - libcuml=0.13.0=cuda10.0_0
  - libcumlprims=0.13.0=cuda10.0_0
  - libedit=3.1.20181209=hc058e9b_0
  - libevent=2.1.10=h72c5cf5_0
  - libffi=3.3=he6710b0_1
  - libgcc-ng=9.2.0=h24d8f2e_2
  - libhwloc=2.1.0=h3c4fd83_0
  - libiconv=1.15=h516909a_1006
  - liblapack=3.8.0=16_openblas
  - libllvm8=8.0.1=hc9558a2_0
  - libnvstrings=0.13.0=cuda10.0_0
  - libopenblas=0.3.9=h5ec1e0e_0
  - libpng=1.6.37=hed695b0_1
  - libprotobuf=3.8.0=h8b12597_0
  - librmm=0.13.0=cuda10.0_0
  - libstdcxx-ng=9.1.0=hdf63c60_0
  - libtiff=4.1.0=hfc65ed5_0
  - libxml2=2.9.10=hee79883_0
  - llvm-openmp=10.0.0=hc9558a2_0
  - llvmlite=0.32.0=py37h5202443_0
  - locket=0.2.0=py_2
  - lz4-c=1.8.3=he1b5a44_1001
  - markupsafe=1.1.1=py37h8f50634_1
  - msgpack-python=1.0.0=py37h99015e2_1
  - nccl=2.6.4.1=hd6f8bf8_0
  - ncurses=6.2=he6710b0_1
  - numba
  - numpy=1.17.5=py37h95a1406_0
  - nvstrings=0.13.0=py37_0
  - olefile=0.46=py_0
  - openssl=1.1.1g=h516909a_0
  - packaging=20.4=pyh9f0ad1d_0
  - pandas=0.25.3=py37hb3f55d8_0
  - parquet-cpp=1.5.1=2
  - partd=1.1.0=py_0
  - pillow=5.3.0=py37h00a061d_1000
  - pip=20.0.2=py37_3
  - psutil=5.7.0=py37h8f50634_1
  - pyarrow=0.15.0=py37h8b68381_1
  - pyparsing=2.4.7=pyh9f0ad1d_0
  - python=3.7.7=hcff3b4d_5
  - python-dateutil=2.8.1=py_0
  - python_abi=3.7=1_cp37m
  - pytz=2020.1=pyh9f0ad1d_0
  - pyyaml=5.3.1=py37h8f50634_0
  - re2=2020.04.01=he1b5a44_0
  - readline=8.0=h7b6447c_0
  - rmm=0.13.0=py37_0
  - setuptools=46.4.0=py37_0
  - six=1.15.0=pyh9f0ad1d_0
  - snappy=1.1.8=he1b5a44_1
  - sortedcontainers=2.1.0=py_0
  - sqlite=3.31.1=h62c20be_1
  - tblib=1.6.0=py_0
  - thrift-cpp=0.12.0=hf3afdfd_1004
  - tk=8.6.8=hbc83047_0
  - toolz=0.10.0=py_0
  - tornado=6.0.4=py37h8f50634_1
  - typing_extensions=3.7.4.2=py_0
  - ucx=1.7.0+g9d06c3a=cuda10.0_0
  - ucx-py=0.13.0+g9d06c3a=py37_0
  - uriparser=0.9.3=he1b5a44_1
  - wheel=0.34.2=py37_0
  - xz=5.2.5=h7b6447c_0
  - yaml=0.2.4=h516909a_0
  - zict=2.0.0=py_0
  - zlib=1.2.11=h7b6447c_3
  - zstd=1.4.3=h3b9ef0a_0
prefix: /databricks/conda/envs/databricks-ml-gpu
EOF

time conda env update --prefix /databricks/conda/envs/databricks-ml-gpu --file rapids0.13_cuda10.0_ubuntu16.04.yml -vv
time conda install numba=0.48

echo "1" > $INSTALL_FILE 
