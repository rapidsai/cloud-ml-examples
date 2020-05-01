#!/bin/bash
set -e

source /opt/anaconda3/etc/profile.d/conda.sh

mkdir -p /opt/anaconda3/envs/rapids_py37
wget -q --show-progress https://storage.googleapis.com/drobison-gcp-gtc-2020/rapids_py37.tar.gz
tar -xzf rapids_py37.tar.gz -C /opt/anaconda3/envs/rapids_py37

source activate rapids_py37
conda unpack

ipython kernel install --user --name=rapids_py37

source deactivate
