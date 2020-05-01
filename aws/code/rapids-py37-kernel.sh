#!/bin/bash

set -e

cat > /home/ec2-user/lifecycle_script.sh <<EOF
#!/bin/bash
set -e
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh

mkdir -p /home/ec2-user/anaconda3/envs/rapids_py37
wget -q https://drobison-sagemaker-gtc-2020.s3-us-west-2.amazonaws.com/rapids_py37.tar.gz
tar -xzf rapids_py37.tar.gz -C /home/ec2-user/anaconda3/envs/rapids_py37

source activate rapids_py37
conda unpack

ipython kernel install --user --name=rapids_py37
EOF

chmod +x /home/ec2-user/lifecycle_script.sh

echo "Running: /home/ec2-user/lifecycle_script.sh"
sudo -u ec2-user -i bash "/home/ec2-user/lifecycle_script.sh"