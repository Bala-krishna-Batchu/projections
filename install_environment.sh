# This file will create a new conda environment with python 3.8 & install requirements.txt to the env
## To run -> source install_environment.sh. 
### Use /home/jupyter/runenv for all the LP related development and runs.

conda deactivate 
conda env remove -p /home/jupyter/runenv -y
conda env list 
conda create --prefix=/home/jupyter/runenv python=3.8 -y
conda env list 
conda activate /home/jupyter/runenv
DL_ANACONDA_ENV_HOME="/home/jupyter/runenv"
conda install ipykernel -y
python -m ipykernel install --prefix "${DL_ANACONDA_ENV_HOME}" --name runenv --display-name runenv
pip install --upgrade pip --index-url https://nexus-ha.cvshealth.com:9443/repository/pypi-proxy/simple
pip install --upgrade -r requirements.txt --index-url https://nexus-ha.cvshealth.com:9443/repository/pypi-proxy/simple || pip install --upgrade -r requirements.txt