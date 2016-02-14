#!/bin/bash

MY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# echo 'Updating system...'
# sudo apt-get update -y
# sudo apt-get upgrade -y

echo 'Installing dependencies...'

sudo apt-get install -U -y build-essential
sudo apt-get install -U -y python3.4-dev
sudo apt-get install -U -y python3.4-venv
#sudo apt-get install -y libssl-dev
#sudo apt-get install -y libffi-dev
sudo apt-get install -U -y libxml2-dev libxslt-dev libcurl4-openssl-dev
sudo apt-get install -y python3-setuptools
sudo apt-get install -y python3-pip
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libatlas-dev
sudo apt-get install -y liblapack-dev
sudo apt-get install -y python3-numpy
sudo apt-get install -y python3-scipy

sudo pip3.4 install -U pip
sudo pip3.4 install -U virtualenv

echo 'Setting up virtualenv...'

virtualenv-3.4 --python=python3.4 --system-site-packages venv
. venv/bin/activate
pip install -U -r ${MY_DIR}/requirements.txt

deactivate

echo 'Done.'
