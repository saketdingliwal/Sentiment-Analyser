#!/bin/bash
DICT_URL="https://owncloud.iitd.ac.in/owncloud/index.php/s/tjcJtCiEoxg3q99/download"
MODEL_URL="https://owncloud.iitd.ac.in/owncloud/index.php/s/cSp2XTc2aRT7EmL/download"
DICT=simple_dict.pkl
MODEL=simple4.pth
pip3 install --user -r requirements.txt
pip install --user -r requirements.txt
wget $MODEL_URL -O $MODEL
wget $DICT_URL -O $DICT
