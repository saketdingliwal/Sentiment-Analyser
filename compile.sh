#!/bin/bash
NEG_URL="https://owncloud.iitd.ac.in/owncloud/index.php/s/BRDF7BiQSLDTQMz/download"
POS_URL="https://owncloud.iitd.ac.in/owncloud/index.php/s/ecyS4tasGsssXx6/download"
MODEL_URL="https://owncloud.iitd.ac.in/owncloud/index.php/s/r9qAnyrWwsH2a2b/download"
MODEL=model.pkl
pip3 install --user -r requirements.txt
pip install --user -r requirements.txt
python3 -m nltk.downloader punkt
python3 -m nltk.downloader stopwords
wget $NEG_URL -O "negative-words.txt"
wget $POS_URL -O "positive-words.txt"
wget $DICT_URL -O $DICT
wget $MODEL_URL -O $MODEL
