#!/bin/bash

CONFIG_FILE=$1

# shellcheck disable=SC2164
cd /home/avt/github/avt_preprocess
conda activate avt
python main.py --config_file "$CONFIG_FILE"
