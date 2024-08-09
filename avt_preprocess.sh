#!/bin/bash

# shellcheck disable=SC2164
cd /home/avt/github/avt_preprocess
conda activate avt
echo $1
python main.py --config_file $1
