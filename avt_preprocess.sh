#!/bin/bash

if [ -z "$1" ]
then
  echo "Usage: $0 <CONFIG_FILE>"
  exit 1
fi

CONFIG_FILE=$1

# shellcheck disable=SC2164
cd /home/avt/github/avt_preprocess
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda run -n avt python main.py --config_file "$CONFIG_FILE"
