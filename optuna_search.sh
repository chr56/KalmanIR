#!/usr/bin/env bash

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <config_path> <cuda_device_id>"
  exit 1
fi

config_path="$1"
config_name="${config_path##*/}"
log_file="logs_optuna_search_${config_name}.log"

cuda_device="$2"

export CUDA_VISIBLE_DEVICES="$cuda_device"

echo "Device: CUDA ${cuda_device}"
echo "Config: ${config_path}"
echo "Log:    ${log_file}"

nohup python basicsr/optuna_search.py -opt "$config_path" > "$log_file" 2>&1 &

echo "Starting..."
