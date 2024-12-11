# !/usr/bin/bash

if [ $# -eq 0 ]; then
    echo "No argument provided. Please pass an argument."
    exit 1
fi

if [ $1 == "CPU" ]; then
    python benchmark.py --all --not_print_header --cfg_overwrite_backend_target=0
    python benchmark_custom.py --all --not_print_header --cfg_overwrite_backend_target=0
elif [ $1 == "GPU" ]; then
    python benchmark.py --all --not_print_header --cfg_overwrite_backend_target=0
    python benchmark_custom.py --all --not_print_header --cfg_overwrite_backend_target=1
fi