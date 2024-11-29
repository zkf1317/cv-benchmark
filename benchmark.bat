@echo off
if "%1"=="" (
    echo No arguments provided.
    exit
) else (
    echo Use Device: %1
)

if "%1"=="CPU" (
    python.exe .\benchmark.py --all --not_print_header --cfg_overwrite_backend_target=0
    python.exe .\benchmark_custom.py --all --not_print_header --cfg_overwrite_backend_target=0
)
