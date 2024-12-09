# Computer Vision Benchmark

Benchmarking the speed of DNN inferring different models in the opencv zoo or your own models added to the 'custom_models' directory.

Result of each model includes the time of its preprocessing, inference and postprocessing stages.

Data for benchmarking will be downloaded and loaded in [data](./data) based on given config.

## Preparation

1. Install `python >= 3.6`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Install library : `apt update && apt install -y libgl1-mesa-glx`
4. Download data for benchmarking.
    1. Download all data: `python download_data.py`
    2. Download one or more specified data: `python download_data.py object_detection`. Available names can be found in `download_data.py`.
    3. You can also download all data from https://pan.baidu.com/s/18sV8D4vXUb2xC9EG45k7bg (code: pvrw). Please place and extract data packages under [./data](./data).


## Benchmarking

**Linux**:
- Test OpenCV
```shell
# Single config
python benchmark.py --cfg ./config/face_detection_yunet.yaml

# All configs
python benchmark.py --all

# All configs but only fp32 models (--fp32, --fp16, --int8 --int8bq are available for now)
python benchmark.py --all --fp32

# All configs but exclude some of them (fill with config name keywords, not sensitive to upper/lower case, seperate with colons)
python benchmark.py --all --cfg_exclude YoutuReID
python benchmark.py --all --cfg_exclude YuNet:YoutuReID

# All configs but exclude some of the models (fill with exact model names, sensitive to upper/lower case, seperate with colons)
python benchmark.py --all --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx

# All configs with overwritten backend and target (run with --help to get available combinations)
python benchmark.py --all --cfg_overwrite_backend_target 1
```

- Test Custom Model
```shell
python benchmark_custom.py --all
```

- Test All
```shell
# CPU inference test
bash benchmark.sh CPU
```

**Windows**:
- Test OpenCV
```shell
python benchmark.py --cfg ./config/face_detection_yunet.yaml
```
- Test Custom Model
```shell
python benchmark_custom.py --cfg ./custom_config/object_detection_yolov8.yaml
```
- Test All
```shell
# CPU inference test
benchmark.bat CPU
```

## Detailed Results

Benchmark is done with latest opencv-python & opencv-contrib-python (current 4.10.0) on the following platforms. Some models are excluded because of support issues.

All models in 'models' are from [opencv zoo](https://github.com/opencv/opencv_zoo).

You also can add your own model to 'custom_models' directory, extend the benchmark on your own model very easily.

### XXX Device
TODO  
