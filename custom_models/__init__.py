from pathlib import Path
import glob
import os

from .object_detection_yolov8.yolov8 import YoloV8
from .industry_detection_cv.cv import IndustryDetection

class ModuleRegistery:
    def __init__(self, name):
        self._name = name
        self._dict = dict()

        self._base_path = Path(__file__).parent

    def get(self, key):
        '''
        Returns a tuple with:
        - a module handler,
        - a list of model file paths
        '''
        return self._dict[key]

    def register(self, item):
        '''
        Registers given module handler along with paths of model files
        '''
        # search for model files
        model_dir = str(self._base_path / item.__module__.split(".")[1])
        fp32_model_paths = []
        fp16_model_paths = []
        int8_model_paths = []
        int8bq_model_paths = []
        # onnx
        ret_onnx = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
        if "object_tracking" in item.__module__:
            # object tracking models usually have multiple parts
            fp32_model_paths = [ret_onnx]
        else:
            for r in ret_onnx:
                if "int8" in r:
                    int8_model_paths.append([r])
                elif "fp16" in r: # exclude fp16 for now
                    fp16_model_paths.append([r])
                elif "blocked" in r:
                    int8bq_model_paths.append([r])
                else:
                    fp32_model_paths.append([r])
        # caffe
        ret_caffemodel = sorted(glob.glob(os.path.join(model_dir, "*.caffemodel")))
        ret_prototxt = sorted(glob.glob(os.path.join(model_dir, "*.prototxt")))
        caffe_models = []
        for caffemodel, prototxt in zip(ret_caffemodel, ret_prototxt):
            caffe_models += [prototxt, caffemodel]
        if caffe_models:
            fp32_model_paths.append(caffe_models)

        all_model_paths = dict(
            fp32=fp32_model_paths,
            fp16=fp16_model_paths,
            int8=int8_model_paths,
            int8bq=int8bq_model_paths
        )

        self._dict[item.__name__] = (item, all_model_paths)

CustomModels = ModuleRegistery('Models')
CustomModels.register(YoloV8)
CustomModels.register(IndustryDetection)