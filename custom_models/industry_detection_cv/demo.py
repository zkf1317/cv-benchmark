import cv2
import numpy as np
import cv2 as cv
import argparse

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from custom_models.comm import ModelInferBackend

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from cv import IndustryDetection

# Valid combinations of backends
backend_target = [ModelInferBackend.CPU, ModelInferBackend.CUDA, ModelInferBackend.APPLE]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CV Function inference')
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the input image. Omit for using default camera.')
    parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: OpenCV + CPU,
                        {:d}: OpenCV + GPU (CUDA),
                        {:d}: OpenCV + Apple Silicon
                    '''.format(*[x for x in range(len(backend_target))]))
    parser.add_argument('--save', '-s', action='store_true',
                        help='Specify to save results. This flag is invalid when using camera.')
    parser.add_argument('--vis', '-v', action='store_true',
                        help='Specify to open a window for result visualization. This flag is invalid when using camera.')
    args = parser.parse_args()

    backend_id = backend_target[args.backend_target]
    model_net = IndustryDetection("", backendId=backend_id)

    tm = cv.TickMeter()
    tm.reset()
    if args.input is not None:
        image = cv.imread(args.input)

        # Inference
        tm.start()
        model_net.infer(image)
        tm.stop()
        print("Inference time: {:.2f} ms".format(tm.getTimeMilli()))