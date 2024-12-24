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

from yoloworld import YOLOWorld
from tools import detections_dog

# Valid combinations of backends
backend_target = [ModelInferBackend.CPU, ModelInferBackend.CUDA, ModelInferBackend.APPLE]

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Yolov8 inference')
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the input image. Omit for using default camera.')
    parser.add_argument('--model', '-m', type=str, default='yoloworld_pcba_v12_640.onnx',
                        help="Path to the model")
    parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: ONNXRuntime + CPU,
                        {:d}: ONNXRuntime + GPU (CUDA),
                        {:d}: ONNXRuntime + Apple Silicon
                    '''.format(*[x for x in range(len(backend_target))]))
    parser.add_argument('--confidence', default=0.5, type=float,
                        help='Class confidence')
    parser.add_argument('--nms', default=0.5, type=float,
                        help='Enter nms IOU threshold')
    parser.add_argument('--obj', default=0.5, type=float,
                        help='Enter object threshold')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Specify to save results. This flag is invalid when using camera.')
    parser.add_argument('--vis', '-v', action='store_true',
                        help='Specify to open a window for result visualization. This flag is invalid when using camera.')
    args = parser.parse_args()

    backend_id = backend_target[args.backend_target]
    model_net = YOLOWorld(model_path= args.model,
                      conf_thres=args.confidence,
                      backendId=backend_id)

    tm = cv.TickMeter()
    tm.reset()
    if args.input is not None:
        image = cv.imread(args.input)

        # Inference
        tm.start()
        results = model_net.infer(image)
        tm.stop()
        print("Inference time: {:.2f} ms".format(tm.getTimeMilli()))

        if args.vis:
            det_img = detections_dog(image, results['boxes'], results['scores'], results['labels'])
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, det_img)
            cv.waitKey(0)
