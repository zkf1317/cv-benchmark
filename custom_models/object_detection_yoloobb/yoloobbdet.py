import cv2
import math
import random
import numpy as np
import onnxruntime as ort

from custom_models.comm import ModelInferBackend
from custom_models.object_detection_yoloobb.tools import class_names

class RotatedBOX:
    def __init__(self, box, score, class_index):
        self.box = box
        self.score = score
        self.class_index = class_index

class YoloObbDet:
    def __init__(self, model_path, conf_thres=0.5, nms_thres=0.4, backendId = ModelInferBackend.CPU) -> None:
        self.model_path = model_path
        self.class_names = class_names
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.backendId = backendId
        self.device = self._select_device(self.backendId)

        # print(f"Loading model on {self.device}...")
        self.session_model = ort.InferenceSession(
            self.model_path,
            providers=self.device,
            sess_options=self._get_session_options()
        )

    @property
    def name(self):
        return self.__class__.__name__

    def _select_device(self, backendId):
        if backendId == ModelInferBackend.CUDA:
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        return providers

    def _get_session_options(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.intra_op_num_threads = 4
        return sess_options

    def preprocess(self, img):
        """
        Preprocess the image for inference.
        :param img: Input image.
        :return: Preprocessed image blob, original image width, and original image height.
        """
        # print("Preprocessing input image to [1, channels, input_w, input_h] format")
        height, width = img.shape[:2]
        length = max(height, width)
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = img

        input_shape = self.session_model.get_inputs()[0].shape[2:]
        input_type = self.session_model.get_inputs()[0].type
        # print(f"Input shape: {input_shape}")
        # print(f"Input type: {input_type}")

        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1 / 255, size=tuple(input_shape), swapRB=True)
        # print(f"Preprocessed image blob shape: {blob.shape}")

        if input_type == "tensor(float16)":
            blob = blob.astype(np.float16)

        return blob, image, width, height

    def infer(self, img):
        """
        Perform inference on the image.
        :param img: Input image.
        :return: Inference results.
        """
        blob, resized_image, orig_width, orig_height = self.preprocess(img)
        inputs = {self.session_model.get_inputs()[0].name: blob}
        try:
            outputs = self.session_model.run(None, inputs)
        except Exception as e:
            print(f"Inference failed: {e}")
            raise
        return self.postprocess(outputs, resized_image, orig_width, orig_height)

    def postprocess(self, outputs, resized_image, orig_width, orig_height):
        """
        Postprocess the model output.
        :param outputs: Model outputs.
        :param resized_image: Resized image used for inference.
        :param orig_width: Original image width.
        :param orig_height: Original image height.
        :return: List of RotatedBOX objects.
        """
        output_data = outputs[0]
        # print(f"Postprocessing output data with shape: {output_data.shape}")

        input_shape = self.session_model.get_inputs()[0].shape[2:]
        x_factor = resized_image.shape[1] / float(input_shape[1])
        y_factor = resized_image.shape[0] / float(input_shape[0])

        flattened_output = output_data.flatten()
        reshaped_output = np.reshape(
            flattened_output, (output_data.shape[1], output_data.shape[2])).T

        detected_boxes = []
        confidences = []
        rotated_boxes = []

        num_classes = len(self.class_names)

        for detection in reshaped_output:
            class_scores = detection[4:4 + num_classes]
            class_id = np.argmax(class_scores)
            confidence_score = class_scores[class_id]

            if confidence_score > self.conf_thres:
                cx, cy, width, height = detection[:4] * \
                    [x_factor, y_factor, x_factor, y_factor]
                angle = detection[4 + num_classes]

                if 0.5 * math.pi <= angle <= 0.75 * math.pi:
                    angle -= math.pi

                box = ((cx, cy), (width, height), angle * 180 / math.pi)
                rotated_box = RotatedBOX(box, confidence_score, class_id)

                detected_boxes.append(cv2.boundingRect(cv2.boxPoints(box)))
                rotated_boxes.append(rotated_box)
                confidences.append(confidence_score)

        nms_indices = cv2.dnn.NMSBoxes(
            detected_boxes, confidences, self.conf_thres, self.nms_thres)
        remain_boxes = [rotated_boxes[i] for i in nms_indices.flatten()]

        # print(f"Detected {len(remain_boxes)} objects after NMS")
        return remain_boxes