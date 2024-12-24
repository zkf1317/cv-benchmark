import onnxruntime as ort
import numpy as np
import time
import cv2
import numbers
import logging
import math
import os
import sys
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from custom_models.comm import ModelInferBackend

class YOLOWorld:
    def __init__(self, model_path, conf_thres=0.3, iou_thres=0.5, backendId = ModelInferBackend.CPU):
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.means = [0., 0., 0.]
        self.stds = [255., 255., 255.]
        self.threshold = conf_thres
        self.iou_threshold = iou_thres
        self.backendId = backendId
        self.pre_topk = 30000
        self.keep_topk = 1000

        self.init_model(model_path)

    @property
    def name(self):
        return self.__class__.__name__

    def init_model(self, model_path):
        if self.backendId == ModelInferBackend.CUDA:
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get model info
        self.get_input_details()
        self.get_output_details()
        # print(f"input name: {self.input_names}, output name: {self.output_names}")

    def imnormalize(self, img, mean, std, to_rgb=True):
        """Inplace normalize an image with mean and std.

        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.

        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept uint8
        #assert img.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def impad(self, img,
            *,
            shape=None,
            padding=None,
            pad_val=(114,114,114),
            padding_mode='constant'):
        assert (shape is not None) ^ (padding is not None)
        if shape is not None:
            #padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])
            padding_h = shape[0] - img.shape[0]
            padding_w = shape[1] - img.shape[1]
            top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
                round(padding_w // 2 - 0.1))
            bottom_padding = padding_h - top_padding
            right_padding = padding_w - left_padding
            padding = (left_padding, top_padding, right_padding, bottom_padding)

        # check pad_val
        if isinstance(pad_val, tuple):
            assert len(pad_val) == img.shape[-1]
        elif not isinstance(pad_val, numbers.Number):
            raise TypeError('pad_val must be a int or a tuple. '
                            f'But received {type(pad_val)}')

        # check padding
        if isinstance(padding, tuple) and len(padding) in [2, 4]:
            if len(padding) == 2:
                padding = (padding[0], padding[1], padding[0], padding[1])
        elif isinstance(padding, numbers.Number):
            padding = (padding, padding, padding, padding)
        else:
            raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                                f'But received {padding}')

        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        border_type = {
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT_101,
            'symmetric': cv2.BORDER_REFLECT
        }
        # logging.error("padding...")
        # logging.error(padding)

        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            border_type[padding_mode],
            value=pad_val)

        return img, padding
    
    def preprocess(self, image, input_shape):
        h, w, _ = image.shape
        resize_height = input_shape[0]
        resize_width = input_shape[1]
        img_scale = (resize_width, resize_height)
        max_long_edge = max(img_scale)
        max_short_edge = min(img_scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
        scale_w = int(w * float(scale_factor) + 0.5)
        scale_h = int(h * float(scale_factor) + 0.5)
        image = cv2.resize(image, (scale_w, scale_h))
        image = np.asarray(image).astype(np.float32)

        pad_h, pad_w = resize_height, resize_width

        image, padding = self.impad(image, shape=(pad_h, pad_w), pad_val=(114,114,114))
        image = np.asarray(image).astype(np.float32)

        means = np.array(self.means, dtype=np.float32)
        stds = np.array(self.stds, dtype=np.float32)
        image = self.imnormalize(image, means, stds)
        image = np.asarray(image).astype(np.float32)

        image = np.transpose(image, [2, 0, 1])

        image = image[np.newaxis, :, :]
        return image, scale_factor, padding, (h, w)

    def nms_numpy(self, boxes, scores, iou_threshold):
        """
        Pure numpy NMS implementation.
        :param boxes: [N, 4] where each box is represented as [x1, y1, x2, y2]
        :param scores: [N] where each score corresponds to a box
        :param iou_threshold: IoU threshold for NMS
        :return: indices of boxes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]  # Sort boxes by score in descending order

        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            indices = np.where(iou <= iou_threshold)[0]
            order = order[indices + 1]

        return keep
    
    def postprocess_nms_numpy(self, output_data, scale_factor, padding, image_shape, iou_threshold, score_threshold, pre_topk, keep_topk):
        boxes = output_data[0]  # Shape: [batch_size, num_boxes, 4]
        scores = np.max(output_data[1], axis=2)  # Shape: [batch_size, num_boxes]
        labels = np.argmax(output_data[1], axis=2)  # Shape: [batch_size, num_boxes]
        det = np.concatenate([boxes, scores[..., None], labels[..., None]], axis=-1)
        det = det[0]  # Process first image in batch

        # Filter by score threshold
        val_idxs = det[:, 4] > score_threshold
        det = det[val_idxs]

        # Sort by score in descending order
        score_sort = np.argsort(det[:, 4])[::-1]
        det = det[score_sort]

        if len(det) > pre_topk:
            det = det[:pre_topk]

        # Apply NMS per class
        result_boxes = []
        result_scores = []
        result_labels = []

        for label in np.unique(det[:, 5]):
            class_mask = det[:, 5] == label
            class_boxes = det[class_mask][:, :4]
            class_scores = det[class_mask][:, 4]
            
            # Apply NMS for the current class
            keep_idxs = self.nms_numpy(class_boxes, class_scores, iou_threshold)

            # Limit number of boxes to keep_topk
            keep_idxs = keep_idxs[:keep_topk] if len(keep_idxs) > keep_topk else keep_idxs

            if len(keep_idxs) > 0:
                result_boxes.append(class_boxes[keep_idxs])
                result_scores.append(class_scores[keep_idxs])
                result_labels.extend([label] * len(keep_idxs))

        # Only concatenate if result_boxes is not empty
        if len(result_boxes) > 0:
            result_boxes = np.concatenate(result_boxes, axis=0)
            result_scores = np.concatenate(result_scores, axis=0)

            image_h, image_w = image_shape
            final_boxes = []

            for i in range(result_boxes.shape[0]):
                final_boxes.append([
                    max((result_boxes[i][0] - padding[0]) / scale_factor, 0),
                    max((result_boxes[i][1] - padding[1]) / scale_factor, 0),
                    min((result_boxes[i][2] - padding[0]) / scale_factor, image_w - 1),
                    min((result_boxes[i][3] - padding[1]) / scale_factor, image_h - 1)
                ])

            result = {'boxes': final_boxes, 'scores': result_scores.tolist(), 'labels': result_labels}
        else:
            result = {'boxes': [], 'scores': [], 'labels': []}

        return result
    
    def infer(self, image):
        image, scale_factor, padding, image_shape = self.preprocess(image, (self.input_height, self.input_width))
        output_data = self.session.run(self.output_names, {self.input_names[0]: image})
        result = self.postprocess_nms_numpy(output_data, scale_factor, padding, image_shape, self.iou_threshold, self.threshold, self.pre_topk, self.keep_topk)
        return result

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        # print(model_inputs)

        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        # print(model_outputs)

        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

