import torch
import os
import time
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Any, Optional, List

from custom_models.comm import ModelInferBackend

@dataclass
class GoldenFingerItem:
    golden_finger: Any
    golden_finger_box: Any

class IndustryDetection:
    def __init__(self, model_path, backendId = ModelInferBackend.CPU) -> None:
        self.backendId = backendId

    @property
    def name(self):
        return self.__class__.__name__

    def infer(self, img):
        gold_finger_begin = np.array([8, 67, 104])  # hsv
        gold_finger_end = np.array([33, 255, 255])  # hsv

        # filter
        blur = cv2.bilateralFilter(img, 10, 100, 15)

        # Binary thresholding for a bright yellow mask
        hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower_bound = gold_finger_begin
        upper_bound = gold_finger_end
        binary = cv2.inRange(hsv_image, lower_bound, upper_bound)
        # self.vis_binary(binary)

        # Find the connected areas
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # self.vis_connected_area(binary, num_labels, labels, stats, centroids)
        golden_finger_infos: List[GoldenFingerItem] = []

        for idx, label in enumerate(range(1, num_labels)):
            self.cut_single_finger(label, labels, img, golden_finger_infos)

        return golden_finger_infos

    def cut(self, image, mask):
        # 计算组件的最小外接矩形
        x, y, w, h = cv2.boundingRect(mask)

        # 裁剪图像和遮罩到组件的大小
        component_cropped = image[y:y + h, x:x + w]
        mask_cropped = mask[y:y + h, x:x + w]

        # 使用遮罩来只保留组件的像素
        isolated_component = cv2.bitwise_and(component_cropped, component_cropped, mask=mask_cropped)

        return isolated_component, [x, y, w, h]

    def cut_single_finger(self, label, labels, image, golden_finger_infos):
        mask = (labels == label).astype("uint8") * 255

        # 膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        mask = cv2.dilate(mask, kernel, 1)

        isolated_component, locate = self.cut(image, mask=mask)
        gfi = GoldenFingerItem(golden_finger=isolated_component, golden_finger_box=locate)

        golden_finger_infos.append(gfi)

    def vis_binary(self, img):
        cv2.namedWindow('binary', cv2.WINDOW_NORMAL)
        cv2.imshow('binary', img)

    def vis_connected_area(self, img, num_labels, labels, stats, centroids):
        output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(1, num_labels):  # From 1, because 0 is background
            # 获取每个区域的边界框信息
            x, y, w, h, area = stats[i]
            # 绘制矩形框
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 绘制质心
            cx, cy = centroids[i]
            cv2.circle(output_image, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        # 显示图像
        cv2.namedWindow('Connected Components', cv2.WINDOW_NORMAL)
        cv2.imshow("Connected Components", output_image)
        cv2.waitKey(0)
