from typing import List, Tuple

import cv2
import numpy as np
from models.detector import AbstractTimedDetector, Detection
from utils import read_class_names

from .config import cfg


class DNNWrapper(AbstractTimedDetector):
    def __init__(self):
        super().__init__()
        self._weights_path = cfg.weights_path
        self._yolo_config = cfg.yolo_config
        self._conf_threshold = cfg.conf_threshold
        self._nms_threshold = cfg.nms_threshold
        self._class_names = read_class_names(cfg.class_names)
        self._net = cv2.dnn.readNet(self._weights_path, self._yolo_config)
        self._output_layers = self._get_output_layers()  # Get output layers names from the graph once

    def _get_output_layers(self) -> List[str]:
        layer_names = self._net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self._net.getUnconnectedOutLayers().reshape(-1)]

        return output_layers

    def preprocess(self, image: np.ndarray) -> Tuple[int, int]:
        original_height, original_width = image.shape[:2]
        scale = 0.00392  # 1 / 255
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        self._net.setInput(blob)
        return original_height, original_width

    def inference(self, original_height: int, original_width: int) -> (List[np.ndarray], int, int):
        outs = self._net.forward(self._output_layers)
        return outs, original_height, original_width

    def postprocess(self, outs: List[np.ndarray], original_height: int, original_width: int) -> List[Detection]:
        class_ids = []
        confidences = []
        boxes = []
        results = []

        # process box coords and filter out low confidence predictions
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self._conf_threshold:
                    center_x = int(detection[0] * original_width)
                    center_y = int(detection[1] * original_height)
                    w = int(detection[2] * original_width)
                    h = int(detection[3] * original_height)
                    x1 = round(center_x - w / 2)
                    y1 = round(center_y - h / 2)
                    x2 = round(center_x + w / 2)
                    y2 = round(center_y + h / 2)
                    class_ids.append(int(class_id))
                    confidences.append(float(confidence))
                    boxes.append([x1, y1, x2, y2])

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self._conf_threshold, self._nms_threshold)
        indices = indices.reshape(-1)

        for i in indices:
            results.append(Detection(*boxes[i], label=class_ids[i], conf=confidences[i],
                                     class_name=self._class_names[class_ids[i]]))

        return results
