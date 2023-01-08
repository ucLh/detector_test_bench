import cv2
import numpy as np
from models.detector import AbstractTimedDetector, Detection
from utils import read_class_names

from .config import cfg


class DNNWrapper(AbstractTimedDetector):
    def __init__(self):
        super().__init__()
        self.weights_path = cfg.weights_path
        self.yolo_config = cfg.yolo_config
        self.conf_threshold = cfg.conf_threshold
        self.nms_threshold = cfg.nms_threshold
        self.class_names = read_class_names(cfg.class_names)
        self.net = cv2.dnn.readNet(self.weights_path, self.yolo_config)
        self.output_layers = self._get_output_layers()

    def _get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().reshape(-1)]

        return output_layers

    def preprocess(self, image):
        original_height, original_width = image.shape[:2]
        scale = 0.00392  # 1 / 255
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        return original_height, original_width

    def inference(self, original_height, original_width):
        outs = self.net.forward(self.output_layers)
        return outs, original_height, original_width

    def postprocess(self, outs, original_height, original_width):
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
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * original_width)
                    center_y = int(detection[1] * original_height)
                    w = int(detection[2] * original_width)
                    h = int(detection[3] * original_height)
                    x1 = center_x - w / 2
                    y1 = center_y - h / 2
                    x2 = center_x + w / 2
                    y2 = center_y + h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x1, y1, x2, y2])

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        indices = indices.reshape(-1)

        # boxes = np.array(boxes)[indices]
        # class_ids = np.array(class_ids)[indices]
        # confidences = np.array(confidences)[indices]

        for i in indices:
            results.append(Detection(*boxes[i], class_ids[i], confidences[i], self.class_names[class_ids[i]]))

        return results
