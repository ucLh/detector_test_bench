import cv2
import numpy as np
from openvino.model_zoo.model_api.models import SSD
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core

from utils import timer_func


class Detection:
    def __init__(self, x1, y1, x2, y2, label, conf):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.label = label

    def get_coords(self):
        return self.x1, self.y1, self.x2, self.y2


class AbstractDetector:
    def inference(self, *args, **kwargs):
        pass

    def preprocess(self, *args, **kwargs):
        pass

    def postprocess(self, *args, **kwargs):
        pass

    def __call__(self, x):
        preprocess_out = self.preprocess(x)
        infer_out = self.inference(*preprocess_out)
        final_out = self.postprocess(*infer_out)
        # print(time_pre, time_infer, time_post)
        return final_out


class OpenvinoWrapper(AbstractDetector):
    def __init__(self, weights_path):
        model_adapter = OpenvinoAdapter(create_core(), weights_path, device="CPU")
        self.model = SSD(model_adapter, preload=True)

    def preprocess(self, inputs):
        dict_data, input_meta = self.model.preprocess(inputs)
        return dict_data, input_meta

    def inference(self, dict_data, input_meta):
        return self.model.infer_sync(dict_data), input_meta

    def postprocess(self, raw_out, input_meta):
        detections = self.model.postprocess(raw_out, input_meta)

        results = []
        for det in detections:
            results.append(Detection(det.xmin, det.ymin, det.xmax, det.ymax, int(det.id), det.score))

        return results


class DNNWrapper(AbstractDetector):
    def __init__(self, weights_path, config_path, conf_threshold=0.5, nms_threshold=0.4):
        self.weights_path = weights_path
        self.config = config_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.output_layers = self._get_output_layers()

    def _get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

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

        # boxes = np.array(boxes)[indices]
        # class_ids = np.array(class_ids)[indices]
        # confidences = np.array(confidences)[indices]

        for i in indices:
            results.append(Detection(*boxes[i], class_ids[i], confidences[i]))

        return results



















