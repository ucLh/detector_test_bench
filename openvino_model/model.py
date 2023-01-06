import sys

from openvino.model_zoo.model_api.models import SSD
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core

sys.path.append('../')
from openvino_model.config import cfg
from detector import AbstractTimedDetector, Detection
from utils import read_class_names


class OpenvinoWrapper(AbstractTimedDetector):
    def __init__(self):
        super().__init__()
        model_adapter = OpenvinoAdapter(create_core(), cfg.weights_path, device="CPU")
        self.model = SSD(model_adapter, preload=True)
        self.class_names = read_class_names(cfg.class_names)

    def preprocess(self, inputs):
        dict_data, input_meta = self.model.preprocess(inputs)
        return dict_data, input_meta

    def inference(self, dict_data, input_meta):
        return self.model.infer_sync(dict_data), input_meta

    def postprocess(self, raw_out, input_meta):
        detections = self.model.postprocess(raw_out, input_meta)

        results = []
        for det in detections:
            results.append(Detection(det.xmin, det.ymin, det.xmax, det.ymax, int(det.id), det.score,
                                     self.class_names[int(det.id)]))

        return results
