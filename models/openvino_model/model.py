from typing import Dict, List, Tuple

import numpy as np
import openvino.model_zoo.model_api.models.utils

from models.detector import AbstractTimedDetector, Detection
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_zoo.model_api.models import SSD
from utils import read_class_names

from .config import cfg


class OpenvinoWrapper(AbstractTimedDetector):
    """
    A wrapper for the OpenVINO person-detection-0200 model
    """
    def __init__(self):
        super().__init__()
        model_adapter = OpenvinoAdapter(create_core(), cfg.weights_path, device="CPU")
        self._model = SSD(model_adapter, preload=True, configuration={'confidence_threshold': cfg.conf_threshold})
        self._class_names = read_class_names(cfg.class_names)

    def preprocess(self, inputs: np.ndarray) -> (Dict[str, np.ndarray], Dict[str, Tuple[int, ...]]):
        dict_data, input_meta = self._model.preprocess(inputs)
        return dict_data, input_meta

    def inference(self, dict_data: Dict[str, np.ndarray], input_meta: Dict[str, Tuple[int, ...]]) -> \
            (Dict[str, np.ndarray], Dict[str, Tuple[int, ...]]):
        out = self._model.infer_sync(dict_data)
        return out, input_meta

    def postprocess(self, raw_out: Dict[str, np.ndarray], input_meta: Dict[str, Tuple[int, ...]]) -> List[Detection]:
        detections = self._model.postprocess(raw_out, input_meta)

        results = []
        for det in detections:
            results.append(Detection(det.xmin, det.ymin, det.xmax, det.ymax, int(det.id), det.score,
                                     self._class_names[int(det.id)]))

        return results
