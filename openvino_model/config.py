from easydict import EasyDict


cfg = EasyDict()
cfg.weights_path = 'intel/person-detection-0200/FP16-INT8/person-detection-0200.xml'
cfg.class_names = 'openvino_model/classes.txt'

# class OpenvinoConfig:
#     _weights_path = 'intel/person-detection-0200/FP16/person-detection-0200.xml'
#
#     @property
#     def weights_path(self) -> str:
#         return self._weights_path