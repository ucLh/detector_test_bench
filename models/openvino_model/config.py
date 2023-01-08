from easydict import EasyDict

cfg = EasyDict()
cfg.weights_path = 'assets/model_files/openvino_model/person-detection-0200.xml'
cfg.class_names = 'assets/model_files/openvino_model/classes.txt'

# class OpenvinoConfig:
#     _weights_path = 'intel/person-detection-0200/FP16/person-detection-0200.xml'
#
#     @property
#     def weights_path(self) -> str:
#         return self._weights_path