from easydict import EasyDict

cfg = EasyDict()
cfg.weights_path = 'assets/model_files/openvino_model/person-detection-0200.xml'
cfg.conf_threshold = 0.5
cfg.class_names = 'assets/model_files/openvino_model/classes.txt'
