from easydict import EasyDict

cfg = EasyDict()
# Path to the model's weights
cfg.weights_path = 'assets/model_files/openvino_model/person-detection-0200.xml'
# Predictions confidence threshold
cfg.conf_threshold = 0.5
# Path to a text file containing class names
cfg.class_names = 'assets/model_files/openvino_model/classes.txt'
