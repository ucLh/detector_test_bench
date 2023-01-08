from easydict import EasyDict

cfg = EasyDict()
# Path to the model's weights
cfg.weights_path = 'assets/model_files/dnn/yolov4-tiny.weights'
# Path to the model's yolo config from darknet
cfg.yolo_config = 'assets/model_files/dnn/yolov4-tiny.cfg'
# Predictions confidence threshold
cfg.conf_threshold = 0.5
# Non-maximum suppression IoU threshold
cfg.nms_threshold = 0.4
# Path to a text file containing class names
cfg.class_names = 'assets/model_files/dnn/classes.txt'
