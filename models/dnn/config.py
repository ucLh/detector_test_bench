from easydict import EasyDict


cfg = EasyDict()
cfg.weights_path = 'assets/model_files/dnn/yolov4-tiny.weights'
cfg.yolo_config = 'assets/model_files/dnn/yolov4-tiny.cfg'
cfg.conf_threshold = 0.5
cfg.nms_threshold = 0.4
cfg.class_names = 'assets/model_files/dnn/classes.txt'
