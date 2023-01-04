from easydict import EasyDict


cfg = EasyDict()
cfg.weights_path = 'dnn/yolov4-tiny.weights'
cfg.yolo_config = 'dnn/yolov4-tiny.cfg'
cfg.conf_threshold = 0.5
cfg.nms_threshold = 0.4
