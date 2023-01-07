import functools
from time import time

import cv2


def timer_func(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        return result, t2 - t1
    return wrap_func


def visualise_detections(img, detections):
    work_image = img.copy()
    for det in detections:
        work_image = draw_bounding_box(work_image, det)
    return work_image


def draw_bounding_box(img, detection_obj):
    box, conf, class_name = detection_obj.get_coords(), detection_obj.conf, detection_obj.class_name
    # color = COLORS[label]
    color = (255, 0, 0)
    x1, y1, x2, y2 = list(map(round, box))

    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    img = cv2.putText(img, f'{class_name}: {conf:.4f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
    return img


def read_class_names(class_names):
    # read class names from text file
    with open(class_names, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes
