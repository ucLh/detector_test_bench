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


def draw_bounding_box(img, detection_obj):
    box, conf, label = detection_obj.get_coords(), detection_obj.conf, detection_obj.label
    # color = COLORS[label]
    color = (255, 0, 0)
    x1, y1, x2, y2 = list(map(round, box))

    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    img = cv2.putText(img, f'{str(label)}: {conf:.4f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
    return img


# def timer_func(func):
#     t1 = time()
#     result = func(*args, **kwargs)
#     t2 = time()
#     return result, t2 - t1
