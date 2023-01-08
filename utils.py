from typing import List

import cv2
import numpy as np
from models.detector import Detection


def visualise_detections(img: np.ndarray, detections: List[Detection[int]]) -> np.ndarray:
    """
    Draw a list of detections on the input image
    :param img: input image
    :param detections:
    :return: visualised image
    """
    work_image = img.copy()
    for det in detections:
        work_image = draw_bounding_box(work_image, det)
    return work_image


def draw_bounding_box(img: np.ndarray, detection_obj: Detection[int]) -> np.ndarray:
    """
    Draw a single detection on the image: bounding box, class name and confidence
    """
    box, conf, class_name = detection_obj.get_coords(), detection_obj.conf, detection_obj.class_name
    color = (255, 0, 0)
    x1, y1, x2, y2 = list(map(round, box))

    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    img = cv2.putText(img, f'{class_name}: {conf:.4f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
    return img


def read_class_names(class_names_path: str) -> List[str]:
    """
    Read class names from text file
    """
    with open(class_names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes
