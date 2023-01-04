import cv2
import openvino.model_zoo.model_api.models.utils

# import model wrapper class
from detector import OpenvinoWrapper
from utils import draw_bounding_box


# read input image using opencv
input_data = cv2.imread("openvino/persons.jpg")

# define the path to mobilenet-ssd model in IR format
model_path = "intel/person-detection-0200/FP16/person-detection-0200.xml"

ssd_model = OpenvinoWrapper(model_path)

# apply input preprocessing, sync inference, model output postprocessing
detections = ssd_model(input_data)
for det in detections:
    input_data = draw_bounding_box(input_data, det)

cv2.imwrite("openvino/object-detection.jpg", input_data)
