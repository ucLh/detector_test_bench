import cv2
import argparse
import numpy as np

from dnn.model import DNNWrapper
from openvino_model.model import OpenvinoWrapper
from utils import draw_bounding_box


# function to draw bounding box on the detected object with class name
def draw_bounding_box_old(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False, default='img/persons.jpg',
                help='path to input image')
# ap.add_argument('-c', '--config', required=False, default='dnn/yolov4-tiny.cfg',
#                 help='path to yolo config file')
# ap.add_argument('-w', '--weights', required=False, default='dnn/yolov4-tiny.weights',
#                 help='path to yolo pre-trained weights')
ap.add_argument('--model', choices=['openvino', 'dnn'], default='dnn')
ap.add_argument('-cl', '--classes', required=False, default='dnn/classes.txt',
                help='path to text file containing class names')
args = ap.parse_args()


# read input image
image = cv2.imread(args.image)

# read class names from text file
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
if args.model == 'openvino':
    net = OpenvinoWrapper()
elif args.model == 'dnn':
    net = DNNWrapper()

for i in range(10):
    detections, time = net(image, return_time=True)
    print(time)

net.print_mean_metrics()
print(net.metrics['time_infer'])

# go through the detections remaining
# after nms and draw bounding box
for i, det in enumerate(detections):
    image = draw_bounding_box(image, det)

# display output image
cv2.imwrite("object-detection.jpg", image)

# release resources
cv2.destroyAllWindows()
