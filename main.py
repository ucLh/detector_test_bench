import os
import cv2
import argparse
import filetype
import numpy as np
from pathlib import Path

from dnn.model import DNNWrapper
from openvino_model.model import OpenvinoWrapper
from utils import visualise_detections


def process_one_image(image, model):
    detections, time = model(image, return_time=True)

    if args.print_time:
        print(time)

    image = visualise_detections(image, detections)
    return image


def save_image(image, image_path):
    base_name = os.path.basename(image_path)
    base_name_split = base_name.split('.')
    save_path = os.path.join(args.output_dir, f'{base_name_split[0]}_{args.model}.{base_name_split[1]}')
    cv2.imwrite(save_path, image)


# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=False, default='assets/videos/Double1.mp4',
                help='path to input image')
ap.add_argument('--model', choices=['openvino', 'dnn'], default='openvino',)
ap.add_argument('-o', '--output_dir', required=False, default='output',
                help='specify output directory if you want it saved')
ap.add_argument('--print_time', action='store_true',
                help='whether to print inference time')
args = ap.parse_args()

# read pre-trained model
if args.model == 'openvino':
    net = OpenvinoWrapper()
elif args.model == 'dnn':
    net = DNNWrapper()
else:
    raise ValueError(f'Unknown model type: {args.model}')

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

if os.path.exists(args.input):
    # read and process input
    if os.path.isfile(args.input):
        # single image scenario
        if filetype.is_image(args.input):
            image_path = args.input
            img = cv2.imread(image_path)
            out_img = process_one_image(img, net)
            save_image(out_img, image_path)

        # video scenario
        elif filetype.is_video(args.input):
            cap = cv2.VideoCapture(args.input)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            base_name = os.path.basename(args.input).split('.')[0]
            save_path = os.path.join(args.output_dir, f'{base_name}_{args.model}.avi')
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    out_img = process_one_image(frame, net)
                    out.write(out_img)
                else:
                    break
            cap.release()
            out.release()

    # directory with images scenario
    elif os.path.isdir(args.input):
        img_names = os.listdir(args.input)
        img_paths = map(lambda x: os.path.join(args.input, x), img_names)
        img_paths = list(filter(lambda x: filetype.is_image(x), img_paths))
        for image_path in img_paths:
            img = cv2.imread(image_path)
            out_img = process_one_image(img, net)
            save_image(out_img, image_path)

    net.print_mean_metrics()

else:
    raise ValueError(f'Input path `{args.input}` does not exist')

# image = cv2.imread(args.image)
#
# for i in range(10):
#     detections = net(image, return_time=False)
#     # print(time)
#
# print(net.metrics['time_infer'])
#
# # go through the detections remaining
# # after nms and draw bounding box
# visualise_detections(image, detections)
#
# # display output image
# cv2.imwrite("object-detection.jpg", image)
#
# # release resources
# cv2.destroyAllWindows()
