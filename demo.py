"""
A script for visualising the detections on the images/video. Can also provide time spent on running the input
through the model.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import filetype
import numpy as np
from models.detector import AbstractTimedDetector
from models.dnn.model import DNNWrapper
from models.openvino_model.model import OpenvinoWrapper
from utils import visualise_detections


def process_one_image(image: np.ndarray, model: AbstractTimedDetector, print_time: bool) -> np.ndarray:
    """
    Process one image (visualise the detections) with the given model
    """
    detections = model(image)

    if print_time:
        time = model.get_last_inference_time()
        print(time)

    image = visualise_detections(image, detections)
    return image


def save_image(image: np.ndarray, image_path: str, output_dir: str, model_name: str):
    """
    Save image to the output directory
    :param image:
    :param image_path: original image path
    :param output_dir:
    :param model_name: model name in [openvino, dnn], used to construct the output file name
    """
    base_name = os.path.basename(image_path)
    base_name_split = base_name.split('.')
    save_path = os.path.join(output_dir, f'{base_name_split[0]}_{model_name}.{base_name_split[1]}')
    cv2.imwrite(save_path, image)


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default='assets/img',
                    help='Path to the input. Should be either a path to a single image, a path to a video, '
                         'or a path to a directory with images')
    ap.add_argument('-m', '--model', choices=['openvino', 'dnn'], default='dnn',
                    help='Model type')
    ap.add_argument('-o', '--output_dir', default='output',
                    help='Specify output directory where inference results will be saved')
    ap.add_argument('--print_time', action='store_true',
                    help='Whether to print inference time of each run')
    return ap.parse_args(argv)


def main(args: argparse.Namespace):
    # read pre-trained model
    if args.model == 'openvino':
        net = OpenvinoWrapper()
    elif args.model == 'dnn':
        net = DNNWrapper()
    else:
        raise ValueError(f'Unknown model type: {args.model}')

    # create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if os.path.exists(args.input):
        # read and process input
        if os.path.isfile(args.input):
            # single image scenario
            if filetype.is_image(args.input):
                image_path = args.input
                # read, process and save the image
                img = cv2.imread(image_path)
                out_img = process_one_image(img, net, args.print_time)
                save_image(out_img, image_path, args.output_dir, args.model)

            # video scenario
            elif filetype.is_video(args.input):
                # create input cap
                cap = cv2.VideoCapture(args.input)
                # read input cap parameters
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                # prepare output save path
                base_name = os.path.basename(args.input).split('.')[0]
                save_path = os.path.join(args.output_dir, f'{base_name}_{args.model}.avi')
                # create output video writer
                out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                      (frame_width, frame_height))
                # read and process frames, save them to the output video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        out_img = process_one_image(frame, net, args.print_time)
                        out.write(out_img)
                    else:
                        break
                cap.release()
                out.release()

        # directory with images scenario
        elif os.path.isdir(args.input):
            # find all the images in the directory
            img_names = os.listdir(args.input)
            img_paths = map(lambda x: os.path.join(args.input, x), img_names)
            img_paths = list(filter(lambda x: filetype.is_image(x), img_paths))
            # process all the images
            for image_path in img_paths:
                img = cv2.imread(image_path)
                out_img = process_one_image(img, net, args.print_time)
                save_image(out_img, image_path, args.output_dir, args.model)

        # print out time stats in the end
        net.print_time_stats()

    else:
        raise ValueError(f'Input path `{args.input}` does not exist')


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
