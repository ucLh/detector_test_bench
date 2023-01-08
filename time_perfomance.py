import argparse
import sys

import cv2
import filetype

from models.openvino_model.model import OpenvinoWrapper
from models.dnn.model import DNNWrapper


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=False, default='assets/img/persons.jpg',
                    help='path to input image')
    ap.add_argument('--model', choices=['openvino', 'dnn'], default='dnn',)
    ap.add_argument('--num_iters', required=False, default=1000, type=int,
                    help='number of iterations to run')
    ap.add_argument('--print_time', action='store_true',
                    help='whether to print inference time')
    return ap.parse_args(argv)

def main(args):
    if args.model == 'openvino':
        net = OpenvinoWrapper()
    elif args.model == 'dnn':
        net = DNNWrapper()
    else:
        raise ValueError(f'Unknown model type: {args.model}')

    if filetype.is_image(args.image):
        image = cv2.imread(args.image)
    else:
        raise ValueError(f'Input `{args.input}` is not an image')

    for i in range(args.num_iters):
        detections, time = net(image, return_time=True)
        if args.print_time:
            print(time)

    net.print_mean_metrics()


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))