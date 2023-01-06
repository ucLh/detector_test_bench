from collections import deque
from time import time

import numpy as np


class Detection:
    def __init__(self, x1, y1, x2, y2, label, conf):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.label = label

    def get_coords(self):
        return self.x1, self.y1, self.x2, self.y2


class AbstractTimedDetector:
    def __init__(self):
        self.max_time_len = 10000
        self.num_warmup_runs = 1
        self.metrics = {
            'time_pre': deque(maxlen=self.max_time_len),
            'time_infer': deque(maxlen=self.max_time_len),
            'time_post': deque(maxlen=self.max_time_len),
            'fps': deque(maxlen=self.max_time_len)
        }

    def inference(self, *args, **kwargs):
        pass

    def preprocess(self, *args, **kwargs):
        pass

    def postprocess(self, *args, **kwargs):
        pass

    def __call__(self, x):
        t1 = time()
        preprocess_out = self.preprocess(x)
        t2 = time()
        infer_out = self.inference(*preprocess_out)
        t3 = time()
        final_out = self.postprocess(*infer_out)
        t4 = time()

        time_pre, time_infer, time_post, fps = t2 - t1, t3 - t2, t4 - t3, 1 / (t4 - t1)
        self.metrics['time_pre'].append(time_pre)
        self.metrics['time_infer'].append(time_infer)
        self.metrics['time_post'].append(time_post)
        self.metrics['fps'].append(fps)

        return final_out

    def get_mean_metrics(self):
        def apply_aggregator(func):
            stats = {
                'time_pre': func(time_pre_arr),
                'time_infer': func(time_infer_arr),
                'time_post': func(time_post_arr),
                'fps': func(fps_arr),
            }
            return stats

        assert len(self.metrics['fps']) > self.num_warmup_runs, \
            f'More than {self.num_warmup_runs} run required in order to get mean metrics'

        # convert to array and remove first measurement due to warm up
        time_pre_arr = np.array(self.metrics['time_pre'])[1:]
        time_infer_arr = np.array(self.metrics['time_infer'])[1:]
        time_post_arr = np.array(self.metrics['time_post'])[1:]
        fps_arr = np.array(self.metrics['fps'])[1:]

        mean_metrics = apply_aggregator(np.mean)
        median_metrics = apply_aggregator(np.median)
        std_metrics = apply_aggregator(np.std)

        return mean_metrics, median_metrics, std_metrics

    def print_mean_metrics(self):
        mean_metrics, median_metrics, std_metrics = self.get_mean_metrics()

        def format_metrics(message, metrics):
            result_string = f"{message} | Preprocessing: {metrics['time_pre']} | " \
                            f"Inference: {metrics['time_infer']} | Postprocessing {metrics['time_post']} | " \
                            f"FPS: {metrics['fps']}"
            return result_string

        print(format_metrics('Mean', mean_metrics))
        print(format_metrics('Median', median_metrics))
        print(format_metrics('Std', std_metrics))
