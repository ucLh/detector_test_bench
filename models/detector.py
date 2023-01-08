from collections import deque
from enum import Enum
from time import time
from typing import Dict, List, Optional, TypeVar, Generic, Tuple

import numpy as np

T = TypeVar('T', int, float)


class Detection(Generic[T]):
    def __init__(self, x1: T, y1: T, x2: T, y2: T, label: int, conf: float, class_name: str):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.label = label
        self.class_name = class_name

    def get_coords(self) -> Tuple[T, T, T, T]:
        return self.x1, self.y1, self.x2, self.y2


class MetricKeys(Enum):
    FPS = 'fps'
    TIME_INFER = 'time_infer'
    TIME_PRE = 'time_pre'
    TIME_POST = 'time_post'


class AbstractTimedDetector:
    def __init__(self):
        self.max_time_len = 10000
        self.num_warmup_runs = 1
        self.metrics = {
            MetricKeys.TIME_PRE: deque(maxlen=self.max_time_len),
            MetricKeys.TIME_INFER: deque(maxlen=self.max_time_len),
            MetricKeys.TIME_POST: deque(maxlen=self.max_time_len),
            MetricKeys.FPS: deque(maxlen=self.max_time_len)
        }
        self._cur_time: Optional[Dict[MetricKeys, deque]] = None

    def inference(self, *args, **kwargs):
        pass

    def preprocess(self, *args, **kwargs):
        pass

    def postprocess(self, *args, **kwargs) -> List[Detection]:
        pass

    def __call__(self, x) -> List[Detection]:
        t1 = time()
        preprocess_out = self.preprocess(x)
        t2 = time()
        infer_out = self.inference(*preprocess_out)
        t3 = time()
        final_out = self.postprocess(*infer_out)
        t4 = time()

        time_pre, time_infer, time_post, fps = t2 - t1, t3 - t2, t4 - t3, 1 / (t4 - t1)
        self.metrics[MetricKeys.TIME_PRE].append(time_pre)
        self.metrics[MetricKeys.TIME_INFER].append(time_infer)
        self.metrics[MetricKeys.TIME_POST].append(time_post)
        self.metrics[MetricKeys.FPS].append(fps)

        self._cur_time = {MetricKeys.TIME_PRE: time_pre, MetricKeys.TIME_INFER: time_infer,
                          MetricKeys.TIME_POST: time_post, MetricKeys.FPS: fps}

        return final_out

    def get_mean_metrics(self) -> (Dict[MetricKeys, float], Dict[MetricKeys, float], Dict[MetricKeys, float]):
        # TODO: Rename to get_time_stats
        def apply_aggregator(func):
            stats = {
                MetricKeys.TIME_PRE: func(time_pre_arr),
                MetricKeys.TIME_INFER: func(time_infer_arr),
                MetricKeys.TIME_POST: func(time_post_arr),
                MetricKeys.FPS: func(fps_arr),
            }
            return stats

        def metrics_to_array(key):
            # convert to array and remove first warm up measurements
            res_arr = np.array(self.metrics[key])
            if len(res_arr) > self.num_warmup_runs:
                res_arr = res_arr[self.num_warmup_runs:]
            return res_arr

        # convert to array and remove first measurement due to warm up
        time_pre_arr = metrics_to_array(MetricKeys.TIME_PRE)
        time_infer_arr = metrics_to_array(MetricKeys.TIME_INFER)
        time_post_arr = metrics_to_array(MetricKeys.TIME_POST)
        fps_arr = metrics_to_array(MetricKeys.FPS)

        mean_metrics = apply_aggregator(np.mean)
        median_metrics = apply_aggregator(np.median)
        std_metrics = apply_aggregator(np.std)

        return mean_metrics, median_metrics, std_metrics

    def print_mean_metrics(self):
        # TODO: Rename to print_time_stats
        mean_metrics, median_metrics, std_metrics = self.get_mean_metrics()

        def format_metrics(message, metrics):
            result_string = f"{message} | Preprocessing: {metrics[MetricKeys.TIME_PRE]} | " \
                            f"Inference: {metrics[MetricKeys.TIME_INFER]} | Postprocessing {metrics[MetricKeys.TIME_POST]} | " \
                            f"FPS: {metrics[MetricKeys.FPS]}"
            return result_string

        print(format_metrics('Mean', mean_metrics))
        print(format_metrics('Median', median_metrics))
        print(format_metrics('Std', std_metrics))

    def get_last_inference_time(self) -> Optional[Dict[MetricKeys, deque]]:
        return self._cur_time
