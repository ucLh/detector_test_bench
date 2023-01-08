from abc import abstractmethod, ABC
from collections import deque
from enum import Enum
from time import time
from typing import Dict, List, Optional, TypeVar, Generic, Tuple

import numpy as np
from .config import cfg

T = TypeVar('T', int, float)


class Detection(Generic[T]):
    def __init__(self, x1: T, y1: T, x2: T, y2: T, label: int, conf: float, class_name: str):
        """
        Class that represents a single detection
        :param x1: x coordinate of the top left corner
        :param y1: y coordinate of the top left corner
        :param x2: x coordinate of the bottom right corner
        :param y2: y coordinate of the bottom right corner
        :param label: int label of the class
        :param conf: confidence of the detection
        :param class_name: string name of the class
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.label = label
        self.class_name = class_name

    def get_coords(self) -> Tuple[T, T, T, T]:
        """
        Returns the coordinates of the bounding box
        """
        return self.x1, self.y1, self.x2, self.y2


class MetricKeys(Enum):
    """
    Enum that represents the keys for the metrics used in the AbstractTimedDetector
    """
    FPS = 'fps'
    TIME_INFER = 'time_infer'
    TIME_PRE = 'time_pre'
    TIME_POST = 'time_post'


class AbstractTimedDetector(ABC):
    """
    A class that represents a timed detector. It is used to measure the time spent in the preprocessing, inference and
    postprocessing steps, also measures the overall FPS.
    """
    def __init__(self):
        self._max_time_len = cfg.max_time_len  # not to store too many measurements
        self._num_warmup_runs = cfg.num_warmup_runs
        self._metrics = {
            MetricKeys.TIME_PRE: deque(maxlen=self._max_time_len),
            MetricKeys.TIME_INFER: deque(maxlen=self._max_time_len),
            MetricKeys.TIME_POST: deque(maxlen=self._max_time_len),
            MetricKeys.FPS: deque(maxlen=self._max_time_len)
        }
        self._cur_time: Optional[Dict[MetricKeys, deque]] = None

    @abstractmethod
    def inference(self, *args, **kwargs):
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        pass

    @abstractmethod
    def postprocess(self, *args, **kwargs) -> List[Detection]:
        pass

    def __call__(self, x) -> List[Detection]:
        """
        Performs the preprocessing, inference and postprocessing steps and measures the time spent in each step
        :param x: input tensor
        :return: List of detections, List[Detection]
        """
        # Run all the steps
        t1 = time()
        preprocess_out = self.preprocess(x)
        t2 = time()
        infer_out = self.inference(*preprocess_out)
        t3 = time()
        final_out = self.postprocess(*infer_out)
        t4 = time()

        # Calculate the time spent in each step and add it to the metrics
        time_pre, time_infer, time_post, fps = t2 - t1, t3 - t2, t4 - t3, 1 / (t4 - t1)
        self._metrics[MetricKeys.TIME_PRE].append(time_pre)
        self._metrics[MetricKeys.TIME_INFER].append(time_infer)
        self._metrics[MetricKeys.TIME_POST].append(time_post)
        self._metrics[MetricKeys.FPS].append(fps)

        # Save the current time so that it can be accessed later
        self._cur_time = {MetricKeys.TIME_PRE: time_pre, MetricKeys.TIME_INFER: time_infer,
                          MetricKeys.TIME_POST: time_post, MetricKeys.FPS: fps}

        return final_out

    def get_time_stats(self) -> (Dict[MetricKeys, float], Dict[MetricKeys, float], Dict[MetricKeys, float]):
        """
        Calculates mean, median and std of the metrics
        :return: mean, median, std of the metrics
        """
        def _apply_aggregator(func):
            # Apply the aggregator function to each type of time measurements
            stats = {
                MetricKeys.TIME_PRE: func(time_pre_arr),
                MetricKeys.TIME_INFER: func(time_infer_arr),
                MetricKeys.TIME_POST: func(time_post_arr),
                MetricKeys.FPS: func(fps_arr),
            }
            return stats

        def _metrics_to_array(key):
            # convert to array and remove first warm up measurements
            res_arr = np.array(self._metrics[key])
            if len(res_arr) > self._num_warmup_runs:
                res_arr = res_arr[self._num_warmup_runs:]
            return res_arr

        time_pre_arr = _metrics_to_array(MetricKeys.TIME_PRE)
        time_infer_arr = _metrics_to_array(MetricKeys.TIME_INFER)
        time_post_arr = _metrics_to_array(MetricKeys.TIME_POST)
        fps_arr = _metrics_to_array(MetricKeys.FPS)

        mean_metrics = _apply_aggregator(np.mean)
        median_metrics = _apply_aggregator(np.median)
        std_metrics = _apply_aggregator(np.std)

        return mean_metrics, median_metrics, std_metrics

    def print_time_stats(self):
        """
        Formats and prints the time stats
        """
        mean_metrics, median_metrics, std_metrics = self.get_time_stats()

        def _format_metrics(message, metrics):
            result_string = f"{message} | Preprocessing: {metrics[MetricKeys.TIME_PRE]} | " \
                            f"Inference: {metrics[MetricKeys.TIME_INFER]} | Postprocessing {metrics[MetricKeys.TIME_POST]} | " \
                            f"FPS: {metrics[MetricKeys.FPS]}"
            return result_string

        print(_format_metrics('Mean', mean_metrics))
        print(_format_metrics('Median', median_metrics))
        print(_format_metrics('Std', std_metrics))

    def get_last_inference_time(self) -> Optional[Dict[MetricKeys, deque]]:
        return self._cur_time
