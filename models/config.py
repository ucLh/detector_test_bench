from easydict import EasyDict

cfg = EasyDict()
# Maximum number of measurements stored
cfg.max_time_len = 10000
# Number of warm up runs to skip when calculating statistics
cfg.num_warmup_runs = 1
