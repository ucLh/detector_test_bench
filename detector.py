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


class AbstractDetector:
    def inference(self, *args, **kwargs):
        pass

    def preprocess(self, *args, **kwargs):
        pass

    def postprocess(self, *args, **kwargs):
        pass

    def __call__(self, x):
        preprocess_out = self.preprocess(x)
        infer_out = self.inference(*preprocess_out)
        final_out = self.postprocess(*infer_out)
        # print(time_pre, time_infer, time_post)
        return final_out






















