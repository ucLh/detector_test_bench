import cv2
# import model wrapper class
from openvino.model_zoo.model_api.models import SSD
# import inference adapter and helper for runtime setup
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core


# read input image using opencv
input_data = cv2.imread("persons.jpg")

# define the path to mobilenet-ssd model in IR format
model_path = "../intel/person-detection-0200/FP16/person-detection-0200.xml"

# create adapter for OpenVINO™ runtime, pass the model path
model_adapter = OpenvinoAdapter(create_core(), model_path, device="CPU")

# create model API wrapper for SSD architecture
# preload=True loads the model on CPU inside the adapter
ssd_model = SSD(model_adapter, preload=True)

# apply input preprocessing, sync inference, model output postprocessing
results = ssd_model(input_data)
print(results)
